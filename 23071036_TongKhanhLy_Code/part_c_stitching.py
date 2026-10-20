"""
Part C – Image Stitching (own photos)
=====================================
Pipeline
  1. features   : SIFT (float descriptor, L2) vs ORB and AKAZE (binary descriptors, Hamming)
  2. matching   : kNN (k=2) + Lowe ratio test, then RANSAC keeps only geometrically consistent
                  matches -> the match-line figures show raw vs cleaned matches
  3. homography : H estimated with RANSAC for every consecutive pair (x_i ~ H x_{i+1});
                  RANSAC thresholds 1 / 3 / 5 / 10 px are compared
  4. projection : (a) planar   – warp every image with its chained homography
                  (b) cylindrical – focal length f is recovered from the homographies, images are
                      projected onto a cylinder where a camera rotation becomes a translation,
                      so the panorama stays FLAT (no bow-tie stretching); the global tilt is
                      removed and the result is cropped to the largest rectangle with no black
  5. blending   : overwrite vs feather vs multi-band (Laplacian pyramid)

Usage
    put your 4 overlapping photos in data/own_photos/part_c/ (any names – the order is found automatically)
    python part_c_stitching.py
    python part_c_stitching.py --images a.jpg b.jpg c.jpg d.jpg
"""
import argparse
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np

from metrics import (black_border_ratio, inlier_ratio, overlap_rmse, reprojection_errors,
                     reprojection_rmse, straightness)
from utils import DATA_DIR, bgr2rgb, imread, out_dir, resize_max, save_fig, timer, write_table

RATIO = 0.75          # Lowe ratio test
RANSAC_THR = 3.0      # px (main pipeline)
RANSAC_THRESHOLDS = (1.0, 3.0, 5.0, 10.0)
DETECTORS = ("SIFT", "ORB", "AKAZE")
MAX_SIDE = 1200       # photos are downscaled to this size (speed); change if you want more detail


# ============================================================================ features & matching
def detect(img, method, mask=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "SIFT":       # DoG extrema + 128-D gradient histograms
        det = cv2.SIFT_create(nfeatures=6000)
    elif method == "ORB":      # oriented FAST + rotated BRIEF (256 bit)
        det = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    else:                      # AKAZE: nonlinear scale space + binary M-LDB descriptor
        det = cv2.AKAZE_create(threshold=0.0005)
    return det.detectAndCompute(gray, mask)


def match(d_query, d_train, method, ratio=RATIO):
    """kNN + ratio test: keep m only if dist(m) < ratio · dist(2nd best) (rejects ambiguous matches)."""
    # exact brute-force search (deterministic): L2 for SIFT's float descriptor, Hamming for binary ones
    matcher = cv2.BFMatcher(cv2.NORM_L2 if method == "SIFT" else cv2.NORM_HAMMING)
    knn = matcher.knnMatch(d_query, d_train, k=2)
    return [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def pair_points(kq, kt, matches):
    return (np.float32([kq[m.queryIdx].pt for m in matches]),
            np.float32([kt[m.trainIdx].pt for m in matches]))


# ============================================================================ focal length
def focal_from_homography(H, w, h):
    """Szeliski & Shum (1997) closed form (same as OpenCV's detail::focalsFromHomography):
    for a purely rotating camera H = K R K⁻¹, so f can be read off H (in centred coordinates)."""
    T = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], np.float64)
    Hc = np.linalg.inv(T) @ H @ T
    Hc /= Hc[2, 2]
    h_ = Hc.ravel()
    fs = []
    # f1
    d1, d2 = h_[6] * h_[7], (h_[7] - h_[6]) * (h_[7] + h_[6])
    if abs(d1) > 1e-12 and abs(d2) > 1e-12:
        v1 = -(h_[0] * h_[1] + h_[3] * h_[4]) / d1
        v2 = (h_[0] ** 2 + h_[3] ** 2 - h_[1] ** 2 - h_[4] ** 2) / d2
        v1, v2 = max(v1, v2), min(v1, v2)
        if v1 > 0 and v2 > 0:
            fs.append(np.sqrt(v1 if abs(d1) > abs(d2) else v2))
        elif v1 > 0:
            fs.append(np.sqrt(v1))
    # f0
    d1 = h_[0] * h_[3] + h_[1] * h_[4]
    d2 = h_[0] ** 2 + h_[1] ** 2 - h_[3] ** 2 - h_[4] ** 2
    if abs(d1) > 1e-12 and abs(d2) > 1e-12:
        v1, v2 = -h_[2] * h_[5] / d1, (h_[5] ** 2 - h_[2] ** 2) / d2
        v1, v2 = max(v1, v2), min(v1, v2)
        if v1 > 0 and v2 > 0:
            fs.append(np.sqrt(v1 if abs(d1) > abs(d2) else v2))
        elif v1 > 0:
            fs.append(np.sqrt(v1))
    return float(np.sqrt(np.prod(fs))) if len(fs) == 2 else (fs[0] if fs else None)


def exif_focal_px(path, work_width):
    """Focal length in pixels from the photo's EXIF (35-mm equivalent focal length):
    f_px = f_35mm / 36 mm × long side (px), rescaled to the working resolution. None if absent."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            f35 = im.getexif().get_ifd(0x8769).get(41989)
            w0, h0 = im.size
        if not f35:
            return None
        return float(f35) / 36.0 * max(w0, h0) * (work_width / w0)
    except Exception:
        return None


def estimate_focal(Hs_pair, w, h):
    cands = [focal_from_homography(H, w, h) for H in Hs_pair]
    cands = [f for f in cands if f is not None and 0.3 * max(w, h) < f < 5 * max(w, h)]
    if cands:
        return float(np.median(cands)), "estimated from homographies"
    return 0.75 * max(w, h), "fallback 0.75·max(w,h) (typical phone main camera)"


# ============================================================================ warping helpers
def cylindrical_warp(img, f):
    """Project an image onto a cylinder of radius f:  θ = (u−W/2)/f,  x = f·tanθ + cx,
    y = (v−cy)/cosθ + cy.  Returns the warped image and its validity mask."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    W = int(2 * f * np.arctan(cx / f))
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(h, dtype=np.float32))
    theta = (u - W / 2.0) / f
    map_x = (f * np.tan(theta) + cx).astype(np.float32)
    map_y = ((v - cy) / np.cos(theta) + cy).astype(np.float32)
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.remap(np.full((h, w), 255, np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT)
    return warped, mask


def to3x3(A):
    return np.vstack([A, [0, 0, 1]]) if A.shape == (2, 3) else A


def chain_to_reference(pair_T, n, ref):
    """pair_T[i] maps image i+1 -> image i. Returns T_i mapping image i -> reference image."""
    Ts = [None] * n
    Ts[ref] = np.eye(3)
    for i in range(ref + 1, n):
        Ts[i] = Ts[i - 1] @ pair_T[i - 1]
    for i in range(ref - 1, -1, -1):
        Ts[i] = Ts[i + 1] @ np.linalg.inv(pair_T[i])
    return [T / T[2, 2] for T in Ts]


def warp_all(images, masks_in, Ts, max_pixels=80e6):
    corners = []
    for img, T in zip(images, Ts):
        h, w = img.shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(c, T).reshape(-1, 2))
    allc = np.vstack(corners)
    x0, y0 = np.floor(allc.min(0)).astype(int)
    x1, y1 = np.ceil(allc.max(0)).astype(int)
    if (x1 - x0) * (y1 - y0) > max_pixels:
        raise RuntimeError("canvas too large – the planar projection exploded (field of view too wide)")
    off = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    size = (x1 - x0, y1 - y0)
    warped, masks = [], []
    for img, m, T in zip(images, masks_in, Ts):
        warped.append(cv2.warpPerspective(img, off @ T, size, flags=cv2.INTER_LINEAR))
        wm = cv2.warpPerspective(m, off @ T, size, flags=cv2.INTER_NEAREST)
        masks.append(cv2.erode(wm, np.ones((5, 5), np.uint8)))   # drop interpolated black rims
    return warped, masks


# ============================================================================ blending
def blend_overwrite(warped, masks):
    pano = np.zeros_like(warped[0])
    for img, m in zip(warped, masks):
        pano[m > 0] = img[m > 0]
    return pano


def _dist_weights(masks):
    return [cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) for m in masks]


def blend_feather(warped, masks):
    """Weighted average, weight = distance to the image border -> linear cross-fade in overlaps."""
    ws = _dist_weights(masks)
    acc = sum(img.astype(np.float32) * w[..., None] for img, w in zip(warped, ws))
    return np.clip(acc / np.maximum(sum(ws), 1e-6)[..., None], 0, 255).astype(np.uint8)


def blend_multiband(warped, masks, levels=5):
    """Burt & Adelson (1983): blend each Laplacian-pyramid band with a Gaussian-blurred seam mask.
    Low frequencies mix over a wide zone (no exposure step), high frequencies over a narrow zone
    (no ghosting)."""
    ws = _dist_weights(masks)
    idx = np.argmax(np.stack(ws), axis=0)
    any_mask = np.max(np.stack(masks), axis=0) > 0
    seams = [((idx == i) & any_mask).astype(np.float32) for i in range(len(warped))]
    h, w = warped[0].shape[:2]
    ph, pw = [int(np.ceil(s / 2 ** levels) * 2 ** levels) for s in (h, w)]
    pad = lambda a: cv2.copyMakeBorder(a, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT)  # noqa: E731
    acc, wsum = None, None
    for img, sm in zip(warped, seams):
        gw, gi = [pad(sm)], [pad(img.astype(np.float32))]
        for _ in range(levels):
            gw.append(cv2.pyrDown(gw[-1]))
            gi.append(cv2.pyrDown(gi[-1]))
        lap = [gi[k] - cv2.pyrUp(gi[k + 1], dstsize=gi[k].shape[1::-1]) for k in range(levels)] + [gi[-1]]
        contrib = [l * g[..., None] for l, g in zip(lap, gw)]
        acc = contrib if acc is None else [a + b for a, b in zip(acc, contrib)]
        wsum = gw if wsum is None else [a + b for a, b in zip(wsum, gw)]
    bands = [a / np.maximum(s, 1e-6)[..., None] for a, s in zip(acc, wsum)]
    res = bands[-1]
    for k in range(levels - 1, -1, -1):
        res = cv2.pyrUp(res, dstsize=bands[k].shape[1::-1]) + bands[k]
    res = res[:h, :w]
    res[~any_mask] = 0
    return np.clip(res, 0, 255).astype(np.uint8)


BLENDERS = {"Overwrite": blend_overwrite, "Feather": blend_feather, "Multi-band": blend_multiband}


# ============================================================================ flatness helpers
def largest_valid_rectangle(valid):
    """Greedy crop: repeatedly drop the border row/column with the most empty pixels until the
    rectangle contains no empty pixel. Returns (y0, y1, x0, x1)."""
    ys, xs = np.where(valid)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    while y1 > y0 and x1 > x0:
        sub = valid[y0:y1 + 1, x0:x1 + 1]
        bad = {"top": 1 - sub[0].mean(), "bottom": 1 - sub[-1].mean(),
               "left": 1 - sub[:, 0].mean(), "right": 1 - sub[:, -1].mean()}
        side = max(bad, key=bad.get)
        if bad[side] == 0:
            break
        if side == "top":
            y0 += 1
        elif side == "bottom":
            y1 -= 1
        elif side == "left":
            x0 += 1
        else:
            x1 -= 1
    return y0, y1 + 1, x0, x1 + 1


def straighten(Ts, sizes):
    """Remove the global tilt: fit a line through the image centres in the panorama and rotate
    all transforms so that this line becomes horizontal."""
    centres = np.array([(T @ np.array([w / 2, h / 2, 1.0]))[:2] / (T @ np.array([w / 2, h / 2, 1.0]))[2]
                        for T, (w, h) in zip(Ts, sizes)])
    if len(centres) < 2:
        return Ts, 0.0
    slope = np.polyfit(centres[:, 0], centres[:, 1], 1)[0]
    ang = np.degrees(np.arctan(slope))
    R = to3x3(cv2.getRotationMatrix2D(tuple(centres.mean(0)), ang, 1.0))
    return [R @ T for T in Ts], float(ang)


# ============================================================================ pipeline
def pairwise(images, method, model, masks=None):
    """Match consecutive images. model = 'homography' (8 DoF) or 'similarity' (4 DoF)."""
    masks = masks or [None] * len(images)
    feats = [detect(im, method, cv2.erode(m, np.ones((15, 15), np.uint8)) if m is not None else None)
             for im, m in zip(images, masks)]
    pairs = []
    for i in range(len(images) - 1):
        (kq, dq), (kt, dt) = feats[i + 1], feats[i]
        good = match(dq, dt, method)
        src, dst = pair_points(kq, kt, good)                      # image i+1 -> image i
        if len(good) < 8:
            raise RuntimeError(f"only {len(good)} matches between images {i + 1} and {i + 2} – "
                               "check the order / overlap of the photos")
        if model == "homography":
            T, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THR, maxIters=5000, confidence=0.999)
        else:
            A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                  ransacReprojThreshold=RANSAC_THR, maxIters=5000)
            T = to3x3(A)
        inl = mask.ravel().astype(bool)
        pairs.append(dict(T=T, inl=inl, good=good, kq=kq, kt=kt, src=src, dst=dst, n_kp=(len(kt), len(kq))))
    return pairs


def build_panorama(images, method, projection, focal=None):
    """Returns dict with warped images, masks, transforms, per-pair statistics and focal info."""
    n = len(images)
    ref = (n - 1) // 2
    sizes = [im.shape[1::-1] for im in images]
    t = {}
    with timer(t, "total"):
        # homographies on the original images (always needed: they are the main deliverable and
        # they give us the focal length)
        hpairs = pairwise(images, method, "homography")
        info = {"focal": None, "focal_src": "-", "tilt_deg": 0.0}
        if projection == "planar":
            pairs, imgs, msks = hpairs, images, [np.full(im.shape[:2], 255, np.uint8) for im in images]
            Ts = chain_to_reference([p["T"] for p in pairs], n, ref)
        else:
            h, w = images[0].shape[:2]
            f_h, src = estimate_focal([p["T"] for p in hpairs], w, h)
            f = focal if focal else f_h               # EXIF focal is preferred when available
            info.update(focal=f, focal_src="EXIF" if focal else src, focal_from_H=f_h)
            cyl = [cylindrical_warp(im, f) for im in images]
            imgs, msks = [c[0] for c in cyl], [c[1] for c in cyl]
            pairs = pairwise(imgs, method, "similarity", msks)
            Ts = chain_to_reference([p["T"] for p in pairs], n, ref)
            Ts, info["tilt_deg"] = straighten(Ts, [im.shape[1::-1] for im in imgs])
        warped, masks = warp_all(imgs, msks, Ts)
        raw = blend_multiband(warped, masks)                 # blending is part of the stitching time
        final, kept = finish(raw, masks)
    stats = []
    for i, (p, hp) in enumerate(zip(pairs, hpairs)):
        stats.append({"Detector": method, "Projection": projection, "Pair": f"{i + 1}-{i + 2}",
                      "Keypoints": f"{hp['n_kp'][0]}/{hp['n_kp'][1]}", "Ratio_matches": len(hp["good"]),
                      "H_inliers": int(hp["inl"].sum()), "H_inlier_%": inlier_ratio(hp["inl"]),
                      "H_reproj_RMSE_px": reprojection_rmse(hp["T"], hp["src"][hp["inl"]], hp["dst"][hp["inl"]])})
    return dict(warped=warped, masks=masks, Ts=Ts, hpairs=hpairs, stats=stats, info=info,
                time_ms=t["total"], sizes=sizes, raw=raw, final=final, kept=kept)


def finish(pano_raw, masks):
    valid = np.max(np.stack(masks), axis=0) > 0
    y0, y1, x0, x1 = largest_valid_rectangle(valid)
    return pano_raw[y0:y1, x0:x1], 100.0 * (y1 - y0) * (x1 - x0) / valid.sum()


# ============================================================================ visualisation
def spread_subset(points, n_max=60, grid=8, w=1, h=1):
    """Pick at most one match per grid cell so the drawn lines cover the image evenly."""
    cell = {}
    for i, (x, y) in enumerate(points):
        key = (int(x / w * grid), int(y / h * grid))
        cell.setdefault(key, i)
    idx = list(cell.values())
    return idx[:n_max]


def draw_match_lines(img_l, img_r, pts_l, pts_r, colors, thickness=2):
    h = max(img_l.shape[0], img_r.shape[0])
    canvas = np.zeros((h, img_l.shape[1] + img_r.shape[1], 3), np.uint8)
    canvas[:img_l.shape[0], :img_l.shape[1]] = img_l
    canvas[:img_r.shape[0], img_l.shape[1]:] = img_r
    off = img_l.shape[1]
    for (xa, ya), (xb, yb), c in zip(pts_l, pts_r, colors):
        pa, pb = (int(xa), int(ya)), (int(xb) + off, int(yb))
        cv2.line(canvas, pa, pb, c, thickness, cv2.LINE_AA)
        cv2.circle(canvas, pa, 5, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, pb, 5, c, -1, cv2.LINE_AA)
    return canvas


def match_figure(images, res, method, out):
    """For each pair: (top) every ratio-test match, inliers green / outliers red;
    (bottom) cleaned result – RANSAC inliers only, spread evenly, one colour per line."""
    hp = res["hpairs"]
    aspect = images[0].shape[0] / (2.0 * images[0].shape[1])          # height / width of one canvas
    fig, axes = plt.subplots(len(hp), 2, figsize=(18, 9 * aspect * len(hp) + 1))
    axes = np.atleast_2d(axes)
    cmap = plt.get_cmap("hsv")
    for i, p in enumerate(hp):
        a, b = images[i], images[i + 1]
        # dst = points in image i (left), src = points in image i+1 (right)
        raw = draw_match_lines(a, b, p["dst"], p["src"],
                               [(0, 200, 0) if ok else (0, 0, 255) for ok in p["inl"]], 1)
        ii = np.where(p["inl"])[0]
        sel = ii[spread_subset(p["dst"][ii], 50, 14, a.shape[1], a.shape[0])]
        cols = [tuple(int(255 * c) for c in cmap(k / max(len(sel), 1))[:3][::-1]) for k in range(len(sel))]
        clean = draw_match_lines(a, b, p["dst"][sel], p["src"][sel], cols, 2)
        cv2.imwrite(os.path.join(out, f"C1_{method}_pair{i + 1}-{i + 2}_raw.jpg"), raw)
        cv2.imwrite(os.path.join(out, f"C1_{method}_pair{i + 1}-{i + 2}_clean.jpg"), clean)
        axes[i, 0].imshow(bgr2rgb(raw))
        axes[i, 0].set_title(f"{method} {i + 1}↔{i + 2}: all {len(p['good'])} ratio-test matches "
                             f"(green = inlier, red = outlier)")
        axes[i, 1].imshow(bgr2rgb(clean))
        axes[i, 1].set_title(f"after RANSAC: {int(p['inl'].sum())} inliers ({inlier_ratio(p['inl']):.0f}%), "
                             f"{len(sel)} evenly spread shown")
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(f"Part C – {method} matches before and after cleaning", fontweight="bold")
    save_fig(fig, os.path.join(out, f"C1_matches_{method}.png"), dpi=110)


def homography_estimator_study(images, out):
    """Same SIFT matches of pair 1-2: least squares vs RANSAC (3 thresholds) vs LMedS,
    with and without 20 % extra synthetic outliers."""
    (kq, dq), (kt, dt) = detect(images[1], "SIFT"), detect(images[0], "SIFT")
    good = match(dq, dt, "SIFT")
    src, dst = pair_points(kq, kt, good)
    H_ref, m_ref = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
    ref_inl = m_ref.ravel().astype(bool)
    rng = np.random.default_rng(0)
    h, w = images[0].shape[:2]
    n_out = len(src) // 5
    src_c = np.vstack([src, rng.uniform([0, 0], [w, h], (n_out, 2)).astype(np.float32)])
    dst_c = np.vstack([dst, rng.uniform([0, 0], [w, h], (n_out, 2)).astype(np.float32)])
    rows = []
    for name, meth, thr in [("Least squares (all)", 0, 0), ("RANSAC thr=1px", cv2.RANSAC, 1.0),
                            ("RANSAC thr=3px", cv2.RANSAC, 3.0), ("RANSAC thr=10px", cv2.RANSAC, 10.0),
                            ("LMedS", cv2.LMEDS, 0)]:
        for label, s, d in (("ratio-test matches", src, dst), ("+20% random outliers", src_c, dst_c)):
            t = {}
            with timer(t, "ms"):
                H, mask = cv2.findHomography(s, d, meth, thr)
            inl = mask.ravel().astype(bool) if (mask is not None and meth) else np.ones(len(s), bool)
            e = reprojection_errors(H, src[ref_inl], dst[ref_inl])
            rows.append({"Estimator": name, "Data": label, "Inliers": int(inl.sum()), "Total": len(s),
                         "RMSE_on_true_inliers_px": float(np.sqrt(np.mean(e ** 2))), "Time_ms": t["ms"]})
    write_table(rows, os.path.join(out, "homography_estimator_comparison"))


def keypoint_figure(images, out, max_draw=1500):
    """Detected keypoints of every detector on the reference (middle) image."""
    ref = images[(len(images) - 1) // 2]
    fig, axes = plt.subplots(1, len(DETECTORS), figsize=(6 * len(DETECTORS), 6 * ref.shape[0] / ref.shape[1] + 0.6))
    for ax, det in zip(np.atleast_1d(axes), DETECTORS):
        kps, _ = detect(ref, det)
        strongest = sorted(kps, key=lambda k: -k.response)[:max_draw]
        vis = cv2.drawKeypoints(ref, strongest, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(os.path.join(out, f"C0_keypoints_{det}.jpg"), vis)
        ax.imshow(bgr2rgb(vis))
        ax.set_title(f"{det}: {len(kps)} keypoints (strongest {len(strongest)} drawn, circle = scale)")
        ax.axis("off")
    fig.suptitle("Detected keypoints on the reference image", fontweight="bold")
    save_fig(fig, os.path.join(out, "C0_keypoints.png"), dpi=110)


def ransac_threshold_study(images, out):
    """Every detector × every consecutive pair × RANSAC thresholds 1/3/5/10 px."""
    rows = []
    for det in DETECTORS:
        feats = [detect(im, det) for im in images]
        for thr in RANSAC_THRESHOLDS:
            inl_tot, match_tot, rmses = 0, 0, []
            for i in range(len(images) - 1):
                (kq, dq), (kt, dt) = feats[i + 1], feats[i]
                good = match(dq, dt, det)
                src, dst = pair_points(kq, kt, good)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thr, maxIters=5000, confidence=0.999)
                inl = mask.ravel().astype(bool)
                inl_tot += int(inl.sum())
                match_tot += len(good)
                rmses.append(reprojection_rmse(H, src[inl], dst[inl]))
            rows.append({"Detector": det, "RANSAC_thr_px": thr, "Initial_matches": match_tot,
                         "Inliers": inl_tot, "Inlier_%": 100.0 * inl_tot / max(match_tot, 1),
                         "Mean_RMSE_px": float(np.mean(rmses))})
    write_table(rows, os.path.join(out, "ransac_threshold_study"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for det, mk in zip(DETECTORS, ("o-", "s-", "^-")):
        sub = [r for r in rows if r["Detector"] == det]
        axes[0].plot(RANSAC_THRESHOLDS, [r["Inlier_%"] for r in sub], mk, label=det)
        axes[1].plot(RANSAC_THRESHOLDS, [r["Mean_RMSE_px"] for r in sub], mk, label=det)
    for ax, lab in zip(axes, ("inlier ratio (%)", "reprojection RMSE of inliers (px)")):
        ax.set_xlabel("RANSAC threshold (px)"); ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("RANSAC threshold: looser threshold = more inliers but less precise inliers", fontweight="bold")
    save_fig(fig, os.path.join(out, "C7_ransac_threshold_study.png"))
    return rows


# ============================================================================ automatic ordering
def auto_order(images, method="SIFT", min_inliers=25):
    """Find the left -> right order automatically, so file names do not matter.
    For every pair (i, j) with enough RANSAC inliers, dx_ij = median(x_i − x_j) of the matched
    points ≈ how far image j sits to the right of image i. Positions p are then solved by least
    squares (p_j − p_i = dx_ij, p_0 = 0) and the images are sorted by p."""
    n = len(images)
    feats = [detect(im, method) for im in images]
    eqs = []
    for i in range(n):
        for j in range(i + 1, n):
            good = match(feats[j][1], feats[i][1], method)
            if len(good) < 8:
                continue
            src, dst = pair_points(feats[j][0], feats[i][0], good)        # j -> i
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THR)
            if mask is None or mask.sum() < min_inliers:
                continue
            inl = mask.ravel().astype(bool)
            eqs.append((i, j, float(np.median(dst[inl, 0] - src[inl, 0])), int(inl.sum())))
    A, b, wts = [], [], []
    for i, j, dx, w in eqs:
        row = np.zeros(n)
        row[j], row[i] = 1, -1
        A.append(row); b.append(dx); wts.append(np.sqrt(w))
    A.append(np.eye(n)[0]); b.append(0.0); wts.append(100.0)          # anchor p_0 = 0
    A, b, wts = np.array(A), np.array(b), np.array(wts)
    p = np.linalg.lstsq(A * wts[:, None], b * wts, rcond=None)[0]
    order = list(np.argsort(p))
    return order, p, eqs


# ============================================================================ main
def find_own_photos():
    files = [f for f in glob.glob(os.path.join(DATA_DIR, "own_photos", "part_c", "*"))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))]
    num = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]  # noqa: E731
    return sorted(files, key=num)


def main(paths=None):
    out = out_dir("part_c")
    paths = paths or find_own_photos()
    if len(paths) < 2:
        raise SystemExit("Put the overlapping photos in data/own_photos/part_c/")
    images = [resize_max(imread(p), MAX_SIDE) for p in paths]
    order, pos, _ = auto_order(images)
    if order != list(range(len(images))):
        print(f"  auto-order: {[os.path.basename(paths[k]) for k in order]} (left -> right)")
    images, paths = [images[k] for k in order], [paths[k] for k in order]
    with open(os.path.join(out, "image_order.txt"), "w") as f:
        f.write("left -> right: " + ", ".join(os.path.basename(p) for p in paths) + "\n")
    print(f"[Part C] {len(images)} images: {[os.path.basename(p) for p in paths]}, size {images[0].shape[:2]}")

    f_exif = exif_focal_px(paths[0], images[0].shape[1])
    print(f"  focal length from EXIF: {f_exif:.1f} px" if f_exif else "  no EXIF focal – estimated from H")
    keypoint_figure(images, out)
    combos = [(d, p) for d in DETECTORS for p in ("cylindrical", "planar")]
    results, rows, summary = {}, [], []
    for det, proj in combos:
        try:
            build_panorama(images, det, proj, f_exif)            # warm-up (fair timing)
            r = build_panorama(images, det, proj, f_exif)
        except RuntimeError as e:
            print(f"  {det}/{proj}: FAILED – {e}")
            summary.append({"Detector": det, "Projection": proj, "Status": f"failed: {e}"})
            continue
        results[(det, proj)] = r
        rows += r["stats"]
        raw, final, kept = r["raw"], r["final"], r["kept"]
        summary.append({"Detector": det, "Projection": proj,
                        "Focal_px": round(r["info"]["focal"], 1) if r["info"]["focal"] else "-",
                        "Focal_from_H_px": round(r["info"]["focal_from_H"], 1) if r["info"].get("focal_from_H") else "-",
                        "Tilt_removed_deg": round(r["info"]["tilt_deg"], 2),
                        "Mean_H_inliers": float(np.mean([s["H_inliers"] for s in r["stats"]])),
                        "Mean_H_RMSE_px": float(np.mean([s["H_reproj_RMSE_px"] for s in r["stats"]])),
                        "Overlap_RMSE": overlap_rmse(r["warped"], r["masks"]),
                        "Black_border_%_before_crop": black_border_ratio(raw),
                        "Straightness_%": straightness(raw),
                        "Area_kept_after_crop_%": kept,
                        "Final_size": f"{final.shape[1]}x{final.shape[0]}",
                        "Total_time_ms": r["time_ms"]})
        cv2.imwrite(os.path.join(out, f"C3_panorama_{det}_{proj}_uncropped.jpg"), raw)
        cv2.imwrite(os.path.join(out, f"C3_panorama_{det}_{proj}_final.jpg"), final)
        print(f"  {det:5s}/{proj:11s}: inliers {[s['H_inliers'] for s in r['stats']]}, "
              f"straightness {summary[-1]['Straightness_%']:.1f}%, final {summary[-1]['Final_size']}, "
              f"{r['time_ms']:.0f} ms")
    write_table(rows, os.path.join(out, "pairwise_matching_metrics"))
    write_table(summary, os.path.join(out, "panorama_comparison"))

    # ---- required detector comparison table (homographies on the original images) ----
    det_rows = []
    for det in DETECTORS:
        if (det, "cylindrical") not in results:
            continue
        r = results[(det, "cylindrical")]
        hp = r["hpairs"]
        kps = [hp[0]["n_kp"][0]] + [p["n_kp"][1] for p in hp]
        n_match = sum(len(p["good"]) for p in hp)
        n_inl = sum(int(p["inl"].sum()) for p in hp)
        det_rows.append({"Detector": det, "Keypoints_per_image": float(np.mean(kps)),
                         "Initial_matches": n_match, "Inliers": n_inl,
                         "Inlier_%": 100.0 * n_inl / max(n_match, 1),
                         "Mean_RMSE_px": float(np.mean([s["H_reproj_RMSE_px"] for s in r["stats"]])),
                         "Total_stitching_time_ms": r["time_ms"]})
    write_table(det_rows, os.path.join(out, "detector_comparison"))

    for det in DETECTORS:
        if (det, "planar") in results:
            match_figure(images, results[(det, "planar")], det, out)

    # ---- final panorama of every detector ----
    avail = [d for d in DETECTORS if (d, "cylindrical") in results]
    pano0 = results[(avail[0], "cylindrical")]["final"]
    fig, axes = plt.subplots(len(avail), 1, figsize=(14, len(avail) * (14 * pano0.shape[0] / pano0.shape[1] + 0.6)))
    for ax, det in zip(np.atleast_1d(axes), avail):
        ax.imshow(bgr2rgb(results[(det, "cylindrical")]["final"]))
        d = next(x for x in det_rows if x["Detector"] == det)
        ax.set_title(f"{det}: {d['Inliers']} inliers, RMSE {d['Mean_RMSE_px']:.2f} px, "
                     f"{d['Total_stitching_time_ms']:.0f} ms")
        ax.axis("off")
    fig.suptitle("Final panorama for each feature detector (cylindrical + multi-band)", fontweight="bold")
    save_fig(fig, os.path.join(out, "C6_panorama_per_detector.png"), dpi=110)

    # ---- planar vs cylindrical (the "flat panorama" argument), SIFT ----
    pk = [k for k in (("SIFT", "planar"), ("SIFT", "cylindrical")) if k in results]
    fig, axes = plt.subplots(len(pk), 2, figsize=(18, 8 * len(pk)))
    axes = np.atleast_2d(axes)
    for row, k in zip(axes, pk):
        s = next(x for x in summary if x["Detector"] == k[0] and x["Projection"] == k[1])
        row[0].imshow(bgr2rgb(results[k]["raw"]))
        row[0].set_title(f"{k[0]} + {k[1]} – before crop (straightness {s['Straightness_%']:.1f}%, "
                         f"black {s['Black_border_%_before_crop']:.0f}%)")
        row[1].imshow(bgr2rgb(results[k]["final"]))
        row[1].set_title(f"final, cropped to largest rectangle ({s['Area_kept_after_crop_%']:.0f}% kept)")
        for ax in row:
            ax.axis("off")
    fig.suptitle("Planar (homography) vs cylindrical projection", fontweight="bold")
    save_fig(fig, os.path.join(out, "C2_projection_comparison.png"), dpi=100)

    # ---- blending comparison on the best pipeline ----
    best = ("SIFT", "cylindrical") if ("SIFT", "cylindrical") in results else avail[0]
    r = results[best]
    panos = {name: fn(r["warped"], r["masks"]) for name, fn in BLENDERS.items()}
    valid = np.max(np.stack(r["masks"]), axis=0) > 0
    y0, y1, x0, x1 = largest_valid_rectangle(valid)
    for name, p in panos.items():
        cv2.imwrite(os.path.join(out, f"C4_blend_{name.lower().replace('-', '')}.jpg"), p[y0:y1, x0:x1])
    diff = cv2.absdiff(panos["Overwrite"], panos["Multi-band"]).astype(np.float32).mean(axis=2)
    both = np.sum([m > 0 for m in r["masks"]], axis=0) >= 2
    diff = cv2.boxFilter(np.where(both, diff, 0), -1, (61, 61))
    cy, cx = np.unravel_index(np.argmax(diff), diff.shape)
    half = 160
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), gridspec_kw=dict(height_ratios=[1, 1.4]))
    for k, (name, p) in enumerate(panos.items()):
        axes[0, k].imshow(bgr2rgb(p[y0:y1, x0:x1]))
        axes[0, k].set_title(name)
        axes[1, k].imshow(bgr2rgb(p[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half]))
        axes[1, k].set_title(f"{name} – zoom on a seam")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Part C – Blending comparison ({best[0]} + {best[1]})", fontweight="bold")
    save_fig(fig, os.path.join(out, "C4_blending_comparison.png"), dpi=110)
    cv2.imwrite(os.path.join(out, "C5_FINAL_PANORAMA.jpg"), results[best]["final"])

    with open(os.path.join(out, "homographies_SIFT.txt"), "w") as f:
        if ("SIFT", "planar") in results:
            for i, p in enumerate(results[("SIFT", "planar")]["hpairs"]):
                f.write(f"H_{i + 2}->{i + 1} =\n{np.array2string(p['T'], precision=5, suppress_small=True)}\n\n")
    homography_estimator_study(images, out)
    ransac_threshold_study(images, out)
    print(f"  final panorama -> results/part_c/C5_FINAL_PANORAMA.jpg ({best[0]} + {best[1]})")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=None, help="ordered left -> right")
    main(ap.parse_args().images)
