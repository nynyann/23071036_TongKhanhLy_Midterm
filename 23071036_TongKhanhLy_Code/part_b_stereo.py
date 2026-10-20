"""
Part B – 3D Reconstruction from stereo (Middlebury data given by the lecturer)
=============================================================================
Data: https://vision.middlebury.edu/flow/data/  – unzip other-color-twoframes.zip,
other-gt-flow.zip and eval-color-twoframes.zip into data/middlebury/ (or run
download_middlebury.py). Venus and Teddy are the two *stereo* scenes of that set:
  * frame10.png = LEFT view, frame11.png = RIGHT view (rows already aligned, |v| = 0)
  * the horizontal ground-truth flow u of frame10 equals the negative disparity: d = −u
  * the two frames are crops of the original rectified pair, shifted horizontally by
    DOFFS pixels (Venus 10 px, Teddy 26 px – measured once by template matching frame10 /
    frame11 against im2 / im6 of the original Middlebury stereo release). Disparities can
    therefore be negative and depth is  Z = f·B / (d + doffs)  (Middlebury's own convention).
  * Teddy's ground-truth flow is hidden in the flow benchmark; its public ground-truth
    disparity from the Middlebury stereo release (data/middlebury/stereo-gt/Teddy/disp2.png,
    scale 4) is cropped to the same window and used for evaluation.

Steps
1. Disparity: StereoBM vs StereoSGBM (two block sizes each); the search range is estimated
   from sparse SIFT correspondences (the ground truth is used ONLY for evaluation).
2. 3D point cloud -> interactive HTML (Plotly, rotate with the mouse), .ply for MeshLab, views.
3. Fundamental matrix from SIFT matches: 8-point vs RANSAC vs LMedS + epipolar lines.

Usage:  python part_b_stereo.py [--scenes Venus Teddy]
"""
import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from metrics import epipolar_distance, epipolar_line_angle, sampson_error, stereo_report
from utils import DATA_DIR, bgr2rgb, out_dir, save_fig, timer, write_table

DOFFS = {"venus": 10.0, "teddy": 26.0}      # crop offset between frame10 and frame11 (px)
STEREO_GT_SCALE = {"venus": 8.0, "teddy": 4.0}


# ============================================================================ data loading
def read_flo(path):
    """Middlebury .flo: float32 magic 202021.25, int32 width, int32 height, then (u,v) pairs."""
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, 1)[0]
        if abs(magic - 202021.25) > 1e-3:
            raise ValueError(f"{path} is not a .flo file")
        w, h = np.fromfile(f, np.int32, 2)
        data = np.fromfile(f, np.float32, 2 * w * h).reshape(h, w, 2)
    return data


def _find(pattern):
    hits = sorted(glob.glob(os.path.join(DATA_DIR, "**", pattern), recursive=True))
    return hits[0] if hits else None


def load_scene(name):
    """Returns dict(left, right, gt, valid, doffs, source). gt is the disparity of the LEFT frame
    in frame coordinates (NaN where unknown), or None when no ground truth exists."""
    f10 = _find(os.path.join(name, "frame10.png"))
    if f10 is None:
        raise FileNotFoundError(f"{name}/frame10.png not found under {DATA_DIR} – run download_middlebury.py")
    left = cv2.imread(f10)
    right = cv2.imread(os.path.join(os.path.dirname(f10), "frame11.png"))
    h, w = left.shape[:2]
    doffs = DOFFS.get(name.lower(), 0.0)
    gt, source = None, "frame10/frame11"
    flo = _find(os.path.join(name, "flow10.flo"))
    if flo:                                                   # Venus: GT flow is public
        u = read_flo(flo)[..., 0]
        gt = np.where(np.abs(u) < 1e9, -u, np.nan).astype(np.float32)   # d = x_L − x_R = −u
        source += " + flow10.flo"
    else:                                                     # Teddy: GT from the stereo release
        sgt = _find(os.path.join("stereo-gt", name, "disp2.png"))
        if sgt:
            g = cv2.imread(sgt, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            o = int(doffs)
            g = g[:h, o:o + w]
            gt = np.where(g > 0, g / STEREO_GT_SCALE[name.lower()] - doffs, np.nan).astype(np.float32)
            source += " + stereo-release disp2.png (cropped)"
    valid = np.isfinite(gt) if gt is not None else None
    return dict(left=left, right=right, gt=gt, valid=valid, doffs=doffs, source=source)


# ============================================================================ matchers
def sift_matches(left, right, ratio=0.75):
    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    return (np.float32([k1[m.queryIdx].pt for m in good]),
            np.float32([k2[m.trainIdx].pt for m in good]))


def disparity_search_range(left, right, margin=4):
    """Estimate [minDisparity, minDisparity + numDisparities) from sparse SIFT matches that are
    consistent with the epipolar geometry (no ground truth needed)."""
    p1, p2 = sift_matches(left, right)
    _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.999)
    dx = (p1[:, 0] - p2[:, 0])[mask.ravel().astype(bool)]
    lo = int(np.floor(np.percentile(dx, 1))) - margin
    hi = int(np.ceil(np.percentile(dx, 99))) + margin
    num = int(np.ceil((hi - lo) / 16.0) * 16)
    return lo, num


def make_bm(block, num_disp, min_disp=0):
    """Block Matching: slide a block along the same row of the right image, keep the shift with
    the smallest Sum of Absolute Differences (local, winner-takes-all)."""
    bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
    bm.setMinDisparity(min_disp)
    bm.setPreFilterCap(31)
    bm.setTextureThreshold(10)
    bm.setUniquenessRatio(10)
    bm.setSpeckleWindowSize(100)
    bm.setSpeckleRange(32)
    return bm


def make_sgbm(block, num_disp, min_disp=0):
    """Semi-Global Matching: minimise E(D) = Σ C(p,D_p) + Σ P1·[|ΔD|=1] + Σ P2·[|ΔD|>1],
    aggregating the cost along several 1-D paths (OpenCV 3-way mode) -> dense, piecewise-smooth disparity."""
    return cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=block,
        P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2, disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32, preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def compute_disparity(matcher, left, right, is_bm):
    """Float disparity in pixels; pixels without a valid match are NaN."""
    if is_bm:
        left, right = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    d = matcher.compute(left, right).astype(np.float32) / 16.0   # OpenCV returns fixed point ×16
    d[d < matcher.getMinDisparity()] = np.nan                     # invalid = minDisparity − 1
    return d


def right_disparity(matcher, left, right, is_bm):
    """Disparity referenced to the right image (mirror trick) – needed for the LR check."""
    return cv2.flip(compute_disparity(matcher, cv2.flip(right, 1), cv2.flip(left, 1), is_bm), 1)


def run_matchers(name, sc):
    """BM (blocks 9, 15) and SGBM (blocks 3, 7) with the SIFT-estimated range; returns rows, maps."""
    min_disp, num_disp = disparity_search_range(sc["left"], sc["right"])
    print(f"  {name}: disparity search range [{min_disp}, {min_disp + num_disp}) estimated from SIFT matches")
    configs = [("BM block=9", make_bm(9, num_disp, min_disp), True),
               ("BM block=15", make_bm(15, num_disp, min_disp), True),
               ("SGBM block=3", make_sgbm(3, num_disp, min_disp), False),
               ("SGBM block=7", make_sgbm(7, num_disp, min_disp), False)]
    rows, disps = [], {}
    for cname, m, is_bm in configs:
        compute_disparity(m, sc["left"], sc["right"], is_bm)          # warm-up for fair timing
        t = {}
        with timer(t, "ms"):
            d = compute_disparity(m, sc["left"], sc["right"], is_bm)
        dr = right_disparity(m, sc["left"], sc["right"], is_bm)
        disps[cname] = d
        rows.append({"Scene": name, "Method": cname, "Range": f"[{min_disp},{min_disp + num_disp})",
                     **stereo_report(d, sc["gt"], sc["valid"], dr), "Time_ms": t["ms"]})
    return rows, disps


def disparity_figure(name, sc, rows, disps, path):
    gt = sc["gt"]
    vals = np.concatenate([d[np.isfinite(d)] for d in disps.values()] + ([gt[sc["valid"]]] if gt is not None else []))
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99.5)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes[0, 0].imshow(bgr2rgb(sc["left"])); axes[0, 0].set_title("Left (frame10)")
    axes[0, 1].imshow(bgr2rgb(sc["right"])); axes[0, 1].set_title("Right (frame11)")
    if gt is not None:
        axes[0, 2].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax); axes[0, 2].set_title("Ground-truth disparity")
    for ax, (cname, d), r in zip(axes[1:].ravel()[:4], disps.items(), rows):
        ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax)
        txt = f"{cname}\nvalid {r['Valid_%']:.0f}%"
        if gt is not None:
            txt += f" | bad>1px {r['Bad1.0_%']:.1f}% | EPE {r['EPE_px']:.2f}"
        ax.set_title(txt, fontsize=10)
    if gt is not None:   # error maps of the best BM and the best SGBM configuration
        for ax, fam in ((axes[2, 1], "BM"), (axes[2, 2], "SGBM")):
            best = min((r for r in rows if r["Method"].startswith(fam + " ")), key=lambda r: r["Bad1.0_%"])
            im = ax.imshow(np.abs(disps[best["Method"]] - gt), cmap="inferno", vmin=0, vmax=4)
            ax.set_title(f"|d − d_gt| of {best['Method']} (px)")
            fig.colorbar(im, ax=ax, fraction=0.04)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{name}: BM vs SGBM disparity (black = no estimate)", fontweight="bold")
    save_fig(fig, path)


def parameter_study(name, sc, out):
    """Sensitivity of BM / SGBM to blockSize and numDisparities (other parameters fixed)."""
    lo, num = disparity_search_range(sc["left"], sc["right"])
    rows = []
    settings = ([("BM", "blockSize", b, make_bm(b, num, lo), True) for b in (5, 9, 15, 21)] +
                [("SGBM", "blockSize", b, make_sgbm(b, num, lo), False) for b in (3, 5, 7, 11)] +
                [("BM", "numDisparities", n, make_bm(9, n, lo), True) for n in (16, 32, 48, 64)] +
                [("SGBM", "numDisparities", n, make_sgbm(5, n, lo), False) for n in (16, 32, 48, 64)])
    for algo, param, val, m, is_bm in settings:
        compute_disparity(m, sc["left"], sc["right"], is_bm)
        t = {}
        with timer(t, "ms"):
            d = compute_disparity(m, sc["left"], sc["right"], is_bm)
        rep = stereo_report(d, sc["gt"], sc["valid"])
        rows.append({"Scene": name, "Algorithm": algo, "Parameter": param, "Value": val,
                     "Range": f"[{lo},{lo + (num if param == 'blockSize' else val)})", **rep, "Time_ms": t["ms"]})
    write_table(rows, os.path.join(out, "parameter_study"))

    metric = "Bad1.0_%" if sc["gt"] is not None else "Valid_%"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for ax, param in zip(axes[:2], ("blockSize", "numDisparities")):
        for algo, mk in (("BM", "o-"), ("SGBM", "s-")):
            sub = [r for r in rows if r["Algorithm"] == algo and r["Parameter"] == param]
            ax.plot([r["Value"] for r in sub], [r[metric] for r in sub], mk, label=f"{algo} – {metric}")
            ax.plot([r["Value"] for r in sub], [r["Valid_%"] for r in sub], mk.replace("-", "--"), alpha=0.6,
                    label=f"{algo} – Valid_%")
        ax.set_xlabel(param); ax.set_ylabel("%"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"{name}: effect of {param}")
    for algo, mk in (("BM", "o-"), ("SGBM", "s-")):
        sub = [r for r in rows if r["Algorithm"] == algo and r["Parameter"] == "numDisparities"]
        axes[2].plot([r["Value"] for r in sub], [r["Time_ms"] for r in sub], mk, label=algo)
    axes[2].set_xlabel("numDisparities"); axes[2].set_ylabel("runtime (ms)"); axes[2].grid(alpha=0.3)
    axes[2].legend(); axes[2].set_title("runtime grows linearly with the search range")
    save_fig(fig, os.path.join(out, "B3_parameter_study.png"))
    return rows


# ============================================================================ 3D
def disparity_to_points(disp, img, focal, baseline, doffs=0.0):
    """Triangulation with Middlebury's convention Z = f·B / (d + doffs)."""
    h, w = disp.shape
    cx, cy = w / 2.0, h / 2.0
    # [X Y Z W]ᵀ = Q [u v d 1]ᵀ  ->  W = (d + doffs)/B,  Z = f/W
    Q = np.float32([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, focal], [0, 0, 1.0 / baseline, doffs / baseline]])
    d = np.nan_to_num(disp, nan=-1e6).astype(np.float32)
    pts = cv2.reprojectImageTo3D(d, Q)
    mask = np.isfinite(disp) & (disp + doffs > 1.0)
    xyz, rgb = pts[mask], bgr2rgb(img)[mask]
    keep = np.isfinite(xyz).all(1) & (xyz[:, 2] < np.percentile(xyz[:, 2], 99))
    return xyz[keep], rgb[keep], Q


def write_ply(path, xyz, rgb):
    """Binary PLY – open with MeshLab (File > Import Mesh), CloudCompare or Open3D."""
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz.T
    v["red"], v["green"], v["blue"] = rgb.T
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(v.tobytes())


def interactive_point_cloud(xyz, rgb, path, title, n_max=50000):
    """Plotly HTML: open in any browser and rotate/zoom with the mouse (works in Colab too)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed -> skip interactive HTML (pip install plotly)")
        return None
    idx = np.random.default_rng(0).choice(len(xyz), min(n_max, len(xyz)), replace=False)
    p, c = xyz[idx], rgb[idx]
    colors = [f"rgb({r},{g},{b})" for r, g, b in c]
    fig = go.Figure(go.Scatter3d(x=p[:, 0], y=p[:, 2], z=-p[:, 1], mode="markers",
                                 marker=dict(size=1.6, color=colors)))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis_title="X", yaxis_title="Z (depth)", zaxis_title="−Y",
                                 aspectmode="data",
                                 camera=dict(eye=dict(x=0.0, y=-1.8, z=0.6))))
    fig.write_html(path, include_plotlyjs="cdn")   # small file; needs internet to load plotly.js
    return fig


def static_views(xyz, rgb, path, title, n_max=60000):
    """Three fixed viewpoints of the same cloud, for the PDF report."""
    idx = np.random.default_rng(0).choice(len(xyz), min(n_max, len(xyz)), replace=False)
    p, c = xyz[idx], rgb[idx] / 255.0
    fig = plt.figure(figsize=(16, 5.5))
    for i, (elev, azim, name) in enumerate([(0, -90, "Front"), (25, -60, "Oblique"), (89, -90, "Top (X–Z)")], 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(p[:, 0], p[:, 2], -p[:, 1], c=c, s=0.8, linewidths=0, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X"); ax.set_ylabel("Z (depth)"); ax.set_zlabel("−Y")
        ax.set_title(name)
    fig.suptitle(title, fontweight="bold")
    save_fig(fig, path)


# ============================================================================ fundamental matrix
def draw_epilines(left, right, F, p1, p2, n=16, seed=3):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(p1), min(n, len(p1)), replace=False)
    a, b = left.copy(), right.copy()
    w = a.shape[1]
    lr = cv2.computeCorrespondEpilines(p1[sel].reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    ll = cv2.computeCorrespondEpilines(p2[sel].reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    cmap = plt.get_cmap("hsv")
    for i, j in enumerate(sel):
        col = tuple(int(255 * v) for v in cmap(i / len(sel))[:3][::-1])
        for img, (A, B, C), pt in ((b, lr[i], p2[j]), (a, ll[i], p1[j])):
            cv2.line(img, (0, int(round(-C / B))), (w, int(round(-(C + A * w) / B))), col, 1, cv2.LINE_AA)
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, col, -1, cv2.LINE_AA)
    return a, b


def fundamental_study(left, right, out):
    p1, p2 = sift_matches(left, right)
    rows, Fs = [], {}
    for name, flag, thr in [("8-point (all matches)", cv2.FM_8POINT, None),
                            ("RANSAC (1 px)", cv2.FM_RANSAC, 1.0), ("LMedS", cv2.FM_LMEDS, None)]:
        t = {}
        with timer(t, "ms"):
            F, mask = (cv2.findFundamentalMat(p1, p2, flag) if thr is None
                       else cv2.findFundamentalMat(p1, p2, flag, thr, 0.999))
        F = F[:3] / F[2, 2]
        inl = mask.ravel().astype(bool) if flag != cv2.FM_8POINT else np.ones(len(p1), bool)
        err = epipolar_distance(F, p1, p2)
        samp = sampson_error(F, p1, p2)
        s = np.linalg.svd(F, compute_uv=False)
        Fs[name] = (F, inl)
        rows.append({"Method": name, "Matches": len(p1), "Inliers": int(inl.sum()),
                     "Inlier_%": 100.0 * inl.mean(), "Mean_epi_err_inliers_px": float(err[inl].mean()),
                     "Mean_Sampson_inliers_px": float(samp[inl].mean()),
                     "Median_epi_err_all_px": float(np.median(err)),
                     "Mean_abs_line_angle_deg": float(np.abs(epipolar_line_angle(F, p1[inl])).mean()),
                     "Rank2_s3/s1": float(s[2] / s[0]), "Time_ms": t["ms"]})
    write_table(rows, os.path.join(out, "fundamental_matrix_metrics"), float_fmt="{:.4g}")
    F_r, inl_r = Fs["RANSAC (1 px)"]
    np.savetxt(os.path.join(out, "F_ransac.txt"), F_r, fmt="%.6e")

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    for row_axes, (name, (F, inl)), r in zip(axes, Fs.items(), rows):
        a, b = draw_epilines(left, right, F, p1[inl_r], p2[inl_r])   # same points for every F
        row_axes[0].imshow(bgr2rgb(a)); row_axes[1].imshow(bgr2rgb(b))
        row_axes[0].set_title(f"{name} – left")
        row_axes[1].set_title(f"right | inliers {r['Inliers']}/{r['Matches']} | "
                              f"err {r['Mean_epi_err_inliers_px']:.2f}px | tilt {r['Mean_abs_line_angle_deg']:.1f}°")
        for ax in row_axes:
            ax.axis("off")
    fig.suptitle("Epipolar lines from the estimated fundamental matrix", fontweight="bold")
    save_fig(fig, os.path.join(out, "B4_epipolar_lines_comparison.png"))
    a, b = draw_epilines(left, right, F_r, p1[inl_r], p2[inl_r], n=20)
    cv2.imwrite(os.path.join(out, "B5_epipolar_lines_ransac.png"), np.hstack([a, b]))
    return rows


# ============================================================================ main
def run_scene(name, focal=None, baseline=1.0):
    out = out_dir(os.path.join("part_b", name.lower()))
    sc = load_scene(name)
    h, w = sc["left"].shape[:2]
    focal = focal or 0.8 * w
    print(f"[Part B] {name}: {w}x{h}, {sc['source']}, doffs = {sc['doffs']}")
    rows, disps = run_matchers(name, sc)
    write_table(rows, os.path.join(out, "disparity_metrics"))
    disparity_figure(name, sc, rows, disps, os.path.join(out, "B1_disparity_comparison.png"))

    clouds = {"sgbm": disps["SGBM block=3"], "bm": disps["BM block=9"]}
    if sc["gt"] is not None:
        clouds["groundtruth"] = sc["gt"]
    for tag, d in clouds.items():
        xyz, rgb, Q = disparity_to_points(d, sc["left"], focal, baseline, sc["doffs"])
        write_ply(os.path.join(out, f"point_cloud_{tag}.ply"), xyz, rgb)
        interactive_point_cloud(xyz, rgb, os.path.join(out, f"B2_point_cloud_{tag}_interactive.html"),
                                f"{name} – point cloud from {tag.upper()} disparity ({len(xyz):,} pts)")
        static_views(xyz, rgb, os.path.join(out, f"B2_point_cloud_{tag}.png"),
                     f"{name} – 3D point cloud from {tag.upper()} disparity")
    np.savetxt(os.path.join(out, "Q_matrix.txt"), Q, fmt="%.5f")
    parameter_study(name, sc, out)
    frows = fundamental_study(sc["left"], sc["right"], out)
    return rows, frows


def main(scenes=("Venus", "Teddy"), focal=None, baseline=1.0):
    all_rows, all_f = [], []
    for s in scenes:
        try:
            r, f = run_scene(s, focal, baseline)
        except FileNotFoundError as e:
            print(f"  skip {s}: {e}")
            continue
        all_rows += r
        all_f += [{"Scene": s, **x} for x in f]
    out = out_dir("part_b")
    write_table(all_rows, os.path.join(out, "disparity_metrics_all_scenes"))
    write_table(all_f, os.path.join(out, "fundamental_matrix_all_scenes"), float_fmt="{:.4g}")
    return all_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["Venus", "Teddy"])
    ap.add_argument("--focal", type=float, default=None, help="focal length in px (default 0.8*width)")
    ap.add_argument("--baseline", type=float, default=1.0)
    a = ap.parse_args()
    main(a.scenes, a.focal, a.baseline)
