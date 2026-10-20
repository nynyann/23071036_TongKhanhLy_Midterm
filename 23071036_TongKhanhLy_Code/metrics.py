"""
Evaluation metrics used in the report (one function per metric, all NumPy/OpenCV).

Part A – filtering      : mse, psnr, ssim, edge_preservation_index, edge_f1, sharpness
Part B – stereo         : disparity_density (valid ratio), lr_consistency, bad_pixel_rate,
                          disparity_epe, disparity_rmse, epipolar_distance, sampson_error,
                          epipolar_line_angle
Part C – stitching      : inlier_ratio, reprojection_errors, reprojection_rmse, overlap_rmse,
                          black_border_ratio, straightness
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity


# ============================================================================ Part A
def mse(ref, img):
    """Mean Squared Error: (1/N) Σ (I_ref − I)²."""
    return float(np.mean((ref.astype(np.float64) - img.astype(np.float64)) ** 2))


def psnr(ref, img, peak=255.0):
    """Peak Signal-to-Noise Ratio: 10·log10(peak² / MSE) [dB]. Higher = less noise."""
    m = mse(ref, img)
    return float("inf") if m == 0 else float(10 * np.log10(peak ** 2 / m))


def ssim(ref, img):
    """Structural Similarity (Wang et al. 2004): luminance·contrast·structure, in [−1, 1]."""
    return float(structural_similarity(ref, img, data_range=255, channel_axis=-1 if ref.ndim == 3 else None))


def edge_preservation_index(ref, img):
    """EPI (Sattar et al. 1997): normalised correlation of the Laplacian high-pass of the clean
    and the filtered image. 1 = edges perfectly preserved, 0 = edges destroyed."""
    a = cv2.Laplacian(ref.astype(np.float64), cv2.CV_64F)
    b = cv2.Laplacian(img.astype(np.float64), cv2.CV_64F)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def edge_f1(ref, img, low=50, high=150, tol=1):
    """F1 score between Canny edge maps of ref and img (a match may be off by `tol` pixels)."""
    e_ref = cv2.Canny(ref, low, high) > 0
    e_img = cv2.Canny(img, low, high) > 0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    ref_d = cv2.dilate(e_ref.astype(np.uint8), k) > 0
    img_d = cv2.dilate(e_img.astype(np.uint8), k) > 0
    precision = (e_img & ref_d).sum() / max(e_img.sum(), 1)
    recall = (e_ref & img_d).sum() / max(e_ref.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def sharpness(img):
    """Variance of the Laplacian – a standard no-reference focus/sharpness measure."""
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def filter_report(ref, img):
    """All Part-A metrics in one dict."""
    return dict(PSNR=psnr(ref, img), SSIM=ssim(ref, img), MSE=mse(ref, img),
                EPI=edge_preservation_index(ref, img), EdgeF1=edge_f1(ref, img))


# ============================================================================ Part B
def disparity_density(disp):
    """% of pixels for which the matcher returned a disparity (invalid pixels are NaN)."""
    return float(100.0 * np.mean(np.isfinite(disp)))


def lr_consistency(disp_left, disp_right, tol=1.0):
    """Left-right check: a left pixel x with disparity d must map to x−d in the right image,
    whose (right-referenced) disparity should also be ≈ d. Returns % of valid pixels passing."""
    h, w = disp_left.shape
    valid = np.isfinite(disp_left)
    xr = np.round(np.arange(w)[None, :] - np.nan_to_num(disp_left)).astype(int)
    inside = valid & (xr >= 0) & (xr < w)
    d_r = np.full_like(disp_left, np.nan)
    rows = np.broadcast_to(np.arange(h)[:, None], (h, w))
    d_r[inside] = disp_right[rows[inside], xr[inside]]
    ok = inside & np.isfinite(d_r) & (np.abs(disp_left - d_r) <= tol)
    return float(100.0 * ok.sum() / max(valid.sum(), 1))


def bad_pixel_rate(disp, gt, gt_valid, threshold=1.0, count_missing=True):
    """Middlebury 'bad-δ' metric: % of GT pixels with |d − d_gt| > δ.
    If count_missing, pixels without an estimate are counted as bad (fair to sparse methods)."""
    est = np.isfinite(disp)
    err = np.abs(np.nan_to_num(disp) - np.nan_to_num(gt)) > threshold
    bad = gt_valid & ((est & err) | (~est if count_missing else False))
    denom = gt_valid.sum() if count_missing else (gt_valid & est).sum()
    return float(100.0 * bad.sum() / max(denom, 1))


def disparity_epe(disp, gt, gt_valid):
    """End-point error = mean |d − d_gt| over pixels that have both estimate and GT."""
    m = gt_valid & np.isfinite(disp)
    return float(np.mean(np.abs(disp[m] - gt[m])))


def disparity_rmse(disp, gt, gt_valid):
    m = gt_valid & np.isfinite(disp)
    return float(np.sqrt(np.mean((disp[m] - gt[m]) ** 2)))


def stereo_report(disp, gt=None, gt_valid=None, disp_right=None):
    r = {"Valid_%": disparity_density(disp)}
    if disp_right is not None:
        r["LR_consistency_%"] = lr_consistency(disp, disp_right)
    if gt is not None:
        r["Bad1.0_%"] = bad_pixel_rate(disp, gt, gt_valid, 1.0)
        r["Bad2.0_%"] = bad_pixel_rate(disp, gt, gt_valid, 2.0)
        r["Bad1.0_on_estimated_%"] = bad_pixel_rate(disp, gt, gt_valid, 1.0, count_missing=False)
        r["EPE_px"] = disparity_epe(disp, gt, gt_valid)
        r["RMSE_px"] = disparity_rmse(disp, gt, gt_valid)
    return r


def epipolar_distance(F, p1, p2):
    """Symmetric point-to-epipolar-line distance [px]; the constraint is x2ᵀ F x1 = 0."""
    x1 = np.hstack([p1, np.ones((len(p1), 1))])
    x2 = np.hstack([p2, np.ones((len(p2), 1))])
    l2 = x1 @ F.T          # epipolar lines in image 2: l' = F x
    l1 = x2 @ F            # epipolar lines in image 1: l  = Fᵀ x'
    num = np.abs(np.sum(x2 * l2, axis=1))
    return 0.5 * (num / np.hypot(l2[:, 0], l2[:, 1]) + num / np.hypot(l1[:, 0], l1[:, 1]))


def sampson_error(F, p1, p2):
    """Sampson (first-order geometric) error in pixels:
    sqrt( (x2ᵀFx1)² / ((Fx1)₁² + (Fx1)₂² + (Fᵀx2)₁² + (Fᵀx2)₂²) )."""
    x1 = np.hstack([p1, np.ones((len(p1), 1))])
    x2 = np.hstack([p2, np.ones((len(p2), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / den)


def epipolar_line_angle(F, p1):
    """Angle of each epipolar line in image 2, wrapped to [−90°, 90°) (0° = horizontal).
    For a rectified pair every line must be horizontal."""
    l2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    ang = np.degrees(np.arctan2(-l2[:, 0], l2[:, 1]))
    return (ang + 90.0) % 180.0 - 90.0


# ============================================================================ Part C
def inlier_ratio(mask):
    return float(100.0 * np.mean(mask))


def reprojection_errors(H, src, dst):
    """‖dst − H·src‖ for every correspondence [px]."""
    proj = cv2.perspectiveTransform(np.float32(src).reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.linalg.norm(proj - dst, axis=1)


def reprojection_rmse(H, src, dst):
    e = reprojection_errors(H, src, dst)
    return float(np.sqrt(np.mean(e ** 2)))


def overlap_rmse(warped, masks):
    """Photometric RMSE between consecutive warped images inside their overlap
    (lower = better geometric alignment; also includes exposure differences)."""
    vals = []
    for i in range(len(warped) - 1):
        ov = (masks[i] > 0) & (masks[i + 1] > 0)
        if ov.sum() < 100:
            continue
        a = cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY)[ov].astype(np.float32)
        b = cv2.cvtColor(warped[i + 1], cv2.COLOR_BGR2GRAY)[ov].astype(np.float32)
        vals.append(np.sqrt(np.mean((a - b) ** 2)))
    return float(np.mean(vals)) if vals else float("nan")


def black_border_ratio(pano):
    """% of the panorama's bounding box that is empty (black) – 0 after a perfect crop."""
    return float(100.0 * np.mean(pano.max(axis=2) == 0))


def straightness(pano):
    """How 'flat' the panorama is: vertical spread (px) of the top content edge relative to the
    height. A bow-tie / curved panorama has a large value, a flat one ≈ 0."""
    valid = pano.max(axis=2) > 0
    cols = np.where(valid.any(axis=0))[0]
    top = np.array([np.argmax(valid[:, c]) for c in cols])
    bottom = np.array([valid.shape[0] - 1 - np.argmax(valid[::-1, c]) for c in cols])
    spread = (np.percentile(top, 95) - np.percentile(top, 5) +
              np.percentile(bottom, 95) - np.percentile(bottom, 5)) / 2
    return float(100.0 * spread / pano.shape[0])
