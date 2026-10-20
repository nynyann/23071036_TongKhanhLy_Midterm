"""
Part A – Image Filtering and Enhancement
========================================
Input: my own photograph of the place where I live (data/own_photos/part_a/IMG_3115.jpg).
Noise models: additive Gaussian noise (σ = 20) and salt-and-pepper noise (5 %).
Filters: Mean 3/5/7, Gaussian 3/5/7 (+ σ = 0.5 and 2.0 at 5x5), Median 3/5/7,
         Laplacian sharpening (alone and after denoising).

Metrics (against the clean image): PSNR, SSIM, MSE, execution time (median of 7 runs),
plus EPI (edge preservation index) and Edge-F1 for edge preservation.

Usage:
    python part_a_filtering.py                       # own photo
    python part_a_filtering.py --image other.jpg     # any image
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from metrics import filter_report, psnr, sharpness
from utils import DATA_DIR, imread, out_dir, resize_max, save_fig, timer, write_table

RNG_SEED = 42
PART_A_IMAGE = os.path.join(DATA_DIR, "own_photos", "part_a", "IMG_3115.jpg")


# ----------------------------------------------------------------------------- noise models
def add_gaussian_noise(img, sigma=20.0, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount=0.05, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    r = rng.random(img.shape[:2])
    noisy[r < amount / 2] = 0          # pepper
    noisy[r > 1 - amount / 2] = 255    # salt
    return noisy


# ----------------------------------------------------------------------------- filters
def mean_filter(img, k):
    """Box filter: every output pixel is the average of a k x k window."""
    return cv2.blur(img, (k, k))


def gaussian_filter(img, k, sigma=0):
    """Weighted average with a 2-D Gaussian kernel G(x,y)=exp(-(x²+y²)/2σ²)/(2πσ²).
    sigma=0 lets OpenCV derive σ = 0.3*((k-1)*0.5 - 1) + 0.8 from k."""
    return cv2.GaussianBlur(img, (k, k), sigma)


def median_filter(img, k):
    """Non-linear: replaces each pixel with the median of its k x k neighbourhood."""
    return cv2.medianBlur(img, k)


def laplacian_sharpen(img, alpha=1.0, ksize=1):
    """g = f - α ∇²f with the 4-neighbour kernel [[0,1,0],[1,-4,1],[0,1,0]] (ksize=1).
    The kernel centre is negative, so subtracting the Laplacian boosts edges."""
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=ksize)
    return np.clip(f - alpha * lap, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- metrics
def evaluate(ref, img):
    """PSNR, SSIM, EPI, Edge-F1 (+MSE) – implemented in metrics.py."""
    return filter_report(ref, img)


# ----------------------------------------------------------------------------- experiment
KERNELS = (3, 5, 7)
GAUSS_SIGMAS = {3: 0.8, 5: 1.1, 7: 1.4}          # OpenCV's default σ for each kernel size
EXTRA_SIGMAS = (0.5, 2.0)                        # σ study at a fixed 5x5 kernel


def build_filter_bank():
    """(name, function) – kernel sizes 3/5/7 for every filter and several σ for the Gaussian."""
    bank = []
    for k in KERNELS:
        bank.append((f"Mean {k}x{k}", lambda x, k=k: mean_filter(x, k)))
    for k in KERNELS:
        s = GAUSS_SIGMAS[k]
        bank.append((f"Gaussian {k}x{k} σ={s}", lambda x, k=k, s=s: gaussian_filter(x, k, s)))
    for s in EXTRA_SIGMAS:
        bank.append((f"Gaussian 5x5 σ={s}", lambda x, s=s: gaussian_filter(x, 5, s)))
    for k in KERNELS:
        bank.append((f"Median {k}x{k}", lambda x, k=k: median_filter(x, k)))
    # Laplacian sharpening alone on a noisy image amplifies noise; the useful pipeline is
    # "denoise first, then sharpen". Both are reported so the difference is visible.
    bank.append(("Laplacian sharpen (raw)", lambda x: laplacian_sharpen(x, 0.5)))
    bank.append(("Gaussian 5x5 + Laplacian", lambda x: laplacian_sharpen(gaussian_filter(x, 5, 1.1), 0.5)))
    bank.append(("Median 3x3 + Laplacian", lambda x: laplacian_sharpen(median_filter(x, 3), 0.5)))
    return bank


def timed(fn, x, repeats=7):
    """Run fn(x) several times and return (result, median runtime in ms) – stable timings."""
    times = []
    for _ in range(repeats):
        t = {}
        with timer(t, "ms"):
            res = fn(x)
        times.append(t["ms"])
    return res, float(np.median(times))


def side_by_side(clean, noisy, outputs, title, path, crop=None):
    """Original | Noisy | each filter, optionally zoomed into `crop` = (y0, y1, x0, x1)."""
    panels = [("Original", clean), ("Noisy", noisy)] + outputs
    if crop is not None:
        y0, y1, x0, x1 = crop
        panels = [(n, im[y0:y1, x0:x1]) for n, im in panels]
    ref = panels[0][1]
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, im) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        label = name if name == "Original" else f"{name}\nPSNR {psnr(ref, im):.2f} dB"
        ax.set_title(label, fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    save_fig(fig, path)


def line_profile(clean, noisy, outputs, row, path, title):
    """Intensity profile along one image row: shows how each filter treats an edge."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(noisy[row], color="0.75", lw=0.8, label="Noisy")
    ax.plot(clean[row], color="black", lw=2, label="Original")
    for name, im in outputs:
        ax.plot(im[row], lw=1.2, label=name)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8)
    save_fig(fig, path)


def bar_chart(rows, path):
    """PSNR, SSIM and runtime of every filter for both noise models."""
    noises = list(dict.fromkeys(r["Noise"] for r in rows))
    filters = [r["Filter"] for r in rows if r["Noise"] == noises[0]]
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    x = np.arange(len(filters))
    w = 0.38
    for ax, metric, unit in zip(axes, ("PSNR", "SSIM", "Time_ms"), ("dB", "", "ms")):
        for i, noise in enumerate(noises):
            vals = [next(r[metric] for r in rows if r["Noise"] == noise and r["Filter"] == f) for f in filters]
            ax.bar(x + (i - 0.5) * w, vals, w, label=noise)
        ax.set_ylabel(f"{metric} {unit}".strip())
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(filters, rotation=40, ha="right", fontsize=9)
    fig.suptitle("PSNR, SSIM (higher is better) and runtime (lower is better) of every filter", fontweight="bold")
    save_fig(fig, path)


def kernel_size_plot(rows, path):
    """Parameter sensitivity: PSNR and EPI versus kernel size for the three smoothing filters."""
    noises = list(dict.fromkeys(r["Noise"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric in zip(axes, ("PSNR", "EPI")):
        for noise, ls in zip(noises, ("-", "--")):
            for fam in ("Mean", "Gaussian", "Median"):
                vals = []
                for k in KERNELS:
                    name = f"Gaussian {k}x{k} σ={GAUSS_SIGMAS[k]}" if fam == "Gaussian" else f"{fam} {k}x{k}"
                    vals.append(next(r[metric] for r in rows if r["Noise"] == noise and r["Filter"] == name))
                ax.plot(KERNELS, vals, ls, marker="o", label=f"{fam} – {noise}")
        ax.set_xticks(KERNELS)
        ax.set_xlabel("kernel size k (k×k)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Kernel-size sensitivity: more smoothing removes more noise but erases edges", fontweight="bold")
    save_fig(fig, path)


def sharpening_demo(clean, out):
    """Laplacian as *enhancement*: recover detail of a slightly blurred (noise-free) image."""
    blurred = gaussian_filter(clean, 5, 1.5)
    results = [("Blurred input", blurred)]
    rows = []
    for alpha in (0.5, 1.0, 2.0):
        sharp = laplacian_sharpen(blurred, alpha)
        results.append((f"Sharpened α={alpha}", sharp))
    for name, im in results:
        m = evaluate(clean, im)
        m["Sharpness (var ∇²)"] = sharpness(im)
        rows.append({"Image": name, **m})
    rows.insert(0, {"Image": "Original", **evaluate(clean, clean),
                    "Sharpness (var ∇²)": sharpness(clean)})
    write_table(rows, os.path.join(out, "laplacian_sharpening_metrics"))

    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4.2 * (len(results) + 1), 4))
    for ax, (name, im) in zip(axes, [("Original", clean)] + results):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Laplacian sharpening as enhancement (g = f − α∇²f)", fontweight="bold")
    save_fig(fig, os.path.join(out, "A3_laplacian_sharpening.png"))

    # halo / ringing: profile across the strongest horizontal edge of the image
    gx = np.abs(cv2.Sobel(clean.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    y, x = np.unravel_index(np.argmax(cv2.blur(gx, (1, 9))), gx.shape)
    x0, x1 = max(x - 25, 0), min(x + 25, clean.shape[1])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(x0, x1), clean[y, x0:x1], color="black", lw=2.5, label="Original")
    for name, im in results:
        ax.plot(range(x0, x1), im[y, x0:x1], lw=1.4, label=name)
    ax.set_xlabel("x (pixel)"); ax.set_ylabel("Intensity")
    ax.set_title(f"Row {y}: overshoot on both sides of the edge = halo / ringing (grows with α)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    save_fig(fig, os.path.join(out, "A6_halo_profile.png"))


def main(image_path, tag=None, crop=None):
    """crop = (top, bottom) as fractions of the height, e.g. (0.45, 1.0) keeps the lower part
    of a night photo (the dark, clipped sky would make the metrics look artificially good)."""
    tag = tag or os.path.splitext(os.path.basename(image_path))[0]
    out = out_dir(os.path.join("part_a", tag))
    clean = imread(image_path, cv2.IMREAD_GRAYSCALE)
    if crop is not None:
        h = clean.shape[0]
        clean = clean[int(crop[0] * h):int(crop[1] * h)]
    clean = resize_max(clean, 800)
    print(f"[Part A] image {os.path.basename(image_path)} -> {clean.shape}")

    noises = {
        "Gaussian σ=20": add_gaussian_noise(clean, 20),
        "Salt&Pepper 5%": add_salt_pepper(clean, 0.05),
    }
    bank = build_filter_bank()
    rows = []
    h, w = clean.shape
    crop = (h // 3, h // 3 + h // 4, w // 3, w // 3 + w // 4)      # zoom window for detail

    for idx, (noise_name, noisy) in enumerate(noises.items(), start=1):
        rows.append({"Noise": noise_name, "Filter": "(no filter)", **evaluate(clean, noisy), "Time_ms": 0.0})
        outputs = []
        for name, fn in bank:
            res, ms = timed(fn, noisy)
            outputs.append((name, res))
            rows.append({"Noise": noise_name, "Filter": name, **evaluate(clean, res), "Time_ms": ms})
            fname = name.replace(" ", "_").replace("+", "plus").replace("σ=", "s")
            cv2.imwrite(os.path.join(out, f"noise{idx}_{fname}.png"), res)
        cv2.imwrite(os.path.join(out, f"noise{idx}_noisy.png"), noisy)

        tag = "gaussian" if idx == 1 else "saltpepper"
        side_by_side(clean, noisy, outputs, f"Part A – {noise_name}: original, noisy and filtered",
                     os.path.join(out, f"A1_{tag}_side_by_side.png"))
        side_by_side(clean, noisy, outputs, f"Part A – {noise_name}: zoomed detail (edge preservation)",
                     os.path.join(out, f"A2_{tag}_zoom.png"), crop=crop)
        pick = [o for o in outputs if o[0] in ("Mean 5x5", "Gaussian 5x5 σ=1.1", "Median 5x5")]
        line_profile(clean, noisy, pick, h // 2, os.path.join(out, f"A4_{tag}_line_profile.png"),
                     f"Row {h // 2} intensity profile – {noise_name}")

    cv2.imwrite(os.path.join(out, "original_gray.png"), clean)
    write_table(rows, os.path.join(out, "filter_metrics"))
    bar_chart([r for r in rows if r["Filter"] != "(no filter)"], os.path.join(out, "A5_metrics_bar_chart.png"))
    kernel_size_plot(rows, os.path.join(out, "A7_kernel_size_sensitivity.png"))
    sharpening_demo(clean, out)

    # concise console summary: best filter per noise type
    for noise_name in noises:
        sub = [r for r in rows if r["Noise"] == noise_name]
        best = max(sub, key=lambda r: r["PSNR"])
        print(f"  best PSNR on {noise_name}: {best['Filter']} ({best['PSNR']:.2f} dB, SSIM {best['SSIM']:.3f})")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=PART_A_IMAGE)
    ap.add_argument("--crop", nargs=2, type=float, default=None, metavar=("TOP", "BOTTOM"),
                    help="keep rows TOP..BOTTOM (fractions of height), e.g. 0.45 1.0")
    a = ap.parse_args()
    main(a.image, tag="own_photo", crop=a.crop)
