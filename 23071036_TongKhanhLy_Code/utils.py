"""Shared helpers for the INS3155 midterm project (I/O, tables, figures)."""
import csv
import os
import time
from contextlib import contextmanager

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend: figures are written to disk
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")


def out_dir(part: str) -> str:
    """Return (and create) results/<part>/."""
    path = os.path.join(RESULTS_DIR, part)
    os.makedirs(path, exist_ok=True)
    return path


def imread(path: str, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def resize_max(img, max_side: int):
    """Downscale so that the longer side is <= max_side (keeps aspect ratio)."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def bgr2rgb(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_fig(fig, path: str, dpi: int = 130):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.relpath(path, ROOT)}")


def write_table(rows, path_no_ext: str, float_fmt: str = "{:.4f}"):
    """Write a list of dicts as both CSV and a Markdown table (handy for the report)."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path_no_ext + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return float_fmt.format(v) if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    with open(path_no_ext + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {os.path.relpath(path_no_ext, ROOT)}.csv/.md")


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = (time.perf_counter() - t0) * 1000.0  # milliseconds
