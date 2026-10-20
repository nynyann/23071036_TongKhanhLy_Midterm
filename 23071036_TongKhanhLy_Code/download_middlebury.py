"""Download the Middlebury data the lecturer linked (https://vision.middlebury.edu/flow/data/)
into data/middlebury/.  Run once:  python download_middlebury.py   (needs internet)."""
import io
import os
import urllib.request
import zipfile

BASE = "https://vision.middlebury.edu/flow/data/comp/zip/"
FILES = ["other-color-twoframes.zip",   # Venus frame10/frame11 (+ other scenes)
         "other-gt-flow.zip",           # ground-truth flow (.flo) -> Venus disparity
         "eval-color-twoframes.zip"]    # Teddy frame10/frame11 (GT is hidden)
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "middlebury")

os.makedirs(DEST, exist_ok=True)
for name in FILES:
    print("downloading", name, "...")
    with urllib.request.urlopen(BASE + name) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DEST)
print("done ->", DEST)
for root, _, files in os.walk(DEST):
    if os.path.basename(root) in ("Venus", "Teddy"):
        print(" ", os.path.relpath(root, DEST), sorted(files))
