"""Run Part A, B and C in one go. All figures/tables are written to results/part_*/.

    python run_all.py
Part A: my own photo   data/own_photos/part_a/IMG_3115.jpg
Part B: Middlebury     data/middlebury/ (Venus, Teddy – from the course link)
Part C: my own photos  data/own_photos/part_c/ (4 overlapping photos, order found automatically)"""
import time

import part_a_filtering
import part_b_stereo
import part_c_stitching

if __name__ == "__main__":
    t0 = time.time()
    part_a_filtering.main(part_a_filtering.PART_A_IMAGE, tag="own_photo")
    part_b_stereo.main()
    part_c_stitching.main()
    print(f"\nDone in {time.time() - t0:.1f}s – see the results/ folder.")
