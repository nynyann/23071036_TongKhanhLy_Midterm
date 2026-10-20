"""Interactive 3D viewer for the reconstructed point cloud (needs: pip install open3d).

    python view_point_cloud.py                                   # SGBM cloud
    python view_point_cloud.py results/part_b/point_cloud_bm.ply # BM cloud
"""
import sys

import open3d as o3d

path = sys.argv[1] if len(sys.argv) > 1 else "results/part_b/point_cloud_sgbm.ply"
pcd = o3d.io.read_point_cloud(path)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)   # clean flying pixels
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # image coords -> viewer coords
print(pcd)
o3d.visualization.draw_geometries([pcd], window_name="Stereo point cloud")
