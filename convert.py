import open3d as o3d
import numpy as np

# 1. 讀入只有 XYZ 的 PCD
pcd = o3d.io.read_point_cloud("lidar-01.pcd")

# 2. 估算法線（knn=30 可自行調整）
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

# 3. 統一賦白色（RGB 0–1 浮點）
white = np.ones((len(pcd.points), 3), dtype=np.float64)
pcd.colors = o3d.utility.Vector3dVector(white)

# 4. 寫出 binary_little_endian PLY
o3d.io.write_point_cloud(
    "output.ply",
    pcd,
    write_ascii=False,   # binary little endian
    compressed=False     # 不做壓縮
)
