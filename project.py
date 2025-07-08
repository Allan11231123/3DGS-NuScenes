import numpy as np
import open3d as o3d
import cv2

# --- 1. 讀點雲 ---
pcd = o3d.io.read_point_cloud("input.pcd")
points = np.asarray(pcd.points)
N = len(points)

# 準備一個 colors 陣列
colors = np.zeros((N, 3), dtype=np.float64)

# --- 2. 設定所有相機的參數 ---
# 假設你有 M 個相機，各自的 K, R, t, 影像路徑存在下面 list 裡
Ks = [K_cam0, K_cam1, ...]          # 每個 K_cam 是 3×3 numpy array
RTs = [RT_cam0, RT_cam1, ...]       # 每個 RT_cam 是 3×4 numpy array [R|t]
imgs = [cv2.imread("cam0.png"), ...]  # BGR 讀進來後會轉成 RGB

# --- 3. 多相機投影取色 ---
# 我們先記每點目前最小的 z_depth，用來選最「正面」的相機
best_z = np.full(N, np.inf)

for K, RT, img in zip(Ks, RTs, imgs):
    # 拆 R, t
    R = RT[:, :3]          # 3×3
    t = RT[:, 3].reshape(3,1)  # 3×1

    # 把所有點一次性轉到相機座標
    # p_cam = R @ p_world + t
    p_cam = (R @ points.T) + t   # shape = (3, N)
    x, y, z = p_cam

    # 投影到像平面
    uv = (K @ p_cam)             # (3×3)×(3×N) → (3×N)
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]

    # 四捨五入成整數像素
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)

    h, w = img.shape[:2]
    # 篩出「有效」的點：在影像範圍內 & 前方 z>0
    valid = (
        (ui >= 0) & (ui < w) &
        (vi >= 0) & (vi < h) &
        (z > 0)
    )

    # 對於每個 valid 點，如果這個相機對該點的 z 比先前更小（更正面），就更新 color
    idx = np.where(valid)[0]
    for i in idx:
        if z[i] < best_z[i]:
            best_z[i] = z[i]
            b, g, r = img[vi[i], ui[i]]   # OpenCV 預設 BGR
            colors[i] = np.array([r, g, b]) / 255.0

# --- 4. 寫回點雲並輸出 PLY ---
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(
    "colored_output.ply",
    pcd,
    write_ascii=False
)
