import argparse
import os
import numpy as np
import open3d as o3d
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Process point cloud and color it using multi-camera projections.")
    parser.add_argument("--input_pcd", type=str, required=True, help="Input point cloud file in PCD format.")
    parser.add_argument("--project_info", type=str, required=True, help="Folder that contains projecting information.")
    parser.add_argument("--output_ply", type=str, default="output.ply", help="Output colored point cloud file in PLY format.")
    return parser.parse_args()

def main():
    args = parse_args()
    # --- reading point cloud ---
    pcd = o3d.io.read_point_cloud(args.input_pcd)
    points = np.asarray(pcd.points)
    N = len(points)

    # prepare a colors array
    colors = np.zeros((N, 3), dtype=np.float64)

    # --- Set up camera parameters ---
    Ks = [K_cam0, K_cam1, ...]       
    RTs = [RT_cam0, RT_cam1, ...]
    imgs = [cv2.imread("cam0.png"), ...]
    # --- Multi-camera projection and color assignment ---
    best_z = np.full(N, np.inf)

    for K, RT, img in zip(Ks, RTs, imgs):
        # split R, t
        R = RT[:, :3]          # 3×3
        t = RT[:, 3].reshape(3,1)  # 3×1

        # p_cam = R @ p_world + t
        p_cam = (R @ points.T) + t   # shape = (3, N)
        x, y, z = p_cam

        # project to image plane
        uv = (K @ p_cam)             # (3×3)×(3×N) → (3×N)
        u = uv[0] / uv[2]
        v = uv[1] / uv[2]

        # round to integer pixel
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)

        h, w = img.shape[:2]
        # filter valid points: within image bounds & in front of camera (z > 0)
        valid = (
            (ui >= 0) & (ui < w) &
            (vi >= 0) & (vi < h) &
            (z > 0)
        )

        # For each valid point, if this camera's z is smaller (more frontal) than before, update color
        idx = np.where(valid)[0]
        for i in idx:
            if z[i] < best_z[i]:
                best_z[i] = z[i]
                b, g, r = img[vi[i], ui[i]]   # OpenCV defaults to BGR
                colors[i] = np.array([r, g, b]) / 255.0

    # --- Write back the point cloud and output PLY ---
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(
        "colored_output.ply",
        pcd,
        write_ascii=False
    )
