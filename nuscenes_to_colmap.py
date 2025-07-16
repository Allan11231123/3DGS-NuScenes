import os
import shutil
import argparse
import numpy as np
import open3d as o3d
from open3d.geometry import PointCloud
from nuscenes.nuscenes import NuScenes
import scripts.process_nuscenes as process_nuscenes
import scripts.colmap_utils as colmap_utils
from scripts.utils import SensorParameters, get_novel_cam_params
from copy import deepcopy
from pyquaternion import Quaternion

def parse_args():
    parser = argparse.ArgumentParser(description="Process NuScenes dataset for COLMAP and novel view synthesis.")
    parser.add_argument("--scene_idx", type=int, default=1, help="Scene index to process.")
    parser.add_argument("--set_size", type=int, default=5, help="Number of samples per set. Use <1 to take entire sequence as a single sample.")
    parser.add_argument("--samples_per_scene", type=int, default=1, help="Number of samples to process per scene.")
    parser.add_argument("--use_lidar", action="store_true", help="Use LiDAR data if specified.")
    parser.add_argument("--dataroot", type=str, default="data/sets/nuscenes", help="Path to NuScenes dataset root.")
    return parser.parse_args()

def setup_directories(directory_path, use_lidar):
    sub_dirs = ["sparse/0", "manual_sparse", "novel", "depth", "images"]
    if use_lidar:
        sub_dirs.append("lidar")
        sub_dirs.append("project_info")
    
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(directory_path, sub_dir), exist_ok=True)

def gt_pointcloud(pc:np.ndarray,reference_camera_info:list): #TODO: Implement this function
    """
    Get the ground truth point cloud from the NuScenes dataset(with RGB value).
    """
    target_pc = deepcopy(pc)
    points = pc.T[:, :3]  # Assuming pc is in shape (N, 4) with RGB values in the last column
    print(f"Shape of point cloud (should be (N,3)): {points.shape}")
    pc_numpy = pc.T[:, :3]
    o3d_pcl = o3d.geometry.PointCloud()
    o3d_pcl.points = o3d.utility.Vector3dVector(pc_numpy)
    return o3d_pcl
### multi cameras decision
## step1: projecting and filtering which camera can see the point
# points_hom: (N,4)，LiDAR 坐标系下的齐次点云
# for i in range(M):
#     T = T_l2c_list[i]           # 第 i 台相机的 4×4 外参
#     K = K_list[i]               # 第 i 台相机的 3×3 内参
#     # 1) 变换到该相机坐标系
#     pts_cam = (T @ points_hom.T).T   # (N,4)
#     xyz = pts_cam[:, :3]
#     # 2) 投影到像素
#     uv_h = (K @ xyz.T).T             # (N,3)
#     uv = uv_h[:, :2] / uv_h[:, 2:3]  # (N,2)
#     # 3) 深度 & 视野筛选
#     valid_i = (
#         (xyz[:,2] > 0)                  # 在相机前方
#         & (uv[:,0] >= 0) & (uv[:,0] < width_i)
#         & (uv[:,1] >= 0) & (uv[:,1] < height_i)
#     )
#     camera_assignments[i] = valid_i    # 布尔数组，标记哪些点“看得见”该相机
## step2: decide the point belongs to which camera
# 预先计算每台相机的“光轴方向” d_i（ego frame）
# 若你有 calibrated_sensor 的旋转矩阵 R_s2e，第 3 列就是光轴向量
# d_i = [R_s2e_list[i][:,2] for i in range(M)]  

# 计算每个点到各相机光轴的夹角
# assignments = np.full(N, -1, dtype=int)
# for j in range(N):
#     candidates = [i for i in range(M) if camera_assignments[i][j]]
#     if not candidates:
#         continue    # 该点不在任何相机视野内
#     angles = []
#     for i in candidates:
#         # 向量从相机坐标原点到该点（在 ego frame）
#         v = pts_ego[j] - t_e_list[i]
#         cos_theta = (v @ d_i[i]) / (np.linalg.norm(v) * np.linalg.norm(d_i[i]))
#         angles.append(np.arccos(np.clip(cos_theta, -1, 1)))
#     # 取最小者
#     best_cam = candidates[np.argmin(angles)]
#     assignments[j] = best_cam
## step3: get rgb value
# image = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
def process_scene(nusc, scene_idx, set_size, samples_per_scene, use_lidar, dataroot):
    directory_path = os.path.join(os.getcwd(), "data/colmap_data", f"scene-{scene_idx}")
    print(f"Created {directory_path}")

    os.makedirs(directory_path, exist_ok=True)
    
    scene = nusc.scene[scene_idx]
    scene_sample = nusc.get('sample', scene['first_sample_token'])
    cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    lidar = 'LIDAR_TOP' if use_lidar else None
    sensor_params = SensorParameters(nusc, scene_sample, sensors=cameras)
    
    # initialize variables
    sample_count, set_fill, img_id = 1, 0, 1
    
    print(f"Processing scene. name: {scene['name']} Token: {scene['token']}")
    while scene_sample['next'] != '' and sample_count <= samples_per_scene:
        sample_dir = os.path.join(directory_path, f"sample-{sample_count:02}")
        setup_directories(sample_dir, use_lidar)
        
        colmap_manual_sparse_folder = os.path.join(sample_dir, "manual_sparse")
        points3D_file = os.path.join(colmap_manual_sparse_folder, "points3D.txt")
        open(points3D_file, "w").close()  # Create empty points3D.txt file
        
        # Write camera intrinsics
        colmap_utils.write_intrinsics_file_nuscenes(colmap_manual_sparse_folder, sensor_params, img_width=1600, img_height=900)
        
        transform_vectors, _ = sensor_params.global_pose(scene_sample)
        novel_cam_params = get_novel_cam_params(sensor_params)
        novel_cam_intrinsics = [cam.intrinsics for cam in novel_cam_params]
        
        colmap_utils.write_intrinsics_file_novelcam(os.path.join(sample_dir, "novel"), novel_cam_intrinsics)
        colmap_utils.write_extrinsics_file_novelcam(os.path.join(sample_dir, "novel"), "w", transform_vectors=transform_vectors)
        
        save_depth_paths = []
        
        pc_data = None
        reference_camera_infos = list()
        # Copy LiDAR data if required
        if use_lidar:
            lidar_token = scene_sample['data'][lidar]
            lidar_data = nusc.get('sample_data', lidar_token)
            if not os.path.exists(os.path.join(sample_dir, "project_info", "lidar_extrinsics.txt")):
                lidar_extrinsics = nusc.get("calibrated_sensor", lidar_data['calibrated_sensor_token'])
                lidar_translation = np.array(lidar_extrinsics['translation'])
                lidar_rotation = np.array(lidar_extrinsics['rotation'])
                with open(os.path.join(sample_dir, "project_info", "lidar_extrinsics.txt"), "w") as f:
                    f.write(f"lidar_translation:\n{lidar_translation[0]},{lidar_translation[1]},{lidar_translation[2]}\n")
                    f.write(f"lidar_rotation:\n{lidar_rotation[0]},{lidar_rotation[1]},{lidar_rotation[2]}\n")
            pcl_path = os.path.join(dataroot, nusc.get('sample_data', lidar_token)['filename'])
            pc_data = np.asarray(o3d.io.read_point_cloud(pcl_path).points)
            print(f"Shape of point cloud (should be (4,N)): {pc_data.shape}")
            cs_lidar = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            rotation = Quaternion(cs_lidar['rotation']).rotation_matrix
            translation = np.array(cs_lidar['translation']).reshape(3, 1)
            pc_data = rotation.dot(pc_data[:3, :]) + translation # pc_data is now in ego frame
            pc = process_nuscenes.project_pointcloud_global_frame(nusc, pcl_path, lidar_data)
            pc_numpy = pc.points.T[:, :3]
            o3d_pcl = o3d.geometry.PointCloud()
            o3d_pcl.points = o3d.utility.Vector3dVector(pc_numpy)
            o3d.io.write_point_cloud(os.path.join(sample_dir, "lidar", f"lidar-{sample_count:02}.pcd"), o3d_pcl)
        
        # Copy images and write extrinsics
        with open(os.path.join(colmap_manual_sparse_folder, "images.txt"), "w") as file:
            for idx, tv in enumerate(transform_vectors):
                if not os.path.exists(os.path.join(sample_dir, "project_info", f"image_extrinsics_{cameras[idx]}.txt")):
                    camera_data = nusc.get('sample_data', scene_sample['data'][cameras[idx]])
                    calibrated_camera = nusc.get("calibrated_sensor", camera_data['calibrated_sensor_token'])
                    camera_intrinsics = np.array(calibrated_camera['camera_intrinsic'])
                    camera_rotation = np.array(calibrated_camera['rotation'])
                    camera_translation = np.array(calibrated_camera['translation'])
                    with open(os.path.join(sample_dir, "project_info", f"image_extrinsics_{cameras[idx]}.txt"), "w") as f:
                        f.write("camera_extrinsics:\n")
                        f.write(f"translation:\n{camera_translation[0]},{camera_translation[1]},{camera_translation[2]}\n")
                        f.write(f"rotation:\n{camera_rotation[0]},{camera_rotation[1]},{camera_rotation[2]},{camera_rotation[3]}\n")
                    with open(os.path.join(sample_dir, "project_info", f"image_intrinsics_{cameras[idx]}.txt"), "w") as f:
                        f.write(f"camera_intrinsics:\n")
                        f.write(f"{camera_intrinsics[0][0]},{camera_intrinsics[0][1]},{camera_intrinsics[0][2]}\n")
                        f.write(f"{camera_intrinsics[1][0]},{camera_intrinsics[1][1]},{camera_intrinsics[1][2]}\n")
                        f.write(f"{camera_intrinsics[2][0]},{camera_intrinsics[2][1]},{camera_intrinsics[2][2]}\n")
                filename = nusc.get('sample_data', scene_sample['data'][cameras[idx]])['filename']
                source_path = os.path.join(dataroot, filename)
                target_path = os.path.join(sample_dir, "images", f"image-{sample_count:02}-{img_id:02}.jpg")
                if img_id<=6:
                    reference_camera_infos.append({"image_path":target_path,"intrinsics":camera_intrinsics,"rotation":camera_rotation,"translation":camera_translation})
                    reference_path = os.path.join(sample_dir, "project_info", "reference_images")
                    if not os.path.exists(reference_path):
                        os.makedirs(reference_path, exist_ok=True)
                    project_reference_path = os.path.join(reference_path, f"image-{sample_count:02}-{img_id:02}.jpg")
                    shutil.copy(source_path, project_reference_path)
                shutil.copy(source_path, target_path)
                file.write(f"{img_id} {tv[0]} {tv[1]} {tv[2]} {tv[3]} {tv[4]} {tv[5]} {tv[6]} {idx + 1} image-{sample_count:02}-{img_id:02}.jpg\n\n")
                save_depth_paths.append(os.path.join(sample_dir, "depth", f"image-{sample_count:02}-{img_id:02}.png"))
                img_id += 1
        # Project point cloud to image to get RGB values
        if pc_data is not None:
            gt_pc = gt_pointcloud(pc_data, reference_camera_infos)
            o3d.io.write_point_cloud(os.path.join(sample_dir, "lidar", f"lidar-{sample_count:02}.ply"), gt_pc)
        process_nuscenes.img_lidar_depth_map(nusc, scene_sample, cameras, plot_img=False, save_paths=save_depth_paths)
        print(f"Processed sample {sample_count}, set {set_fill}")

        scene_sample = nusc.get('sample', scene_sample['next'])
        
        set_fill += 1
        if set_size > 1 and set_fill % set_size == 0:
            sample_count += 1
            set_fill = 0
            img_id = 1
        
        colmap_utils.write_batch_file(sample_dir, colmap_manual_sparse_folder, os.path.join(sample_dir, "sparse/0"))
    
    # Delete Temp file
    colmap_utils.write_intrinsics_file_nuscenes(delete_temp=True)
    print(f"Processing lidar extrinsics, cameras extrinsics and intrinsics for scene {scene['name']}")
    print(f"Finished processing scene {scene['name']}")

def main():
    args = parse_args()
    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    process_scene(nusc, args.scene_idx, args.set_size, args.samples_per_scene, args.use_lidar, args.dataroot)

if __name__ == "__main__":
    main()