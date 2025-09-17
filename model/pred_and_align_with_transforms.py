import os
import json
import torch
import sys
import numpy as np
import glob
import open3d as o3d
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "vggt/"))

from .run_vggt import run_model
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def find_pose_in_transforms(image_name, transforms_data):
    filename = image_name.split("/")[-1]
    filename_no_ext = filename.replace(".png", "")

    # Try to parse both possible formats
    traj_name = None
    traj_folder = None
    frame_number = None
    
    # Check for samples-rgb format
    if "_samples-rgb_" in filename_no_ext:
        parts = filename_no_ext.split("_samples-rgb_")
        if len(parts) == 2:
            traj_name = parts[0]  # e.g., "007_up_simple"
            frame_number = parts[1] + ".png"  # e.g., "040.png"
            traj_folder = "samples-rgb"
    # Check for input format
    elif "_input_" in filename_no_ext:
        parts = filename_no_ext.split("_input_")
        if len(parts) == 2:
            traj_name = parts[0]  # e.g., "001_left_simple"
            frame_number = parts[1] + ".png"  # e.g., "000.png"
            traj_folder = "input"
    else:
        # Try to handle other potential formats by looking at the last parts
        # Split by underscore and check if second-to-last part could be a folder name
        parts = filename_no_ext.split("_")
        if len(parts) >= 2:
            # Check if the second-to-last part is a known folder type
            potential_folder = parts[-2]
            if potential_folder in ["input", "samples-rgb"]:
                traj_folder = potential_folder
                frame_number = parts[-1] + ".png"
                # Reconstruct trajectory name from remaining parts
                traj_name = "_".join(parts[:-2])
            else:
                print(f"Warning: Unexpected filename format: {filename}")
                return None
        else:
            print(f"Warning: Could not parse filename format: {filename}")
            return None
    
    if not all([traj_name, traj_folder, frame_number]):
        print(f"Warning: Could not extract all required components from filename: {filename}")
        print(f"  traj_name: {traj_name}, traj_folder: {traj_folder}, frame_number: {frame_number}")
        return None
    
    
    transforms_pose = None
    for f in transforms_data["frames"]:
        file_parts = f["file_path"].split("/")
        f_file_name = file_parts[-1]  # e.g., "000.png"
        f_traj_folder = file_parts[-2]  # e.g., "input" or "samples-rgb"
        f_traj_name = file_parts[-3]  # e.g., "001_left_simple"
        
        if traj_name == f_traj_name and traj_folder == f_traj_folder and frame_number == f_file_name:
            transforms_pose = f["transform_matrix"]  # cam2world matrix
            break
    
    if transforms_pose is None:
        print(f"No matching pose found for {filename}")
        print(f"  Searched for: traj_name={traj_name}, traj_folder={traj_folder}, frame={frame_number}")
    
    return transforms_pose


def invert_pose(pose):
    """Inverts a 4x4 pose matrix."""
    R = pose[:3, :3]
    t = pose[:3, 3:]
    R_inv = R.T
    t_inv = -R_inv @ t
    pose_inv = np.eye(4)
    pose_inv[:3, :3] = R_inv
    pose_inv[:3, 3:] = t_inv
    return pose_inv


def convert_opencv_to_opengl(c2w):
    """
    Convert a camera pose from OpenCV (right-handed, y-down) to OpenGL/Blender (right-handed, y-up).
    """
    cv_to_gl = np.diag([1, -1, -1, 1])  # Flip y and z
    return c2w @ cv_to_gl


def transform_point_cloud(points, transform):
    """Apply a 4x4 transformation to a Nx3 point cloud."""
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # (N, 4)
    points_transformed = (transform @ points_h.T).T  # (N, 4)
    return points_transformed[:, :3]


def transform_pointcloud_to_gt_space(gt_c2w, pred_w2c, pointcloud_pred):
    """
    Transforms a predicted point cloud from the predicted world space (OpenCV) to GT world space (Blender/OpenGL).

    Parameters:
    - gt_c2w: (N, 4, 4) ground truth camera-to-world matrices in OpenGL convention
    - pred_w2c: (N, 4, 4) predicted world-to-camera matrices in OpenCV convention
    - pointcloud_pred: (M, 3) predicted point cloud in predicted world space

    Returns:
    - transformed point cloud in GT world space
    - transformation matrix from predicted world to GT world
    """

    def get_opencv_to_opengl():
        return np.diag([1, -1, -1, 1])  # flip Y and Z

    # Use i-th camera pair, say i = 0
    i = 0

    W2C_pred = pred_w2c[i]  # (4, 4)
    C2W_gt = gt_c2w[i]  # (4, 4)
    cv2gl = get_opencv_to_opengl()  # (4, 4)

    # Full transform: GT_C2W * CV_to_GL * W2C_pred
    T_pred_to_gt = C2W_gt @ cv2gl @ W2C_pred  # (4, 4)

    # Transform point cloud
    ones = np.ones((pointcloud_pred.shape[0], 1))
    pointcloud_h = np.concatenate([pointcloud_pred, ones], axis=1)  # (M, 4)
    pointcloud_transformed = (T_pred_to_gt @ pointcloud_h.T).T[:, :3]  # (M, 3)å

    return pointcloud_transformed, T_pred_to_gt


def run_vggt_and_align(scene_root):
    print("Initializing and loading VGGT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval()
    model = model.to(device)

    pcl_path = os.path.join(scene_root, "vggt_pcl.ply")

    # load transforms and align poses and point cloud with those
    transforms_json = os.path.join(scene_root, "transforms.json")
    transforms_data = json.load(open(transforms_json, "r"))
    # run vggt predictions
    images_folder = os.path.join(scene_root, "selected_for_vggt")

    predictions, image_names, images = run_model(images_folder, model)

    # parse data
    gt_c2w_list = []
    pred_w2c_list = []
    for img_name, pred_w2c in zip(image_names, predictions["extrinsic"]):
        gt_c2w = find_pose_in_transforms(img_name, transforms_data)
        if gt_c2w is None:
            print(f"Warning: Could not find ground truth pose for image: {img_name}")
            print(f"Image name parts: {img_name.split('/')[-1].split('_')}")
            continue
        gt_c2w = np.array(gt_c2w)
        gt_c2w_list.append(gt_c2w)
        pred_w2c = np.array(pred_w2c)
        pred_w2c = np.concatenate([pred_w2c, np.array([[0, 0, 0, 1]])], axis=0)  # (4, 4)
        pred_w2c_list.append(pred_w2c)
    pred_pointcloud = predictions["world_points_from_depth"].reshape(-1, 3)

    aligned_pts, T = transform_pointcloud_to_gt_space(gt_c2w_list, pred_w2c_list, pred_pointcloud)

    # calc scale factor
    gt_poses_cam_center = np.stack([x[:3, 3] for x in gt_c2w_list], axis=0)
    gt_poses_min = gt_poses_cam_center.min(axis=0)  # (3,)
    gt_poses_max = gt_poses_cam_center.max(axis=0)  # (3,)
    gt_poses_bbox_length = np.linalg.norm(gt_poses_max - gt_poses_min).item()
    pred_poses_cam_center = np.array([np.linalg.inv(p)[:3, 3] for p in pred_w2c_list])  # (N, 3)
    pred_poses_min = pred_poses_cam_center.min(axis=0)  # (3,)
    pred_poses_max = pred_poses_cam_center.max(axis=0)  # (3,)
    pred_poses_bbox_length = np.linalg.norm(pred_poses_max - pred_poses_min).item()
    gt_to_pred_scale = gt_poses_bbox_length / pred_poses_bbox_length

    # Step 1: Compute centers
    gt_center = gt_poses_cam_center.mean(axis=0, keepdims=True)  # (1, 3)

    # Step 2: Center aligned points
    aligned_pts_centered = aligned_pts - gt_center  # subtract center

    # Step 3: Scale
    aligned_pts_scaled = aligned_pts_centered * gt_to_pred_scale

    # Step 4: Un-center
    aligned_pts = aligned_pts_scaled + gt_center

    # save pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(aligned_pts)
    colors = images.permute(0, 2, 3, 1).contiguous().cpu().numpy().reshape(-1, 3)  # (N, 3)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.remove_non_finite_points()
    pcd = pcd.random_down_sample(0.01)
    o3d.io.write_point_cloud(pcl_path, pcd)

    # add pointcloud to transforms.json
    transforms_data["ply_file_path"] = os.path.abspath(pcl_path)

    # save transforms.json
    with open(transforms_json, "w") as f:
        json.dump(transforms_data, f, indent=4)
    print(f"Saved transformed point cloud to {pcl_path}")
