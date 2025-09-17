import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch

from .Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

# Generate depth maps and point clouds from images.
def pcd_from_image(img_path, focal_length_x, focal_length_y, max_depth=20, outdir='./vis_pointcloud', ):
    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': max_depth})
    checkpoint_path = os.path.join(os.path.dirname(__file__), "Depth_Anything_V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth")
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Get the list of image files to process
    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)

    # Create the output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Process each image file
    for k, filename in enumerate(filenames):
        print(f'Processing {k+1}/{len(filenames)}: {filename}')

        # Load the image
        color_image = Image.open(filename).convert('RGB')
        width, height = color_image.size

        # Read the image using OpenCV
        image = cv2.imread(filename)
        pred = depth_anything.infer_image(image, height)

        # Resize depth prediction to match the original image size
        resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

        # Generate mesh grid and calculate point cloud coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / focal_length_x
        y = (y - height / 2) / focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0

        # Create the point cloud and save it to the output directory
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

def torch_to_o3d_pcd(pcd):
    vertices = pcd.points_list()[0]
    colors = pcd.features_list()[0]
    # need to invert the y-axis so that images are not upside down
    vertices[:, 1] = -vertices[:, 1]
    # need to invert the x-axis so that images are not flipped horizontally
    vertices[:, 0] = -vertices[:, 0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

    return pcd

def o3d_pcd_to_torch(pcd, p=None, c=None, device='cuda'):
    points = torch.from_numpy(np.asarray(pcd.points)).to(device, dtype=torch.float32)
    if p is not None:
        points = points.to(p)
    colors = torch.from_numpy(np.asarray(pcd.colors)).to(device, dtype=torch.float32)
    if c is not None:
        colors = colors.to(c)
    # need to invert the y-axis so that images are not upside down
    points[:, 1] = -points[:, 1]
    # need to invert the x-axis so that images are not flipped horizontally
    points[:, 0] = -points[:, 0]
    return points, colors