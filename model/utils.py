import numpy as np
from PIL import Image
import torch
from pytorch3d.transforms import RotateAxisAngle
import math
import os
import cv2

def generate_mask_old(image):
    # Convert to numpy array
    image_array = np.array(image)
    mask_array = np.all(image_array == [255, 255, 255], axis=-1) 
    mask_array = np.uint8(mask_array) * 255  
    # Convert back to an image
    mask_image = Image.fromarray(mask_array, mode="L")  # Grayscale mask (mode "L")
    return mask_image

def generate_mask(image):
    # Convert to numpy array
    image_array = np.array(image)
    mask_array = np.all(image_array == [255, 255, 255], axis=-1) 
    
    # Dilate the mask by 2 pixels
    from scipy.ndimage import binary_dilation
    dilated_mask = binary_dilation(mask_array, structure=np.ones((3, 3)), iterations=2)
    
    # Convert to 0 and 255
    dilated_mask_array = np.uint8(dilated_mask) * 255
    
    # Convert back to an image
    mask_image = Image.fromarray(dilated_mask_array, mode="L")  # Grayscale mask (mode "L")
    return mask_image

def generate_rotation_matrices(angles: list):
    # Create rotation matrices for each angle around the z-axis (camera rotation)
    rotation_matrices = []
    for angle in angles:        
        # Rotate around the z-axis
        rotation = RotateAxisAngle(angle, axis="Y")  
        rotation_matrices.append(rotation.get_matrix()[:, :3, :3])

    return torch.cat(rotation_matrices, dim=0)

def compute_focal_lengths(width, height, vertical_fov_degrees):
    """
    Computes the focal length in the x and y directions (f_x, f_y) in terms 
    of image pixel coordinates, given:
      - width (number of pixels in the image width)
      - height (number of pixels in the image height)
      - vertical_fov_degrees (vertical field of view in degrees)
    """
    # Convert vertical FOV from degrees to radians
    vertical_fov_radians = math.radians(vertical_fov_degrees)
    
    # f_y = (height/2) / tan(vertical_fov/2)
    f_y = (height / 2.0) / math.tan(vertical_fov_radians / 2.0)
    
    # Use the aspect ratio to compute the horizontal FOV
    # horizontal_fov = 2 * arctan( aspect_ratio * tan(vertical_fov/2) )
    aspect_ratio = float(width) / float(height)
    horizontal_fov_radians = 2.0 * math.atan(aspect_ratio * math.tan(vertical_fov_radians / 2.0))
    
    # f_x = (width/2) / tan(horizontal_fov/2)
    f_x = (width / 2.0) / math.tan(horizontal_fov_radians / 2.0)
    
    return f_x, f_y

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_number = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        output_filename = os.path.join(output_folder, f"frame_{frame_number:05d}.png")
        cv2.imwrite(output_filename, frame)
        frame_number += 1

    video_capture.release()
    print(f"Extraction complete. {frame_number} frames saved to {output_folder}")