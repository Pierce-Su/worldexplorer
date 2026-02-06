# Scene expansion module for converting 2D panorama to 3D scene

# Default configuration - can be overridden when importing
ROOT_DIR = "./predefined_trajectories"  # Directory containing predefined conservative trajectories for clockwise generation
NERF_FOLDER = "./nerfstudio_output"  # Directory for NeRF training outputs
TRAJECTORY_ORDER = ["in", "left", "right", "up"]  # Order in which to process trajectory types

# Default input folders for standalone execution (updated to new structure)
INPUT_FOLDERS = [
    ("panoramas/hongkong_at_night/20250827_155600/final", 3),
]

import datetime
import os
import json
import shutil
import subprocess
import numpy as np
from PIL import Image
import gc
import torch

from .pred_and_align_with_transforms import run_vggt_and_align
from .sample_images import select_images_for_vggt
from .rotate_ply import rotate_ply_3dgs


class TrajectoryTransformer:
    def __init__(self, root_dir, work_dir, input_folder=None, trajectory_order=None, translation_scaling_factor=3, scene_id=None, num_images_for_vggt=40):
        """
        Initialize the trajectory transformer.

        Args:
            root_dir: Directory containing the original trajectory folders
            work_dir: Directory where new trajectories will be created
            input_folder: Optional folder containing input images to use
            trajectory_order: Optional list specifying the order to process trajectory types
            translation_scaling_factor: Translation scaling factor for scene expansion
            scene_id: Unique identifier for the scene, used in work_dir naming
            num_images_for_vggt: Number of images to sample for VGGT processing (default: 40)
        """
        self.root_dir = root_dir
        self.work_dir = work_dir
        self.input_folder = input_folder
        self.initial_images = ['000', '001', '002', '003', '004', '005', '006', '007']
        self.trajectory_order = trajectory_order  # User-defined order of trajectories
        self.translation_scaling_factor = translation_scaling_factor
        self.scene_id = scene_id  # Unique identifier for the scene, used in work_dir naming
        self.num_images_for_vggt = num_images_for_vggt
        
        # Ensure work directory exists
        os.makedirs(work_dir, exist_ok=True)

    def replace_floor_images(self, original_folder_path, input_folder_path):
        """
        Creates a new folder with the contents of the original folder, but with
        PNG images 000.png to 007.png replaced by those from the input folder.
        
        Args:
            original_folder_path (str): Path to the original folder containing PNGs and other files
            input_folder_path (str): Path to the folder containing the replacement PNG images
            
        Returns:
            str: Path to the new folder
        """
        # Extract the original folder name
        org_folder_name = os.path.basename(os.path.normpath(original_folder_path))
        
        # Construct the destination path
        dest_parent_path = os.path.join(self.work_dir, "floor_wrapper")
        dest_folder_path = os.path.join(dest_parent_path, org_folder_name)
        
        # Create the parent directory if it doesn't exist
        os.makedirs(dest_parent_path, exist_ok=True)
        
        # If the destination folder already exists, remove it
        if os.path.exists(dest_folder_path):
            shutil.rmtree(dest_folder_path)
        
        # Copy all contents from the original folder to the destination
        shutil.copytree(original_folder_path, dest_folder_path)
        
        # Replace the specific numbered PNG files with those from the input folder
        for i in range(8):  # 000.png to 007.png
            png_filename = f"{i:03d}.png"  # Format: 000.png, 001.png, etc.
            source_png = os.path.join(input_folder_path, png_filename)
            dest_png = os.path.join(dest_folder_path, png_filename)
            
            # Check if the source PNG exists before copying
            if os.path.exists(source_png):
                shutil.copy2(source_png, dest_png)
        
        return dest_folder_path
        
    def find_trajectory_folders(self):
        """Find all trajectory folders in the root directory."""
        trajectory_folders = {}
        
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                # The folder name is the trajectory type (e.g., "in", "left")
                traj_type = folder_name
                trajectory_folders[traj_type] = folder_path
                
        return trajectory_folders
    
    def load_transforms(self, trajectory_folder):
        """Load the transforms.json file from a trajectory folder."""
        # Check if transforms.json is directly in the trajectory folder
        transforms_path = os.path.join(trajectory_folder, 'transforms.json')
        if os.path.exists(transforms_path):
            with open(transforms_path, 'r') as f:
                return json.load(f)
        else:
            print(f"transforms.json not found at {transforms_path}")
            return None
    
    def compute_transformation_matrix(self, source_matrix, target_matrix):
        """
        Compute the transformation matrix from source to target.
        
        Args:
            source_matrix: The 4x4 transform matrix of the source image
            target_matrix: The 4x4 transform matrix of the target image
            
        Returns:
            The 4x4 transformation matrix from source to target
        """
        # Convert to numpy arrays for matrix operations
        source = np.array(source_matrix)
        target = np.array(target_matrix)
        
        # Compute the transformation matrix
        # To go from source to target, we apply: target = T * source
        # So T = target * source^-1
        source_inv = np.linalg.inv(source)
        transformation = np.matmul(target, source_inv)
        
        return transformation.tolist()
    
    def apply_transformation(self, frame_matrix, transformation_matrix):
        """
        Apply a transformation matrix to a frame's transform matrix.
        
        Args:
            frame_matrix: The 4x4 transform matrix of the frame
            transformation_matrix: The 4x4 transformation matrix to apply
            
        Returns:
            The transformed 4x4 matrix
        """
        frame = np.array(frame_matrix)
        transformation = np.array(transformation_matrix)
        
        # Apply the transformation
        transformed_matrix = np.matmul(transformation, frame)
        
        return transformed_matrix.tolist()
    
    def create_trajectory_folder(self, initial_img, traj_type, original_transforms, transformations):
        """
        Create a new trajectory folder with transformed matrices.
        
        Args:
            initial_img: The initial image ID (e.g., '000')
            traj_type: The trajectory type (e.g., 'in')
            original_transforms: The original transforms.json data
            transformations: Dictionary mapping initial image IDs to transformation matrices
            
        Returns:
            Path to the created trajectory wrapper folder
        """
        # Create the nested folder structure:
        # {work_dir}/{initial_image}_{trajectory_name}_wrapper/{initial_image}_{trajectory_name}/
        traj_name = f"{initial_img}_{traj_type}"
        wrapper_folder = os.path.join(self.work_dir, f"{traj_name}_wrapper")
        traj_folder = os.path.join(wrapper_folder, traj_name)
        
        # Create the folders
        os.makedirs(traj_folder, exist_ok=True)
        
        # Create a new transforms.json with transformed matrices
        new_transforms = {
            "orientation_override": original_transforms.get("orientation_override", "none"),
            "frames": []
        }
        
        # Get the transformation matrix for this initial image
        transformation_matrix = transformations.get(initial_img)
        
        # Create transformed frames
        for i, frame in enumerate(original_transforms["frames"]):
            new_frame = frame.copy()
            
            # Extract frame ID from file path to check if it's > 007
            frame_file = os.path.basename(frame["file_path"])
            frame_id = os.path.splitext(frame_file)[0]
            
            # Only apply transformation if:
            # 1. This is not the 008 reference image set, AND
            # 2. The current frame has ID > 007 (e.g., 008, 009, etc.)
            if initial_img != '000' and frame_id > '007':
                new_frame["transform_matrix"] = self.apply_transformation(
                    frame["transform_matrix"], transformation_matrix
                )
            
            # Update file path
            orig_file = os.path.basename(frame["file_path"])
            new_frame["file_path"] = f"./{orig_file}"
            
            new_transforms["frames"].append(new_frame)
        
        # Write the new transforms.json to the inner trajectory folder
        with open(os.path.join(traj_folder, 'transforms.json'), 'w') as f:
            json.dump(new_transforms, f, indent=5)
        
        # Copy or create the image files in the inner trajectory folder
        for i, frame in enumerate(original_transforms["frames"]):
            orig_file = os.path.basename(frame["file_path"])
            frame_id = os.path.splitext(orig_file)[0]
            
            dst_path = os.path.join(traj_folder, orig_file)
            
            # For initial images (000-007), copy from input folder if available, otherwise use original logic
            if i < 8:
                if self.input_folder and frame_id in self.initial_images:
                    # Use image from the specified input folder
                    src_path = os.path.join(self.input_folder, f"{frame_id}.png")
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)
                        continue
                
                # Fall back to original logic if input folder image is not available
                # Check various potential locations for initial images
                potential_paths = [
                    os.path.join(self.root_dir, f"{frame_id}.png"),  # Root directory
                    os.path.join(self.root_dir, traj_type, f"{frame_id}.png"),  # In trajectory folder
                    os.path.join(os.path.dirname(self.root_dir), f"{frame_id}.png"),  # Parent of root dir
                    # Add more potential paths if needed
                ]
                
                src_path = None
                for path in potential_paths:
                    if os.path.exists(path):
                        src_path = path
                        break
                
                if src_path:
                    print(f"Found initial image at {src_path}")
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"Warning: Could not find initial image {frame_id}.png, creating black placeholder")
                    # Create a black image as fallback
                    black_img = Image.new('RGB', (576, 576), color='black')
                    black_img.save(dst_path)
            else:
                # For trajectory frames, create black placeholders
                black_img = Image.new('RGB', (576, 576), color='black')
                black_img.save(dst_path)
        
        # Create train_test_split_8.json in the inner trajectory folder
        train_test_split = {
            "train_ids": list(range(8)),
            "test_ids": list(range(8, len(original_transforms["frames"])))
        }
        
        train_test_path = os.path.join(traj_folder, 'train_test_split_8.json')
        with open(train_test_path, 'w') as f:
            json.dump(train_test_split, f, indent=4)
        
        # Return the wrapper folder path to be passed to model/stable-virtual-camera/demo.py
        return wrapper_folder
    
    def run_command(self, command):
        """Run a command and stream its output."""
        print(f"Running command: {command}")
        process = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return process.poll()
    
    def run(self):
        """Run the trajectory transformation process."""
        # Create a global samples file for this run -> will keep track of all previously generated outputs
        global_samples_file = os.path.join(self.work_dir, "global_samples.pt")

        # Find all trajectory folders
        trajectory_folders = self.find_trajectory_folders()
        print(f"Found {len(trajectory_folders)} trajectory folders: {list(trajectory_folders.keys())}")
        
        # Determine the order of trajectory types to process
        if self.trajectory_order:
            # Use the user-defined order
            traj_types_to_process = [t for t in self.trajectory_order if t in trajectory_folders]
        else:
            # Use the default order (alphabetical from dictionary)
            traj_types_to_process = list(trajectory_folders.keys())
        
        print(f"Processing trajectories in order: {traj_types_to_process}")
        
        # First process the 0 image (reference) for each trajectory type
        for traj_type in traj_types_to_process:  
            folder_path = trajectory_folders[traj_type]
            print(f"\n--- Processing trajectory type: {traj_type} for image 000 (reference) ---\n")
            
            # Load the original transforms
            original_transforms = self.load_transforms(folder_path)
            if not original_transforms:
                print(f"Skipping trajectory {traj_type} due to missing transforms")
                continue
            
            # Extract transform matrices for the initial images
            initial_matrices = {}
            for frame in original_transforms["frames"][:8]:
                frame_id = os.path.splitext(os.path.basename(frame["file_path"]))[0]
                initial_matrices[frame_id] = frame["transform_matrix"]
            
            # For 000, the transformation is the identity matrix
            transformations = {'000': np.eye(4).tolist()}
            
            # Create the 000 trajectory folder and run the command
            print(f"\n--- Creating trajectory folder for 000_{traj_type} ---\n")
            
            # Create the trajectory folder structure and return the wrapper folder path
            wrapper_folder = self.create_trajectory_folder(
                '000', traj_type, original_transforms, transformations
            )
            
            # Check if this trajectory already exists
            expected_output_dir = os.path.join(self.work_dir, "img2trajvid", f"000_{traj_type}")
            if os.path.exists(expected_output_dir) and os.path.exists(os.path.join(expected_output_dir, "transforms.json")):
                print(f"Skipping 000_{traj_type} - already generated (found {expected_output_dir})")
                continue
            
            # Run the command with the wrapper folder
            command = (
                f"python model/stable-virtual-camera/demo.py --data_path {wrapper_folder} --task img2trajvid "
                f"--cfg 3.0,2.3 --use_traj_prior True --L_short 576 --chunk_strategy interp "
                f"--num_inputs 8 --work_dir {self.work_dir} "
                f"--global_samples_file {global_samples_file} "  # Add the global samples file
                f"--translation_scaling_factor {self.translation_scaling_factor}"
            )
            return_code = self.run_command(command)
            
            if return_code == 0:
                print(f"Successfully generated trajectory 000_{traj_type}")
            else:
                print(f"Failed to generate trajectory 000_{traj_type}")
        
        # Now process the other initial images (001-007) for each trajectory type
        transformations = {}
        for traj_type in traj_types_to_process:
            folder_path = trajectory_folders[traj_type]
            print(f"\n--- Processing trajectory type: {traj_type} for other images ---\n")
            
            # Load the original transforms
            original_transforms = self.load_transforms(folder_path)
            if not original_transforms:
                print(f"Skipping trajectory {traj_type} due to missing transforms")
                continue
            
            # Extract transform matrices for the initial images
            initial_matrices = {}
            for frame in original_transforms["frames"][:8]:
                frame_id = os.path.splitext(os.path.basename(frame["file_path"]))[0]
                initial_matrices[frame_id] = frame["transform_matrix"]
            
            # Get the reference matrix for 000
            reference_matrix = initial_matrices.get('000')
            if not reference_matrix:
                print(f"Reference matrix for image 000 not found in {traj_type}")
                continue
            
            # Compute transformations from 000 to each other initial image
            transformations = {}
            for img_id in self.initial_images:
                if img_id == '000':
                    # Already processed 000 above
                    continue
                
                target_matrix = initial_matrices.get(img_id)
                if target_matrix:
                    transformations[img_id] = self.compute_transformation_matrix(
                        reference_matrix, target_matrix
                    )
                else:
                    print(f"Transform matrix for image {img_id} not found in {traj_type}")
            
        # Create a trajectory folder for each initial image and run the command
        for img_id in self.initial_images:
            for traj_type in traj_types_to_process:
                if img_id == '000' or img_id not in transformations:
                    # Skip 000 (already processed) and any images without transformations
                    continue
                
                print(f"\n--- Creating trajectory folder for {img_id}_{traj_type} ---\n")
                # Load the original transforms
                folder_path = trajectory_folders[traj_type]
                original_transforms = self.load_transforms(folder_path)
                if not original_transforms:
                    print(f"Skipping trajectory {traj_type} due to missing transforms")
                    continue
                
                # Check if this trajectory already exists
                expected_output_dir = os.path.join(self.work_dir, "img2trajvid", f"{img_id}_{traj_type}")
                if os.path.exists(expected_output_dir) and os.path.exists(os.path.join(expected_output_dir, "transforms.json")):
                    print(f"Skipping {img_id}_{traj_type} - already generated (found {expected_output_dir})")
                    continue
                
                # Create the trajectory folder structure and return the wrapper folder path
                wrapper_folder = self.create_trajectory_folder(
                    img_id, traj_type, original_transforms, transformations
                )
                
                # Run the command with the wrapper folder
                command = (
                    f"python model/stable-virtual-camera/demo.py --data_path {wrapper_folder} --task img2trajvid "
                    f"--cfg 3.0,2.3 --use_traj_prior True --L_short 576 --chunk_strategy interp "
                    f"--num_inputs 8 --work_dir {self.work_dir} "
                    f"--global_samples_file {global_samples_file} "  # Add the global samples file
                    f"--translation_scaling_factor {self.translation_scaling_factor}"
                )
                return_code = self.run_command(command)
                
                if return_code == 0:
                    print(f"Successfully generated trajectory {img_id}_{traj_type}")
                else:
                    print(f"Failed to generate trajectory {img_id}_{traj_type}")

        # Check if transform conversion already done
        merged_transforms = os.path.join(self.work_dir, "img2trajvid", "transforms.json")
        if os.path.exists(merged_transforms):
            print(f"Transform conversion already done (found {merged_transforms}), skipping...")
        else:
            command = (
                f"python model/svc_to_nerf_transform_sparse.py -r {self.work_dir}/img2trajvid "
            )
            return_code = self.run_command(command)
            if return_code == 0:
                print(f"Successfully created nerf transform")
            else:
                print(f"Failed to create nerf transform")
        
        # Check if VGGT already done
        vggt_pcl = os.path.join(self.work_dir, "img2trajvid", "vggt_pcl.ply")
        if os.path.exists(vggt_pcl):
            print(f"VGGT processing already done (found {vggt_pcl}), skipping...")
        else:
            select_images_for_vggt(f"{self.work_dir}/img2trajvid", max_images=self.num_images_for_vggt)
            # this updates the transforms.json files in place
            run_vggt_and_align(f"{self.work_dir}/img2trajvid")

        # -------------------------------------------------------------------------
        # 3. Train NeRF using the estimated poses
        # -------------------------------------------------------------------------
        # Check if training already completed (has checkpoint)
        scene_root = os.path.join(NERF_FOLDER, f"{self.scene_id if self.scene_id else ''}", "splatfacto")
        training_done = False
        if os.path.exists(scene_root):
            runs = sorted([d for d in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, d))])
            if runs:
                latest_run = runs[-1]
                checkpoint_dir = os.path.join(scene_root, latest_run, "nerfstudio_models")
                if os.path.exists(checkpoint_dir) and len([f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]) > 0:
                    print(f"Training already completed (found checkpoints in {checkpoint_dir}), skipping training...")
                    training_done = True
        
        if not training_done:
            print("running splatfaco training")
            command = [
                "ns-train", "splatfacto-big",
                "--data", f"{self.work_dir}/img2trajvid",
                "--output-dir", NERF_FOLDER,
                "--experiment-name", f"{self.scene_id if self.scene_id else ''}",
                "--vis", "tensorboard",
                "--steps_per_eval_image", "5000",
                "--steps_per_eval_all_images", "1000000",
                "--pipeline.model.camera-optimizer.mode", "off",
                "--pipeline.model.strategy", "mcmc",
                "--pipeline.model.rasterize_mode", "antialiased",
                "--pipeline.model.use_scale_regularization", "True",
                "--pipeline.model.max_gauss_ratio", "5.0",
                "--pipeline.model.cull_scale_thresh", "0.1",
                "--pipeline.model.cull_screen_size", "0.1",
                "nerfstudio-data",
                "--eval_mode", "all",
            ]
            result = subprocess.run(command)
            if result.returncode != 0:
                print(f"\n⚠️  Training failed (exit code {result.returncode})")
                print("This is likely due to gsplat CUDA compilation issues.")
                print("You can:")
                print("  1. Fix CUDA toolkit installation and retry")
                print("  2. Manually run training later: ns-train splatfacto-big --data <path> --load-config <config>")
                print("  3. Export will be skipped until training completes successfully")
                # Don't raise error, just continue to export step (which will also fail but gracefully)
        gc.collect()
        torch.cuda.empty_cache()
        
        print("\n--- Trajectory transformation process completed ---\n")
        
        # Export the trained splatfacto to .ply file
        print("\n--- Exporting splatfacto to .ply file ---\n")
        
        # Find the latest splatfacto config file
        scene_root = os.path.join(NERF_FOLDER, f"{self.scene_id if self.scene_id else ''}", "splatfacto")
        
        # Get the most recent run directory (they're timestamped)
        if os.path.exists(scene_root):
            runs = sorted([d for d in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, d))])
            if runs:
                latest_run = runs[-1]  # Get the most recent run
                config_path = os.path.join(scene_root, latest_run, "config.yml")
                
                if os.path.exists(config_path):
                    output_dir = os.path.join(scene_root, latest_run, "exports", "splat")
                    
                    # Check if export already exists
                    if os.path.exists(os.path.join(output_dir, "splat.ply")):
                        print(f"Export already exists at {output_dir}/splat.ply, skipping...")
                    else:
                        print(f"Exporting splatfacto from config: {config_path}")
                        print(f"Output directory: {output_dir}")
                        
                        
                        # TODO(mschneider): revert once https://github.com/nerfstudio-project/nerfstudio/issues/3683 is fixed / https://github.com/nerfstudio-project/nerfstudio/pull/3711 is merged
                        # export_command = [
                        #     "ns-export", "gaussian-splat",
                        #     "--load-config", config_path,
                        #     "--output-dir", output_dir,
                        # ]
                        export_command = [
                            "python", "-c",
                            f"import sys, torch, os; "
                            f"os.environ['NERFSTUDIO_DISABLE_TORCH_COMPILE'] = '1'; "
                            f"sys.argv = ['ns-export', 'gaussian-splat', '--load-config', '{config_path}', '--output-dir', '{output_dir}']; "
                            f"orig = torch.load; "
                            f"setattr(torch, 'load', lambda *a, **k: orig(*a, **{{**k, 'weights_only': False}})); "
                            f"from nerfstudio.scripts.exporter import entrypoint; "
                            f"entrypoint()"
                        ]
                        
                        # Run the export
                        export_result = subprocess.run(export_command)
                        
                        if export_result.returncode == 0:
                            print(f"Successfully exported splatfacto to {output_dir}/splat.ply")
                            # Rotate the ply file, s.t. standard viewers like SuperSplat use the correct coordinate system
                            if os.path.exists(f"{output_dir}/splat.ply"):
                                rotate_ply_3dgs(f"{output_dir}/splat.ply", f"{output_dir}/splat_rotated.ply")
                            else:
                                print(f"Warning: splat.ply not found at {output_dir}/splat.ply, skipping rotation")
                        else:
                            print(f"Failed to export splatfacto (return code: {export_result.returncode})")
                            print("This is likely due to training failure. Check the training logs above.")
                            print("Common causes:")
                            print("  1. CUDA toolkit not found (gsplat needs nvcc to compile)")
                            print("  2. gsplat CUDA extensions not compiled properly")
                            print("\nTo fix:")
                            print("  1. Ensure CUDA toolkit is installed and in PATH")
                            print("  2. Set CUDA_HOME environment variable")
                            print("  3. Reinstall gsplat: pip install --force-reinstall --no-cache-dir gsplat")
                            print("\nSkipping PLY rotation due to export failure.")

                        # Clear GPU memory after export
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    print(f"Config file not found at {config_path}, cannot export")
            else:
                print(f"No run directories found in {scene_root}, cannot export")
        else:
            print(f"Scene root directory not found at {scene_root}, cannot export")



def run_scene_expansion(input_folder, translation_scaling_factor=3, root_dir=None, trajectory_order=None, num_images_for_vggt=40):
    """
    Run scene expansion on a folder of images.

    Args:
        input_folder: Path to folder containing 8 images (000.png to 007.png)
        translation_scaling_factor: Translation scaling factor for scene expansion
        root_dir: Directory containing original trajectories (uses default if None)
        trajectory_order: Order to process trajectories (uses default if None)
        num_images_for_vggt: Number of images to sample for VGGT processing (default: 40)

    Returns:
        scene_work_dir: Path to the work directory with results
    """
    # Use defaults if not provided
    if root_dir is None:
        root_dir = ROOT_DIR
    if trajectory_order is None:
        trajectory_order = TRAJECTORY_ORDER
        
    # Extract the scene name (last subfolder of input path)
    scene_name = os.path.basename(os.path.normpath(input_folder))
    if scene_name == "final":
        # Go up two directory levels and get that folder name
        parent_parent = os.path.dirname(os.path.dirname(os.path.normpath(input_folder)))
        scene_name = os.path.basename(parent_parent)
    
    # Create a custom work directory for this scene
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_id = f"{scene_name}_{translation_scaling_factor}_{timestamp}"
    scene_work_dir = f"./scenes/{scene_id}"
    
    print(f"\n=== Processing scene: {scene_name} ===\n")
    print(f"Input folder: {input_folder}")
    print(f"Work directory: {scene_work_dir}")
    print(f"Translation scaling factor: {translation_scaling_factor}")
    
    # Create and run the transformer for this scene
    transformer = TrajectoryTransformer(
        root_dir,
        scene_work_dir,
        input_folder=input_folder,
        trajectory_order=trajectory_order,
        translation_scaling_factor=translation_scaling_factor,
        scene_id=scene_id,
        num_images_for_vggt=num_images_for_vggt
    )
    transformer.run()
    
    return scene_work_dir


if __name__ == "__main__":
    # For standalone execution, process from INPUT_FOLDERS
    for input_folder, translation_scaling_factor in INPUT_FOLDERS:
        run_scene_expansion(input_folder, translation_scaling_factor)