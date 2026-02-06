#!/usr/bin/env python
"""
Helper script to generate a 3D scene from a single perspective image.

This script:
1. Generates 3 additional views from your single input image using Stable Virtual Camera
2. Prepares the images in the format required by WorldExplorer (8 images: 000.png to 007.png)
3. Optionally runs the full 3D scene expansion

Usage:
    python single_image_to_3d.py <input_image_path> [options]
"""

import typer
import os
import shutil
import json
import numpy as np
from pathlib import Path
from PIL import Image
import subprocess
import glob
from typing import Optional

app = typer.Typer(
    name="single_image_to_3d",
    help="Generate 3D scene from a single perspective image",
    add_completion=False
)


def create_transforms_json(output_dir: str, num_images: int, image_size: tuple = (576, 576)):
    """Create a transforms.json file for the generated images."""
    # Default camera intrinsics (assuming FOV of 60 degrees)
    fov = 60.0
    focal_length = image_size[0] / (2 * np.tan(np.radians(fov / 2)))
    cx, cy = image_size[0] / 2, image_size[1] / 2
    
    frames = []
    for i in range(num_images):
        # Create identity transform for now (will be updated by SEVA)
        transform_matrix = np.eye(4).tolist()
        frames.append({
            "file_path": f"images/{i:03d}.png",
            "transform_matrix": transform_matrix
        })
    
    metadata = {
        "camera_angle_x": np.radians(fov),
        "camera_angle_y": np.radians(fov),
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": cx,
        "cy": cy,
        "w": image_size[0],
        "h": image_size[1],
        "frames": frames
    }
    
    transforms_path = os.path.join(output_dir, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return transforms_path


def create_train_test_split(output_dir: str, num_inputs: int = 1):
    """Create train_test_split.json file."""
    split_data = {
        "train": list(range(num_inputs)),
        "test": list(range(num_inputs, 8))  # Assuming 8 total images
    }
    
    split_path = os.path.join(output_dir, f"train_test_split_{num_inputs}.json")
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)
    
    return split_path


def generate_views_from_single_image(
    input_image_path: str,
    output_dir: str,
    traj_prior: str = "orbit",
    num_targets: int = 80,
    cfg: str = "4.0,2.0"
):
    """Generate multiple views from a single image using SEVA."""
    print(f"\n{'='*80}")
    print("STEP 1: Generating views from single image")
    print(f"{'='*80}\n")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy input image to output directory
    input_image = Image.open(input_image_path)
    # Resize to 576x576 if needed
    if input_image.size != (576, 576):
        print(f"Resizing image from {input_image.size} to (576, 576)")
        input_image = input_image.resize((576, 576), Image.LANCZOS)
    
    input_filename = os.path.basename(input_image_path)
    input_filename_no_ext = os.path.splitext(input_filename)[0]
    output_image_path = os.path.join(output_dir, f"{input_filename_no_ext}.png")
    input_image.save(output_image_path)
    
    print(f"Input image saved to: {output_image_path}")
    
    # Run SEVA img2trajvid_s-prob task
    print(f"\nGenerating {num_targets} frames using trajectory: {traj_prior}")
    print("This may take a few minutes...\n")
    
    command = [
        "python", "model/stable-virtual-camera/demo.py",
        "--data_path", output_dir,
        "--task", "img2trajvid_s-prob",
        "--replace_or_include_input", "True",
        "--traj_prior", traj_prior,
        "--cfg", cfg,
        "--guider", "1,2",
        "--num_targets", str(num_targets),
        "--L_short", "576",
        "--use_traj_prior", "True",
        "--chunk_strategy", "interp"
    ]
    
    print(f"Running command: {' '.join(command)}\n")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating views: {result.stderr}")
        raise typer.Exit(1)
    
    print("\n✅ View generation completed!")
    
    # Find the generated video/frames
    work_dir = "work_dirs/backwards/img2trajvid_s-prob"
    scene_name = os.path.splitext(input_filename)[0]
    generated_scene_dir = os.path.join(work_dir, scene_name)
    
    # Ensure work_dir exists
    os.makedirs(work_dir, exist_ok=True)
    
    if not os.path.exists(generated_scene_dir):
        print(f"Warning: Expected output directory not found: {generated_scene_dir}")
        print("Checking work_dirs for output...")
        # Try to find the output - look for directories that contain samples-rgb
        if os.path.exists(work_dir):
            possible_dirs = [d for d in glob.glob(os.path.join(work_dir, "*")) if os.path.isdir(d)]
            # Filter to directories that have samples-rgb subdirectory (actual output)
            valid_dirs = [d for d in possible_dirs if os.path.exists(os.path.join(d, "samples-rgb"))]
            if valid_dirs:
                # Use the most recently modified directory
                valid_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                generated_scene_dir = valid_dirs[0]
                print(f"Using: {generated_scene_dir}")
            elif possible_dirs:
                # Fallback to any directory if no samples-rgb found yet
                possible_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                generated_scene_dir = possible_dirs[0]
                print(f"Using: {generated_scene_dir} (output may still be generating)")
            else:
                print(f"Error: No output directories found in {work_dir}")
                raise typer.Exit(1)
        else:
            print(f"Error: Work directory does not exist: {work_dir}")
            raise typer.Exit(1)
    
    return generated_scene_dir


def extract_key_frames_for_scaffold(generated_scene_dir: str, output_scaffold_dir: str):
    """Extract 8 key frames from generated views to create scaffold."""
    print(f"\n{'='*80}")
    print("STEP 2: Extracting key frames for scaffold")
    print(f"{'='*80}\n")
    
    os.makedirs(output_scaffold_dir, exist_ok=True)
    images_dir = os.path.join(output_scaffold_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Look for generated frames - check both direct samples-rgb (final output) and first-pass locations
    samples_dir = os.path.join(generated_scene_dir, "samples-rgb")
    if not os.path.exists(samples_dir):
        # Fallback to first-pass location (if second pass didn't complete)
        samples_dir = os.path.join(generated_scene_dir, "first-pass", "samples-rgb")
        if not os.path.exists(samples_dir):
            print(f"Error: Samples directory not found. Checked:")
            print(f"  - {os.path.join(generated_scene_dir, 'samples-rgb')}")
            print(f"  - {os.path.join(generated_scene_dir, 'first-pass', 'samples-rgb')}")
            print(f"\nAvailable directories in {generated_scene_dir}:")
            if os.path.exists(generated_scene_dir):
                for item in os.listdir(generated_scene_dir):
                    item_path = os.path.join(generated_scene_dir, item)
                    if os.path.isdir(item_path):
                        print(f"  - {item}/")
            raise typer.Exit(1)
    
    # Get all generated frames
    frame_files = sorted(glob.glob(os.path.join(samples_dir, "*.png")))
    
    if len(frame_files) < 8:
        print(f"Warning: Only {len(frame_files)} frames found, need at least 8")
        print("Using all available frames...")
        num_frames = len(frame_files)
    else:
        num_frames = 8
    
    # Extract evenly spaced frames
    indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
    
    print(f"Extracting {num_frames} key frames from {len(frame_files)} total frames")
    
    for i, idx in enumerate(indices):
        src = frame_files[idx]
        dst = os.path.join(images_dir, f"{i:03d}.png")
        shutil.copy2(src, dst)
        print(f"  Frame {i:03d}.png <- frame {idx}")
    
    # Also copy the input image as frame 000 if available
    input_dir = os.path.join(generated_scene_dir, "input")
    if os.path.exists(input_dir):
        input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        if input_files:
            shutil.copy2(input_files[0], os.path.join(images_dir, "000.png"))
            print(f"  Frame 000.png <- original input")
    
    # Create transforms.json
    create_transforms_json(output_scaffold_dir, num_frames)
    
    # Create train_test_split_1.json (single input view)
    create_train_test_split(output_scaffold_dir, num_inputs=1)
    
    print(f"\n✅ Scaffold prepared in: {output_scaffold_dir}")
    
    return output_scaffold_dir


def prepare_final_scaffold(scaffold_dir: str, final_dir: str):
    """Prepare final scaffold directory with 8 images named 000.png to 007.png."""
    print(f"\n{'='*80}")
    print("STEP 3: Preparing final scaffold")
    print(f"{'='*80}\n")
    
    os.makedirs(final_dir, exist_ok=True)
    
    images_dir = os.path.join(scaffold_dir, "images")
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        raise typer.Exit(1)
    
    # Get all images
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    
    if len(image_files) < 8:
        print(f"Warning: Only {len(image_files)} images found, duplicating to reach 8")
        # Duplicate images to reach 8
        while len(image_files) < 8:
            image_files.extend(image_files[:8 - len(image_files)])
    
    # Copy first 8 images
    for i in range(8):
        src = image_files[i % len(image_files)]
        dst = os.path.join(final_dir, f"{i:03d}.png")
        shutil.copy2(src, dst)
        print(f"  {i:03d}.png")
    
    print(f"\n✅ Final scaffold ready in: {final_dir}")
    
    return final_dir


@app.command()
def generate(
    input_image: str = typer.Argument(..., help="Path to your single input image"),
    traj_prior: str = typer.Option("orbit", "--traj-prior", "-t", 
                                   help="Camera trajectory: orbit, spiral, pan-left, pan-right, move-forward, etc."),
    num_targets: int = typer.Option(80, "--num-targets", "-n",
                                     help="Number of frames to generate"),
    cfg: str = typer.Option("4.0,2.0", "--cfg", help="CFG scale (first_pass,second_pass)"),
    skip_expansion: bool = typer.Option(False, "--skip-expansion", help="Skip 3D scene expansion"),
    translation_scaling_factor: float = typer.Option(3.0, "--translation-scaling", "-s",
                                                      help="Translation scaling factor for scene expansion"),
    output_base: Optional[str] = typer.Option(None, "--output", "-o",
                                              help="Base output directory (default: ./single_image_scenes)"),
):
    """
    Generate a 3D scene from a single perspective image.
    
    This command will:
    1. Generate multiple views from your single image using Stable Virtual Camera
    2. Extract 8 key frames to create a scaffold
    3. Optionally expand to full 3D scene using WorldExplorer
    """
    
    # Validate input image
    if not os.path.exists(input_image):
        print(f"❌ Error: Input image not found: {input_image}")
        raise typer.Exit(1)
    
    # Setup output directories
    if output_base is None:
        output_base = "./single_image_scenes"
    
    image_name = Path(input_image).stem
    timestamp = typer.prompt("Enter a name for this scene", default=image_name)
    
    scene_output_dir = os.path.join(output_base, timestamp)
    views_dir = os.path.join(scene_output_dir, "generated_views")
    scaffold_dir = os.path.join(scene_output_dir, "scaffold")
    final_dir = os.path.join(scene_output_dir, "final")
    
    # Step 1: Generate views
    generated_scene_dir = generate_views_from_single_image(
        input_image,
        views_dir,
        traj_prior=traj_prior,
        num_targets=num_targets,
        cfg=cfg
    )
    
    # Step 2: Extract key frames
    extract_key_frames_for_scaffold(generated_scene_dir, scaffold_dir)
    
    # Step 3: Prepare final scaffold
    prepare_final_scaffold(scaffold_dir, final_dir)
    
    print(f"\n{'='*80}")
    print("✅ VIEW GENERATION COMPLETED!")
    print(f"{'='*80}\n")
    print(f"Generated views: {generated_scene_dir}")
    print(f"Scaffold: {scaffold_dir}")
    print(f"Final scaffold (8 images): {final_dir}\n")
    
    if not skip_expansion:
        print(f"\n{'='*80}")
        print("STEP 4: Expanding to 3D scene")
        print(f"{'='*80}\n")
        print("This will take 6-7 hours...\n")
        
        # Import and run scene expansion
        from model.scene_expansion import run_scene_expansion
        
        work_dir = run_scene_expansion(
            final_dir,
            translation_scaling_factor=translation_scaling_factor,
            num_images_for_vggt=40
        )
        
        print(f"\n✅ 3D scene expansion completed!")
        print(f"Results saved in: {work_dir}")
    else:
        print("\n📋 To expand this scaffold to a full 3D scene later, run:")
        print(f"   python worldexplorer.py expand '{final_dir}'")


@app.command()
def expand(
    scaffold_dir: str = typer.Argument(..., help="Path to scaffold directory with 8 images (000.png to 007.png)"),
    translation_scaling_factor: float = typer.Option(3.0, "--translation-scaling", "-t",
                                                      help="Translation scaling factor (3 for indoor, 10 for outdoor)"),
):
    """
    Expand an existing scaffold to a full 3D scene.
    
    The scaffold directory should contain 8 images named 000.png through 007.png.
    """
    # Verify scaffold directory
    if not os.path.exists(scaffold_dir):
        print(f"❌ Error: Scaffold directory not found: {scaffold_dir}")
        raise typer.Exit(1)
    
    # Check for required images
    missing_images = []
    for i in range(8):
        img_path = os.path.join(scaffold_dir, f"{i:03d}.png")
        if not os.path.exists(img_path):
            missing_images.append(f"{i:03d}.png")
    
    if missing_images:
        print(f"❌ Error: Missing required images in {scaffold_dir}:")
        for img in missing_images:
            print(f"   - {img}")
        raise typer.Exit(1)
    
    print(f"✅ Found all 8 required images in: {scaffold_dir}\n")
    
    # Run scene expansion
    from model.scene_expansion import run_scene_expansion
    
    print(f"🚀 Starting scene expansion...")
    print(f"⏱️  This process will take 6-7 hours to complete.\n")
    
    work_dir = run_scene_expansion(
        scaffold_dir,
        translation_scaling_factor=translation_scaling_factor,
        num_images_for_vggt=40
    )
    
    print(f"\n✅ Scene expansion completed!")
    print(f"Results saved in: {work_dir}")


if __name__ == "__main__":
    app()
