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
    traj_prior: str = "pan-in-place",
    num_targets: int = 80,
    cfg: str = "4.0,2.0",
    translation_scaling_factor: float = 3.0,
    num_seva_frames: Optional[int] = 80,
):
    """Generate multiple views from a single image using SEVA.

    num_seva_frames (default 80) is passed to SEVA as num_prior_frames so the first pass
    generates that many frames (use a value divisible by 8, e.g. 40 or 80, for even keyframe spacing).
    If None, SEVA uses its default (20 frames). Collision detection can still truncate the output.
    """
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
    use_num_prior = num_seva_frames is not None and num_seva_frames > 0
    effective_targets = num_seva_frames if use_num_prior else num_targets
    print(f"\nGenerating up to {effective_targets} frames using trajectory: {traj_prior}")
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
        "--chunk_strategy", "interp",
        "--translation_scaling_factor", str(translation_scaling_factor),
    ]
    if use_num_prior:
        command.extend(["--num_prior_frames", str(num_seva_frames)])
    
    print(f"Running command: {' '.join(command)}\n")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating views: {result.stderr}")
        raise typer.Exit(1)
    
    print("\n✅ View generation completed!")
    
    # Find the generated video/frames and report actual count (may be less than num_targets)
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
    
    try:
        samples_dir = _find_seva_samples_dir(generated_scene_dir)
        n_frames = len(glob.glob(os.path.join(samples_dir, "*.png")))
        if n_frames != effective_targets:
            print(f"  SEVA produced {n_frames} frames (requested {effective_targets}). This is normal when collision detection truncates the sequence.")
    except FileNotFoundError:
        pass
    return generated_scene_dir


def _find_seva_samples_dir(generated_scene_dir: str) -> str:
    """Return path to samples-rgb (or first-pass/samples-rgb) under generated_scene_dir."""
    samples_dir = os.path.join(generated_scene_dir, "samples-rgb")
    if os.path.exists(samples_dir):
        return samples_dir
    samples_dir = os.path.join(generated_scene_dir, "first-pass", "samples-rgb")
    if os.path.exists(samples_dir):
        return samples_dir
    raise FileNotFoundError(
        f"No samples-rgb under {generated_scene_dir}. Checked samples-rgb and first-pass/samples-rgb."
    )


def copy_all_seva_frames_to(generated_scene_dir: str, dest_dir: str) -> str:
    """Copy all SEVA-generated frames to dest_dir with normalized names (000.png, 001.png, ...).
    Use this to inspect the full trajectory and verify the trajectory prior is followed.
    Returns dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)
    samples_dir = _find_seva_samples_dir(generated_scene_dir)
    frame_files = sorted(glob.glob(os.path.join(samples_dir, "*.png")))
    for i, src in enumerate(frame_files):
        shutil.copy2(src, os.path.join(dest_dir, f"{i:03d}.png"))
    print(f"  Saved {len(frame_files)} SEVA frames to {dest_dir} (for trajectory inspection)")
    return dest_dir


def extract_four_cardinal_frames_for_hybrid(
    generated_scene_dir: str,
    output_folder: str,
    keyframe_span_ratio: float = 0.875,
    use_input_as_000: bool = True,
) -> str:
    """Extract 4 cardinal views (0°, 90°, 180°, 270°) from SEVA output for hybrid scaffold.
    Writes 000.png, 002.png, 004.png, 006.png into output_folder.
    Frames are chosen evenly over the full sequence: indices 0, n//4, n//2, 3*n//4
    (e.g. for n=20 that is 0, 5, 10, 15). keyframe_span_ratio is unused and kept for API compatibility.
    """
    os.makedirs(output_folder, exist_ok=True)
    samples_dir = _find_seva_samples_dir(generated_scene_dir)
    frame_files = sorted(glob.glob(os.path.join(samples_dir, "*.png")))
    if len(frame_files) < 4:
        raise ValueError(f"Need at least 4 SEVA frames, got {len(frame_files)}")
    n = len(frame_files)
    # Even spacing over full sequence so 0°, 90°, 180°, 270° map to frames 0, n//4, n//2, 3*n//4
    indices = [0, n // 4, n // 2, (3 * n) // 4]
    cardinal_names = ["000", "002", "004", "006"]
    for name, idx in zip(cardinal_names, indices):
        src = frame_files[idx]
        dst = os.path.join(output_folder, f"{name}.png")
        shutil.copy2(src, dst)
        print(f"  {name}.png <- SEVA frame {idx}")
    if use_input_as_000:
        input_dir = os.path.join(generated_scene_dir, "input")
        if os.path.exists(input_dir):
            input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
            if input_files:
                shutil.copy2(input_files[0], os.path.join(output_folder, "000.png"))
                print("  000.png <- original input")
    return output_folder


def run_hybrid_seva_inpaint(
    input_image_path: str,
    output_dir: str,
    traj_prior: str = "pan-in-place",
    num_targets: int = 80,
    cfg: str = "4.0,2.0",
    translation_scaling_factor: float = 3.0,
    keyframe_span_ratio: float = 0.875,
    final_output_dir: Optional[str] = None,
    output_all_seva_frames_to: Optional[str] = None,
    num_seva_frames: Optional[int] = 80,
) -> str:
    """Hybrid scaffold: SEVA for 000, 002, 004, 006; inpainting for 001, 003, 005, 007.
    Returns path to final folder containing 8 images.
    If final_output_dir is set, copies the 8 images there and returns that path.
    If output_all_seva_frames_to is set, copies all SEVA-generated frames there for trajectory inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_image = Image.open(input_image_path)
    if input_image.size != (576, 576):
        input_image = input_image.resize((576, 576), Image.LANCZOS)
    input_name = Path(input_image_path).stem
    staged_path = os.path.join(output_dir, f"{input_name}.png")
    input_image.save(staged_path)

    generated_scene_dir = generate_views_from_single_image(
        staged_path,
        output_dir,
        traj_prior=traj_prior,
        num_targets=num_targets,
        cfg=cfg,
        translation_scaling_factor=translation_scaling_factor,
        num_seva_frames=num_seva_frames,
    )
    if output_all_seva_frames_to:
        copy_all_seva_frames_to(generated_scene_dir, output_all_seva_frames_to)
    hybrid_work = os.path.join(output_dir, "hybrid_work")
    extract_four_cardinal_frames_for_hybrid(
        generated_scene_dir,
        hybrid_work,
        keyframe_span_ratio=keyframe_span_ratio,
        use_input_as_000=True,
    )
    from model.scaffold_generation import GenerationMode, run_inpainting_from_four_images
    final_folder = run_inpainting_from_four_images(
        hybrid_work, generation_mode=GenerationMode.fast
    )
    if final_output_dir:
        os.makedirs(final_output_dir, exist_ok=True)
        for i in range(8):
            src = os.path.join(final_folder, f"{i:03d}.png")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(final_output_dir, f"{i:03d}.png"))
        return final_output_dir
    return final_folder


def extract_key_frames_for_scaffold(
    generated_scene_dir: str,
    output_scaffold_dir: str,
    keyframe_span_ratio: float = 0.875,
):
    """Extract 8 key frames from generated views to create scaffold.

    The 8 keyframes are evenly spaced with a constant step: step = (n-1)//7,
    indices = [0, step, 2*step, ..., 7*step]. E.g. for 24 frames: (0, 3, 6, 9, 12, 15, 18, 21).
    keyframe_span_ratio is kept for API compatibility but no longer used.
    """
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
    n = len(frame_files)
    num_frames = 8

    if n < num_frames:
        print(f"Warning: Only {n} frames found, need at least 8")
        print("Using all available frames...")
        num_frames = n
        indices = list(range(n))
    else:
        # Even spacing: step = (n-1)//7, indices = 0, step, 2*step, ..., 7*step (e.g. n=24 → 0,3,6,9,12,15,18,21).
        step = (n) // (num_frames)
        indices = [round(i * step) for i in range(num_frames)]

    print(f"Extracting {num_frames} key frames from {n} total frames (indices {indices})")
    
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
    scaffold_method: str = typer.Option("single_image", "--scaffold-method", "-m",
                                        help="single_image: 8 views from SEVA. hybrid: SEVA for 0,2,4,6 + inpainting for 1,3,5,7 (often better quality)."),
    traj_prior: str = typer.Option("pan-in-place", "--traj-prior", "-t", 
                                   help="Camera trajectory: pan-in-place (default, rotate only, no position change), orbit, spiral, move-forward, etc."),
    num_targets: int = typer.Option(80, "--num-targets", "-n",
                                     help="Number of frames to generate"),
    cfg: str = typer.Option("4.0,2.0", "--cfg", help="CFG scale (first_pass,second_pass)"),
    skip_expansion: bool = typer.Option(False, "--skip-expansion", help="Skip 3D scene expansion"),
    translation_scaling_factor: float = typer.Option(3.0, "--translation-scaling", "-s",
                                                      help="Translation scaling factor for scene expansion"),
    output_base: Optional[str] = typer.Option(None, "--output", "-o",
                                              help="Base output directory (default: ./single_image_scenes)"),
    output_all_seva_frames: bool = typer.Option(False, "--output-all-seva-frames",
                                                help="Copy all SEVA-generated frames to scene_output_dir/all_seva_frames for trajectory inspection."),
    num_seva_frames: Optional[int] = typer.Option(80, "--num-seva-frames",
                                                  help="Number of frames for SEVA to generate; use a value divisible by 8 (e.g. 40 or 80) for even keyframe spacing. Default 80."),
):
    """
    Generate a 3D scene from a single perspective image.
    
    This command will:
    1. Generate multiple views from your single image using Stable Virtual Camera
    2. Create scaffold (single_image: 8 SEVA frames; hybrid: 4 SEVA + 4 inpainted)
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
    os.makedirs(final_dir, exist_ok=True)
    
    if scaffold_method == "hybrid":
        # Hybrid: SEVA for 0,2,4,6; inpainting for 1,3,5,7
        print(f"\n{'='*80}")
        print("SCAFFOLD: Hybrid (SEVA + inpainting)")
        print(f"{'='*80}\n")
        all_seva_dir = os.path.join(scene_output_dir, "all_seva_frames") if output_all_seva_frames else None
        final_dir = run_hybrid_seva_inpaint(
            input_image,
            views_dir,
            traj_prior=traj_prior,
            num_targets=num_targets,
            cfg=cfg,
            translation_scaling_factor=translation_scaling_factor,
            final_output_dir=final_dir,
            output_all_seva_frames_to=all_seva_dir,
            num_seva_frames=num_seva_frames,
        )
        print(f"\n{'='*80}")
        print("✅ HYBRID SCAFFOLD COMPLETED!")
        print(f"{'='*80}\n")
        print(f"Final scaffold (8 images): {final_dir}\n")
    else:
        # Default: 8 key frames from SEVA
        generated_scene_dir = generate_views_from_single_image(
            input_image,
            views_dir,
            traj_prior=traj_prior,
            num_targets=num_targets,
            cfg=cfg,
            translation_scaling_factor=translation_scaling_factor,
            num_seva_frames=num_seva_frames,
        )
        if output_all_seva_frames:
            copy_all_seva_frames_to(generated_scene_dir, os.path.join(scene_output_dir, "all_seva_frames"))
        extract_key_frames_for_scaffold(generated_scene_dir, scaffold_dir)
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
