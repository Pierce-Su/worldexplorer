#!/usr/bin/env python
import typer
from typing import Optional
from pathlib import Path
import datetime
import os
import subprocess

# Import the refactored functions from existing modules
from model.scaffold_generation import GenerationMode, run_scaffold_generation
from model.scene_expansion import run_scene_expansion

app = typer.Typer(
    name="WorldExplorer",
    help="🌍 WorldExplorer - Towards Fully-Navigable 3D Scenes",
    add_completion=False
)


def check_and_download_checkpoints():
    """Check for required checkpoints and download them if missing."""

    # Check for Video-Depth-Anything checkpoint
    vda_checkpoint_path = Path("model/Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth")
    if not vda_checkpoint_path.exists():
        print("\n📥 Video-Depth-Anything checkpoint not found. Downloading...")

        # Download the checkpoint file
        download_url = "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true"
        download_cmd = f'wget -O {vda_checkpoint_path} "{download_url}"'

        try:
            result = subprocess.run(download_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Successfully downloaded Video-Depth-Anything checkpoint to {vda_checkpoint_path}")
            else:
                print(f"❌ Error downloading Video-Depth-Anything checkpoint: {result.stderr}")
                print("\nPlease download manually from:")
                print(f"  {download_url}")
                print(f"And save to: {vda_checkpoint_path}")
                raise typer.Exit(1)
        except Exception as e:
            print(f"❌ Error downloading Video-Depth-Anything checkpoint: {e}")
            print("\nPlease download manually from:")
            print(f"  {download_url}")
            print(f"And save to: {vda_checkpoint_path}")
            raise typer.Exit(1)
    else:
        print(f"✅ Video-Depth-Anything checkpoint found: {vda_checkpoint_path}")

    # Check for Depth_Anything_V2 checkpoint
    da2_checkpoint_path = Path("model/Depth_Anything_V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth")
    if not da2_checkpoint_path.exists():
        print("\n📥 Depth_Anything_V2 checkpoint not found. Downloading...")

        # Download the checkpoint file
        download_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true"
        download_cmd = f'wget -O {da2_checkpoint_path} "{download_url}"'

        try:
            result = subprocess.run(download_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Successfully downloaded Depth_Anything_V2 checkpoint to {da2_checkpoint_path}")
            else:
                print(f"❌ Error downloading Depth_Anything_V2 checkpoint: {result.stderr}")
                print("\nPlease download manually from:")
                print(f"  {download_url}")
                print(f"And save to: {da2_checkpoint_path}")
                raise typer.Exit(1)
        except Exception as e:
            print(f"❌ Error downloading Depth_Anything_V2 checkpoint: {e}")
            print("\nPlease download manually from:")
            print(f"  {download_url}")
            print(f"And save to: {da2_checkpoint_path}")
            raise typer.Exit(1)
    else:
        print(f"✅ Depth_Anything_V2 checkpoint found: {da2_checkpoint_path}")


@app.command()
def generate(
    theme: Optional[str] = typer.Argument(None, help="The theme for generation (e.g., 'Rustic Farmhouse (Wood, Leather, Wool)'). Not used with --custom"),
    mode: GenerationMode = typer.Option(GenerationMode.manual, "--mode", "-m", help="Generation mode: fast (single output), automatic (CLIP-based selection), or manual (user selection)"),
    translation_scaling_factor: float = typer.Option(3.0, "--translation-scaling", "-t", help="Translation scaling factor for scene expansion"),
    skip_expansion: bool = typer.Option(False, "--skip-expansion", help="Skip scene expansion step"),
    root_dir: Optional[str] = typer.Option(None, "--root-dir", help="Directory containing original trajectories (uses default if not specified)"),
    custom: bool = typer.Option(False, "--custom", "-c", help="Enable custom mode for outdoor/custom scene generation with user-provided prompts"),
    num_images_for_vggt: int = typer.Option(40, "--num-images-for-vggt", help="Number of images to be sampled from the global scene memory as input to VGGT in addition to the scaffold images. The higher the number of images, the better the initial pointcloud used for gaussian-splatting initialization. We recommend to set the parameter according to the available GPU memory.")
):
    """Generate scaffold and optionally expand to 3D scene."""

    # Check and download checkpoints if needed
    check_and_download_checkpoints()

    print(f"\n{'='*80}")
    print("SCAFFOLD GENERATION")
    print(f"{'='*80}")
    
    if custom:
        # Custom mode - force manual mode and query user for prompts
        if mode != GenerationMode.manual:
            print("\n⚠️  Note: Custom mode requires manual selection for best results.")
            print("   Switching to manual mode...")
            mode = GenerationMode.manual
        
        print("\n🎨 Custom Scene Generation Mode")
        print("Please provide 4 prompts for generating a panoramic scene.")
        print("Each prompt will generate one of the 4 cardinal directions.")
        print("Note: The eye-level camera angle will be automatically added.\n")
        
        custom_prompts = []
        for i in range(4):
            direction = ["North", "West", "South", "East"][i]
            prompt = typer.prompt(f"Prompt for {direction} view")
            custom_prompts.append(prompt)
        
        # Query for translation scaling factor if not skipping expansion
        if not skip_expansion:
            print("\n📏 Translation Scaling Factor")
            print("This controls the scale of movement in the 3D scene:")
            print("  • Indoor scenes: typically 3")
            print("  • Outdoor scenes: typically 10")
            translation_scaling_factor = typer.prompt(
                "Enter translation scaling factor",
                type=float,
                default=10.0,
                show_default=True
            )
        
        print(f"\nMode: manual (required for custom scenes)")
        
        # Run scaffold generation with custom prompts
        parent_folder, output_folder, final_folder = run_scaffold_generation(
            theme="custom",
            mode=mode,
            custom=True,
            custom_prompts=custom_prompts
        )
    else:
        # Standard mode
        if theme is None:
            print("❌ Error: Theme is required when not using --custom mode")
            raise typer.Exit(1)
        
        print(f"Theme: {theme}")
        print(f"Mode: {mode.value}")
        
        # Run scaffold generation
        parent_folder, output_folder, final_folder = run_scaffold_generation(theme, mode)
    
    print(f"\n✅ Scaffold generation completed!")
    print(f"Images saved in: {final_folder}")
    
    # For fast and automatic modes, continue with scene expansion
    if mode in [GenerationMode.fast, GenerationMode.automatic] and not skip_expansion:
        print(f"\n{'='*80}")
        print("SCENE EXPANSION")
        print(f"{'='*80}")
        print("\n🚀 Starting scene expansion...")
        print("⏱️  This process will take several hours to complete.")
        print("   The system will continue running until finished.\n")
        
        work_dir = run_scene_expansion(
            final_folder,
            translation_scaling_factor,
            root_dir=root_dir,
            num_images_for_vggt=num_images_for_vggt
        )
        print(f"\n✅ Scene expansion completed!")
        print(f"Results saved in: {work_dir}")
    
    elif mode == GenerationMode.manual and not skip_expansion:
        print(f"\n{'='*80}")
        print("MANUAL MODE - Scene Expansion Paused")
        print(f"{'='*80}")
        print("\n⏸️  Manual selection required before continuing to 3D expansion.")
        print(f"\n📋 Next Steps:")
        print(f"1. Review the generated variations in: {output_folder}")
        print(f"2. Copy your selected images (001, 003, 005, 007) to: {final_folder}")
        print(f"3. Run scene expansion with:")
        print(f"   python worldexplorer.py expand '{final_folder}'")


@app.command()
def expand(
    input_folder: str = typer.Argument(..., help="Path to folder containing 8 images (000.png to 007.png)"),
    translation_scaling_factor: Optional[float] = typer.Option(None, "--translation-scaling", "-t", help="Translation scaling factor for scene expansion"),
    root_dir: Optional[str] = typer.Option(None, "--root-dir", help="Directory containing original trajectories (uses default if not specified)"),
    num_images_for_vggt: int = typer.Option(40, "--num-images-for-vggt", help="Number of images to be sampled from the global scene memory as input to VGGT in addition to the scaffold images. The higher the number of images, the better the initial pointcloud used for gaussian-splatting initialization. We recommend to set the parameter according to the available GPU memory.")
):
    """Expand an existing scaffold to 3D scene."""
    
    # Verify input folder exists
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ Error: Input folder '{input_folder}' does not exist!")
        raise typer.Exit(1)
    
    # Check if all required images exist
    missing_images = []
    for i in range(8):
        img_path = input_path / f"00{i}.png"
        if not img_path.exists():
            missing_images.append(f"00{i}.png")
    
    if missing_images:
        print(f"❌ Error: Missing required images in {input_folder}:")
        for img in missing_images:
            print(f"   - {img}")
        print("\nPlease ensure all 8 images (000.png to 007.png) are present.")
        raise typer.Exit(1)
    
    print(f"\n✅ Found all 8 required images in: {input_folder}")

    # Check and download checkpoints if needed
    check_and_download_checkpoints()

    # If translation scaling factor not provided, prompt user
    if translation_scaling_factor is None:
        print("\n📏 Translation Scaling Factor")
        print("This controls the scale of movement in the 3D scene:")
        print("  • Indoor scenes: typically 3")
        print("  • Outdoor scenes: typically 10")
        translation_scaling_factor = typer.prompt(
            "Enter translation scaling factor",
            type=float,
            default=3.0,
            show_default=True
        )
    
    print(f"Translation scaling factor: {translation_scaling_factor}")
    
    print("\n🚀 Starting scene expansion...")
    print("⏱️  This process will take several hours to complete.")
    print("   The system will continue running until finished.\n")
    
    work_dir = run_scene_expansion(
        str(input_path),
        translation_scaling_factor,
        root_dir=root_dir,
        num_images_for_vggt=num_images_for_vggt
    )
    print(f"\n✅ Scene expansion completed!")
    print(f"Results saved in: {work_dir}")


@app.command()
def scaffold(
    theme: Optional[str] = typer.Argument(None, help="The theme for generation (e.g., 'Rustic Farmhouse (Wood, Leather, Wool)'). Not used with --custom"),
    mode: GenerationMode = typer.Option(GenerationMode.manual, "--mode", "-m", help="Generation mode: fast (single output), automatic (CLIP-based selection), or manual (user selection)"),
    custom: bool = typer.Option(False, "--custom", "-c", help="Enable custom mode for outdoor/custom scene generation with user-provided prompts")
):
    """Generate only the scaffold (no scene expansion)."""

    # Check and download checkpoints if needed
    check_and_download_checkpoints()

    print(f"\n{'='*80}")
    print("SCAFFOLD GENERATION ONLY")
    print(f"{'='*80}")
    
    if custom:
        # Custom mode - force manual mode
        if mode != GenerationMode.manual:
            print("\n⚠️  Note: Custom mode requires manual selection for best results.")
            print("   Switching to manual mode...")
            mode = GenerationMode.manual
        
        print("\n🎨 Custom Scene Generation Mode")
        print("Please provide 4 prompts for generating a panoramic scene.")
        print("Each prompt will generate one of the 4 cardinal directions.")
        print("Note: The eye-level camera angle will be automatically added.\n")
        
        custom_prompts = []
        for i in range(4):
            direction = ["North", "West", "South", "East"][i]
            prompt = typer.prompt(f"Prompt for {direction} view")
            custom_prompts.append(prompt)
        
        parent_folder, output_folder, final_folder = run_scaffold_generation(
            theme="custom",
            mode=mode,
            custom=True,
            custom_prompts=custom_prompts
        )
    else:
        if theme is None:
            print("❌ Error: Theme is required when not using --custom mode")
            raise typer.Exit(1)
        
        parent_folder, output_folder, final_folder = run_scaffold_generation(theme, mode)
    
    print(f"\n✅ Scaffold generation completed!")
    print(f"Output folder: {output_folder}")
    print(f"Final images: {final_folder}")
    print(f"\nTo expand this scaffold later, run:")
    print(f"  python worldexplorer.py expand '{final_folder}'")


@app.command()
def info():
    """Display information about WorldExplorer."""
    print(f"\n{'='*80}")
    print("🌍 WORLDEXPLORER - AI-Powered 3D Scene Generation")
    print(f"{'='*80}")
    
    print("\n📚 Overview:")
    print("WorldExplorer transforms text descriptions into immersive 3D environments")
    print("through a two-stage process:")
    print("  1. Scaffold Generation - Creates panoramic images from text")
    print("  2. Scene Expansion - Converts panoramas into full 3D scenes")
    
    print("\n📋 Generation Modes:")
    print("  • fast      - Quick generation with single output per view")
    print("  • automatic - Multiple variations with AI-based selection")
    print("  • manual    - Generate variations for human curation")
    
    print("\n🎯 Commands:")
    print("\n1. Full Pipeline (indoor scenes):")
    print("   python worldexplorer.py generate 'Modern Office' --mode fast")
    print("\n2. Custom/Outdoor Scenes (manual mode only):")
    print("   python worldexplorer.py generate --custom")
    print("   # You'll be prompted for 4 custom prompts (N, W, S, E)")
    print("   # and translation scaling factor (3 for indoor, 10 for outdoor)")
    print("\n3. Scaffold Only (no 3D expansion):")
    print("   python worldexplorer.py scaffold 'Rustic Farmhouse' --mode manual")
    print("   python worldexplorer.py scaffold --custom")
    print("\n4. Expand Existing Panorama:")
    print("   python worldexplorer.py expand './panoramas/[name]/[timestamp]/final'")
    print("\n5. Manual Curation Workflow:")
    print("   python worldexplorer.py scaffold 'Beach House' --mode manual")
    print("   # Manually select best inpainted images")
    print("   python worldexplorer.py expand './panoramas/[name]/[timestamp]/final'")
    
    print("\n⏱️  Time Estimates:")
    print("  • Scaffold generation: 5 minutes")
    print("  • Scene expansion: 6-7 hours")
    
    print("\n💡 Tips:")
    print("  • Manual mode often produces better results")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    app()