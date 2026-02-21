# THIS IS THE FINAL SCAFFOLD GENERATION SCRIPT

import os
import torch
from diffusers import FluxPipeline
from .depth_utils import o3d_pcd_to_torch, pcd_from_image
import open3d as o3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras,
)
from PIL import Image
from pytorch3d.transforms import RotateAxisAngle
import os
import numpy as np
from pytorch3d.vis.plotly_vis import plot_scene
from diffusers.utils import load_image
from .utils import compute_focal_lengths, generate_mask, generate_rotation_matrices
from datetime import datetime
from pytorch3d.io import IO
import re
from diffusers import StableDiffusionInpaintPipeline
from typing import Optional
from enum import Enum
import shutil
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore

class GenerationMode(str, Enum):
    fast = "fast"
    automatic = "automatic"
    manual = "manual"

DIMENSION = 576
DEVICE = "cuda"
NUM_INFERENCE_STEPS = 100
FOV = 60
eyelevel = "with the horizon line centered, shot from an eye-level, straight-on camera, with no tilt"

# Global CLIP model instance - loaded once and reused
_clip_model = None

def get_clip_model(device="cuda"):
    """Get or initialize the global CLIP model instance."""
    global _clip_model
    if _clip_model is None:
        _clip_model = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    return _clip_model

def cleanup_clip_model():
    """Clean up the global CLIP model to free GPU memory."""
    global _clip_model
    if _clip_model is not None:
        _clip_model.to("cpu")
        _clip_model = None
        torch.cuda.empty_cache()

def create_clean_folder_name(theme_name):
    """Create a clean folder name from the theme name without timestamp."""
    # Create a clean name by removing special characters and replacing spaces with underscores
    clean_name = re.sub(r'[^\w\s]', '', theme_name)
    clean_name = clean_name.replace(' ', '_')
    return clean_name

def calculate_clip_score_for_image(image_path, prompt, device="cuda"):
    """Calculate CLIP score for a single image."""
    clip_metric = get_clip_model(device)
    
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1))
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    score = clip_metric(img_tensor, prompt)
    return score.item()

def process_theme(theme_data, gen_pipe, inpaint_pipe, parent_folder, generation_mode=GenerationMode.fast, is_custom=False, input_image_path=None):
    """Process a single theme to generate panorama images."""
    theme_name = theme_data["name"]
    theme = theme_data.get("theme", "")
    prompts = theme_data["prompts"]
    
    # Use full theme for display, truncated name for folder/file naming
    display_theme = theme if theme else theme_name
    print(f"Processing theme: {display_theme}")
    
    # Create a folder for this theme (this will now be the parent folder)
    folder_name = create_clean_folder_name(theme_name)
    output_folder = parent_folder  # parent_folder now contains the full path
    os.makedirs(output_folder, exist_ok=True)
    
    # Save theme info to a file
    with open(f"{output_folder}/theme_info.txt", "w") as f:
        if is_custom:
            f.write("Custom Scene Generation\n\n")
            for i, prompt in enumerate(prompts):
                f.write(f"Prompt {i*2}: {prompt}\n\n")
        else:
            f.write(f"Theme: {theme}\n\n")
            for i, prompt in enumerate(prompts):
                prompt_text = prompt.format(theme=theme)
                f.write(f"Prompt {i*2}: {prompt_text}\n\n")
        if input_image_path:
            f.write(f"\nInput image: {input_image_path}\n")
    
    # Handle input image for index 0 if provided
    if input_image_path and os.path.exists(input_image_path):
        print(f"  Using input image as 000.png: {input_image_path}")
        input_image = Image.open(input_image_path)
        # Resize to DIMENSION x DIMENSION if needed
        if input_image.size != (DIMENSION, DIMENSION):
            print(f"  Resizing input image from {input_image.size} to ({DIMENSION}, {DIMENSION})")
            input_image = input_image.resize((DIMENSION, DIMENSION), Image.LANCZOS)
        input_image.save(f"{output_folder}/000.png")
        print(f"  Saved input image as 000.png")
    
    # Generate the initial images (000, 002, 004, 006)
    # Skip 000 if we're using an input image
    for i, prompt_template in enumerate(prompts):
        image_idx = i * 2
        
        # Skip image 000 if we're using input image
        if image_idx == 0 and input_image_path and os.path.exists(input_image_path):
            continue
        
        if is_custom:
            # For custom mode, append eyelevel to the prompt
            prompt = f"{prompt_template}, {eyelevel}"
        else:
            # For standard mode, format with theme
            prompt = prompt_template.format(theme=theme)
        
        print(f"Generating image {image_idx} for {display_theme}")
        try:
            image = gen_pipe(
                prompt,
                output_type="pil",
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=torch.Generator("cpu"),
                height=DIMENSION,
                width=DIMENSION,
            ).images[0]
        except (NotImplementedError, RuntimeError) as e:
            if "memory_efficient_attention" in str(e) or "xformers" in str(e).lower():
                print(f"  Warning: xformers error detected, disabling and retrying...")
                # Disable xformers and retry
                try:
                    if hasattr(gen_pipe, 'disable_xformers_memory_efficient_attention'):
                        gen_pipe.disable_xformers_memory_efficient_attention()
                    gen_pipe.enable_attention_slicing()
                except Exception:
                    pass
                # Retry generation
                image = gen_pipe(
                    prompt,
                    output_type="pil",
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    generator=torch.Generator("cpu"),
                    height=DIMENSION,
                    width=DIMENSION,
                ).images[0]
            else:
                raise
        image.save(f"{output_folder}/00{image_idx}.png")
    
    # Generate point cloud and process images
    fx, fy = compute_focal_lengths(DIMENSION, DIMENSION, FOV)
    print(f"fx = {fx:.2f}, fy = {fy:.2f}")

    # rotation_matrices = generate_rotation_matrices([45, 135, 225, 315]).to(DEVICE)
    rotation_matrices = generate_rotation_matrices([315, 225, 135, 45]).to(DEVICE)
    T = torch.tensor([[0.0, 0.0, 0.0]])
    cameras = FoVPerspectiveCameras(device=DEVICE, R=rotation_matrices, T=T, fov=FOV, znear=0.01, aspect_ratio=DIMENSION/DIMENSION)

    raster_settings = PointsRasterizationSettings(
        image_size=(int(DIMENSION), int(DIMENSION)),      
        radius=0.005,         
        points_per_pixel=50, 
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    white_background = (1, 1, 1)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=NormWeightedCompositor(background_color=white_background),
    )

    # Process each image to create point clouds
    o3d_pcd_0 = pcd_from_image(f"{output_folder}/000.png", fx, fy, outdir=output_folder)
    o3d_pcd_2 = pcd_from_image(f"{output_folder}/002.png", fx, fy, outdir=output_folder)
    o3d_pcd_4 = pcd_from_image(f"{output_folder}/004.png", fx, fy, outdir=output_folder)
    o3d_pcd_6 = pcd_from_image(f"{output_folder}/006.png", fx, fy, outdir=output_folder)
    
    points_0, colors_0 = o3d_pcd_to_torch(o3d_pcd_0, device=DEVICE)
    points_2, colors_2 = o3d_pcd_to_torch(o3d_pcd_2, device=DEVICE)
    points_4, colors_4 = o3d_pcd_to_torch(o3d_pcd_4, device=DEVICE)
    points_6, colors_6 = o3d_pcd_to_torch(o3d_pcd_6, device=DEVICE)

    # pcd_rotation_matrices = generate_rotation_matrices([90, 180, 270]).to(DEVICE)
    pcd_rotation_matrices = generate_rotation_matrices([270, 180, 90]).to(DEVICE)
    rotated_points_2 = torch.matmul(points_2, pcd_rotation_matrices[0].T)
    rotated_points_4 = torch.matmul(points_4, pcd_rotation_matrices[1].T)
    rotated_points_6 = torch.matmul(points_6, pcd_rotation_matrices[2].T)

    points = torch.cat([points_0, rotated_points_2, rotated_points_4, rotated_points_6], dim=0)
    colors = torch.cat([colors_0, colors_2, colors_4, colors_6], dim=0)
    pytorch3d_pc = Pointclouds(points=[points for _ in range(4)], features=[colors for _ in range(4)])
    images = renderer(pytorch3d_pc)
    
    # Plot all
    fig_data = {
        "cams": cameras,
        "pcl": pytorch3d_pc
    }
    fig = plot_scene({"cams + reconstruction": fig_data})
    fig.write_html(f"{output_folder}/plotly.html")

    # Inpaint at corners
    for i, image in enumerate(images):
        j = i * 2 + 1
        image_array = (image.cpu().numpy() * 255).clip(0, 255)
        image_array = image_array.astype(np.uint8)  
        pil_image = Image.fromarray(image_array)
        pil_image.save(f"{output_folder}/rotate{j}.png")

        # "corner of a modern penthouse apartment, field of view {FOV} degrees",
        rotated_image = load_image(f"{output_folder}/rotate{j}.png")
        mask_image = generate_mask(rotated_image)
        mask_image.save(f"{output_folder}/mask{j}.png")
        
        if generation_mode == GenerationMode.fast:
            # Fast mode: only generate one image per inpaint location
            seed = 0
            generator = torch.Generator("cuda").manual_seed(seed)
            guidance_scale = 7.0
            try:
                image = inpaint_pipe(prompt="blend", negative_prompt="two images, text, sun, lines, dividers, advertisement, ad, poster, banner, inline, person, people, phone", image=rotated_image, mask_image=mask_image, generator=generator, guidance_scale=guidance_scale, height=576, width=576).images[0]
            except (NotImplementedError, RuntimeError) as e:
                if "memory_efficient_attention" in str(e) or "xformers" in str(e).lower():
                    print(f"  Warning: xformers error in inpainting, disabling and retrying...")
                    try:
                        if hasattr(inpaint_pipe, 'disable_xformers_memory_efficient_attention'):
                            inpaint_pipe.disable_xformers_memory_efficient_attention()
                        inpaint_pipe.enable_attention_slicing()
                    except Exception:
                        pass
                    image = inpaint_pipe(prompt="blend", negative_prompt="two images, text, sun, lines, dividers, advertisement, ad, poster, banner, inline, person, people, phone", image=rotated_image, mask_image=mask_image, generator=generator, guidance_scale=guidance_scale, height=576, width=576).images[0]
                else:
                    raise
            image.save(f"{output_folder}/00{j}.png")
        else:
            # Automatic and Manual modes: generate all variations
            seeds = [0, 42, 12345, 99999999, 1234567890, 3141592653, 2718281828, 12344567890]
            generators = [torch.Generator("cuda").manual_seed(seed) for seed in seeds]
            guidance_scales = [7.0, 7.5, 8.0, 9.0, 4, 5, 6.0, 10.0, 11.0, 12.0, 13.0]
            
            generated_images = []
            xformers_disabled = False
            for guidance_scale in guidance_scales:
                for i, generator in enumerate(generators):
                    try:
                        image = inpaint_pipe(prompt="blend", negative_prompt="two images, text, sun, lines, dividers, advertisement, ad, poster, banner, inline, person, people, phone", image=rotated_image, mask_image=mask_image, generator=generator, guidance_scale=guidance_scale, height=576, width=576).images[0]
                    except (NotImplementedError, RuntimeError) as e:
                        if ("memory_efficient_attention" in str(e) or "xformers" in str(e).lower()) and not xformers_disabled:
                            print(f"  Warning: xformers error in inpainting, disabling and retrying...")
                            try:
                                if hasattr(inpaint_pipe, 'disable_xformers_memory_efficient_attention'):
                                    inpaint_pipe.disable_xformers_memory_efficient_attention()
                                inpaint_pipe.enable_attention_slicing()
                                xformers_disabled = True
                            except Exception:
                                pass
                            image = inpaint_pipe(prompt="blend", negative_prompt="two images, text, sun, lines, dividers, advertisement, ad, poster, banner, inline, person, people, phone", image=rotated_image, mask_image=mask_image, generator=generator, guidance_scale=guidance_scale, height=576, width=576).images[0]
                        else:
                            raise
                    filename = f"{output_folder}/00{j}_blend_{guidance_scale}_{seeds[i]}.png"
                    image.save(filename)
                    generated_images.append(filename)
    
    # Create final output folder based on generation mode
    final_folder = f"{output_folder}/final"
    os.makedirs(final_folder, exist_ok=True)
    
    if generation_mode == GenerationMode.fast:
        # Fast mode: copy generated images directly to final folder
        for i in range(8):
            src = f"{output_folder}/00{i}.png"
            if os.path.exists(src):
                shutil.copy2(src, f"{final_folder}/00{i}.png")
        print(f"Fast mode: Final images saved in {final_folder}")
    
    elif generation_mode == GenerationMode.automatic:
        # Automatic mode: select best images using CLIP score
        print("Selecting best images using CLIP scores...")
        
        # Copy the original generated images (even indices)
        for i in [0, 2, 4, 6]:
            shutil.copy2(f"{output_folder}/00{i}.png", f"{final_folder}/00{i}.png")
        
        # Get the exact prompts used to generate the original images
        prompts = theme_data["prompts"]
        formatted_prompts = [p.format(theme=theme) for p in prompts]
        
        # Map positions to their neighboring prompts (exact prompts combined with "and")
        inpaint_prompts = {
            1: f"{formatted_prompts[0]} and {formatted_prompts[1]}",  # e.g., "kitchen of a Rustic Farmhouse and living room of a Rustic Farmhouse"
            3: f"{formatted_prompts[1]} and {formatted_prompts[2]}",  
            5: f"{formatted_prompts[2]} and {formatted_prompts[3]}",  
            7: f"{formatted_prompts[3]} and {formatted_prompts[0]}"   # wrapping around
        }
        
        # For each inpainted position, select the best one using combined neighbor prompts
        for j in [1, 3, 5, 7]:
            best_score = -float('inf')
            best_image = None
            
            # Find all variations for this position
            import glob
            pattern = f"{output_folder}/00{j}_*.png"
            variations = glob.glob(pattern)
            
            if variations:
                # Calculate CLIP scores using the combined prompt of neighboring scenes
                clip_prompt = inpaint_prompts[j]
                for img_path in variations:
                    score = calculate_clip_score_for_image(img_path, clip_prompt)
                    if score > best_score:
                        best_score = score
                        best_image = img_path
                
                if best_image:
                    shutil.copy2(best_image, f"{final_folder}/00{j}.png")
                    print(f"  Selected {os.path.basename(best_image)} for position {j} (CLIP score: {best_score:.3f})")
        
        print(f"Automatic mode: Best images selected and saved in {final_folder}")
        
        # Clean up CLIP model after automatic selection
        cleanup_clip_model()
    
    elif generation_mode == GenerationMode.manual:
        # Manual mode: copy original images and let user select inpainted ones
        print("Manual mode: Preparing final folder for user selection...")
        
        # Copy the original generated images (even indices)
        for i in [0, 2, 4, 6]:
            shutil.copy2(f"{output_folder}/00{i}.png", f"{final_folder}/00{i}.png")
        
        print(f"Manual mode: Original images (000, 002, 004, 006) copied to {final_folder}")
        print(f"Please manually select the best inpainted images (001, 003, 005, 007) from {output_folder}")
        print(f"and copy them to {final_folder} with the appropriate names (001.png, 003.png, etc.)")
    
    print(f"Completed processing theme: {display_theme}")
    return folder_name


def run_inpainting_from_four_images(
    output_folder: str,
    generation_mode: GenerationMode = GenerationMode.fast,
    inpaint_pipe=None,
):
    """Run point-cloud + render + inpainting when 000, 002, 004, 006 already exist.
    Used by the hybrid SEVA+inpainting scaffold: SEVA provides the 4 cardinal views,
    this function fills in 001, 003, 005, 007 via inpainting.
    """
    for i in [0, 2, 4, 6]:
        p = os.path.join(output_folder, f"00{i}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required image not found: {p}. Need 000, 002, 004, 006.")
    if inpaint_pipe is None:
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        inpaint_pipe.enable_model_cpu_offload()
        try:
            inpaint_pipe.enable_attention_slicing(slice_size="max")
        except Exception:
            try:
                inpaint_pipe.enable_attention_slicing()
            except Exception:
                pass

    fx, fy = compute_focal_lengths(DIMENSION, DIMENSION, FOV)
    rotation_matrices = generate_rotation_matrices([315, 225, 135, 45]).to(DEVICE)
    T = torch.tensor([[0.0, 0.0, 0.0]])
    cameras = FoVPerspectiveCameras(device=DEVICE, R=rotation_matrices, T=T, fov=FOV, znear=0.01, aspect_ratio=DIMENSION/DIMENSION)
    raster_settings = PointsRasterizationSettings(
        image_size=(int(DIMENSION), int(DIMENSION)),
        radius=0.005,
        points_per_pixel=50,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    white_background = (1, 1, 1)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=NormWeightedCompositor(background_color=white_background),
    )

    o3d_pcd_0 = pcd_from_image(f"{output_folder}/000.png", fx, fy, outdir=output_folder)
    o3d_pcd_2 = pcd_from_image(f"{output_folder}/002.png", fx, fy, outdir=output_folder)
    o3d_pcd_4 = pcd_from_image(f"{output_folder}/004.png", fx, fy, outdir=output_folder)
    o3d_pcd_6 = pcd_from_image(f"{output_folder}/006.png", fx, fy, outdir=output_folder)
    points_0, colors_0 = o3d_pcd_to_torch(o3d_pcd_0, device=DEVICE)
    points_2, colors_2 = o3d_pcd_to_torch(o3d_pcd_2, device=DEVICE)
    points_4, colors_4 = o3d_pcd_to_torch(o3d_pcd_4, device=DEVICE)
    points_6, colors_6 = o3d_pcd_to_torch(o3d_pcd_6, device=DEVICE)
    pcd_rotation_matrices = generate_rotation_matrices([270, 180, 90]).to(DEVICE)
    rotated_points_2 = torch.matmul(points_2, pcd_rotation_matrices[0].T)
    rotated_points_4 = torch.matmul(points_4, pcd_rotation_matrices[1].T)
    rotated_points_6 = torch.matmul(points_6, pcd_rotation_matrices[2].T)
    points = torch.cat([points_0, rotated_points_2, rotated_points_4, rotated_points_6], dim=0)
    colors = torch.cat([colors_0, colors_2, colors_4, colors_6], dim=0)
    pytorch3d_pc = Pointclouds(points=[points for _ in range(4)], features=[colors for _ in range(4)])
    images = renderer(pytorch3d_pc)

    for i, image in enumerate(images):
        j = i * 2 + 1
        image_array = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        pil_image.save(f"{output_folder}/rotate{j}.png")
        rotated_image = load_image(f"{output_folder}/rotate{j}.png")
        mask_image = generate_mask(rotated_image)
        mask_image.save(f"{output_folder}/mask{j}.png")
        seed = 0
        generator = torch.Generator("cuda").manual_seed(seed)
        guidance_scale = 7.0
        try:
            image = inpaint_pipe(
                prompt="blend",
                negative_prompt="two images, text, sun, lines, dividers, advertisement, ad, poster, banner, inline, person, people, phone",
                image=rotated_image,
                mask_image=mask_image,
                generator=generator,
                guidance_scale=guidance_scale,
                height=576,
                width=576,
            ).images[0]
        except (NotImplementedError, RuntimeError) as e:
            if "memory_efficient_attention" in str(e) or "xformers" in str(e).lower():
                try:
                    if hasattr(inpaint_pipe, 'disable_xformers_memory_efficient_attention'):
                        inpaint_pipe.disable_xformers_memory_efficient_attention()
                    inpaint_pipe.enable_attention_slicing()
                except Exception:
                    pass
                image = inpaint_pipe(
                    prompt="blend",
                    negative_prompt="two images, text, sun, lines, dividers, advertisement, ad, poster, banner, inline, person, people, phone",
                    image=rotated_image,
                    mask_image=mask_image,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    height=576,
                    width=576,
                ).images[0]
            else:
                raise
        image.save(f"{output_folder}/00{j}.png")

    final_folder = f"{output_folder}/final"
    os.makedirs(final_folder, exist_ok=True)
    for i in range(8):
        src = f"{output_folder}/00{i}.png"
        if os.path.exists(src):
            shutil.copy2(src, f"{final_folder}/00{i}.png")
    print(f"Inpainting from four images completed. Final scaffold: {final_folder}")
    return final_folder


def run_scaffold_generation(theme, mode=GenerationMode.fast, parent_folder=None, custom=False, custom_prompts=None, input_image_path=None):
    """Run scaffold generation for a theme and return output paths.
    
    Args:
        theme: Theme name for generation
        mode: Generation mode (fast, automatic, manual)
        parent_folder: Output folder path
        custom: Whether to use custom prompts
        custom_prompts: List of 4 custom prompts (if custom=True)
        input_image_path: Optional path to input image to use as 000.png
    """
    
    if custom:
        # For custom mode, use provided prompts
        if custom_prompts is None or len(custom_prompts) != 4:
            raise ValueError("Custom mode requires exactly 4 prompts")
        
        # Create a descriptive name from the first prompt (truncated)
        theme_name = custom_prompts[0][:30].lower().replace(' ', '_')
        theme_name = re.sub(r'[^\w\s_]', '', theme_name)
        
        # Create theme data structure for custom prompts
        theme_data = {
            "name": theme_name,
            "theme": None,  # No theme for custom mode
            "prompts": custom_prompts
        }
        
        print("Custom Scene Generation")
        print(f"Derived name: {theme_name}")
    else:
        # Standard mode - derive name from theme
        theme_name = theme[:17].lower().replace(' ', '_')
        
        # Generate prompts based on the theme
        prompts = [
            "kitchen of a {theme}",
            "office of a {theme}",
            "bedroom of a {theme}",
            "living room of a {theme}"
        ]
        
        # Create theme data structure
        theme_data = {
            "name": theme_name,
            "theme": theme,
            "prompts": prompts
        }
        
        print(f"Theme: {theme}")
        print(f"Derived name: {theme_name}")
    
    print("Initializing image generation pipelines...")
    gen_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    gen_pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU
    # Use attention slicing instead of xformers to avoid dtype compatibility issues
    try:
        gen_pipe.enable_attention_slicing(slice_size="max")
    except Exception:
        try:
            gen_pipe.enable_attention_slicing()
        except Exception:
            pass  # If not available, continue without it
    
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    inpaint_pipe.enable_model_cpu_offload()
    # Use attention slicing instead of xformers to avoid dtype compatibility issues
    try:
        inpaint_pipe.enable_attention_slicing(slice_size="max")
    except Exception:
        try:
            inpaint_pipe.enable_attention_slicing()
        except Exception:
            pass  # If not available, continue without it
    
    # Create main panoramas directory if it doesn't exist
    os.makedirs("./panoramas", exist_ok=True)
    
    # Create the theme folder structure: panoramas/theme_abbreviation/timestamp
    folder_name = create_clean_folder_name(theme_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use provided parent folder or create a new one with inverted structure
    if parent_folder is None:
        parent_folder = f"./panoramas/{folder_name}/{timestamp}"
    os.makedirs(parent_folder, exist_ok=True)
    
    # Process the theme with the specified generation mode
    _ = process_theme(theme_data, gen_pipe, inpaint_pipe, parent_folder, generation_mode=mode, is_custom=custom, input_image_path=input_image_path)
    
    # Create a summary file
    with open(f"{parent_folder}/summary.txt", "w") as f:
        f.write("Generated Panorama:\n\n")
        f.write(f"1. {folder_name} - {theme_name}\n")
        f.write(f"Mode: {mode}\n")
        if custom:
            f.write("Type: Custom\n")
    
    print(f"All themes processed successfully in folder: {parent_folder}")
    
    # Return paths for use by other modules (with new structure)
    output_folder = parent_folder  # Now the parent folder IS the output folder
    final_folder = f"{output_folder}/final"
    return parent_folder, output_folder, final_folder

# Entry point removed - use worldexplorer.py instead