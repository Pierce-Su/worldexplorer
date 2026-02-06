# Guide: Generating 3D Scenes from a Single Image

This guide explains how to generate a 3D scene from a single perspective image using WorldExplorer.

## Overview

The process involves three main steps:

1. **Generate Multiple Views**: Use Stable Virtual Camera (SEVA) to generate additional views from your single input image
2. **Create Scaffold**: Extract 8 key frames to create a panoramic scaffold
3. **Expand to 3D**: Use WorldExplorer's scene expansion to create a full navigable 3D scene

## Quick Start

### Option 1: Using the Helper Script (Recommended)

The easiest way is to use the provided helper script:

```bash
python single_image_to_3d.py generate <path_to_your_image.png> --traj-prior orbit
```

This will:
- Generate multiple views from your image
- Extract 8 key frames
- Optionally expand to full 3D scene

**Full example:**
```bash
# Generate views and expand to 3D scene
python single_image_to_3d.py generate my_image.png --traj-prior orbit --translation-scaling 3.0

# Generate views only (skip 3D expansion)
python single_image_to_3d.py generate my_image.png --traj-prior orbit --skip-expansion

# Expand an existing scaffold later
python single_image_to_3d.py expand ./single_image_scenes/my_scene/final --translation-scaling 3.0
```

### Option 2: Manual Workflow

If you prefer more control, follow these steps:

#### Step 1: Generate Views from Single Image

Use Stable Virtual Camera's `img2trajvid_s-prob` task to generate multiple views:

```bash
# Create a directory with your input image
mkdir -p input_scene
cp your_image.png input_scene/scene.png

# Generate views using orbit trajectory
python model/stable-virtual-camera/demo.py \
    --data_path input_scene \
    --task img2trajvid_s-prob \
    --replace_or_include_input True \
    --traj_prior orbit \
    --cfg 4.0,2.0 \
    --guider 1,2 \
    --num_targets 80 \
    --L_short 576 \
    --use_traj_prior True \
    --chunk_strategy interp
```

**Available trajectory options:**
- `orbit` - Circular orbit around the scene (good for showing 3D structure)
- `spiral` - Spiral motion
- `pan-left`, `pan-right` - Horizontal panning
- `move-forward`, `move-backward` - Forward/backward movement
- `move-up`, `move-down` - Vertical movement
- `zoom-in`, `zoom-out` - Zoom effects
- `dolly-zoom-in`, `dolly-zoom-out` - Dolly zoom effects

**Note:** For panning motions, you may want to increase `camera_scale`:
```bash
--camera_scale 10.0  # Add this for pan-left/pan-right/dolly-zoom motions
```

#### Step 2: Extract Key Frames

The generated views will be saved in `work_dirs/backwards/img2trajvid_s-prob/<scene_name>/samples-rgb/`.

Extract 8 evenly-spaced frames to create your scaffold:

```bash
# Create scaffold directory
mkdir -p scaffold/images

# Copy 8 evenly-spaced frames (adjust paths as needed)
# You can use a simple Python script or manually select frames
python -c "
import glob
import shutil
import numpy as np
import os

frames = sorted(glob.glob('work_dirs/backwards/img2trajvid_s-prob/*/samples-rgb/*.png'))
indices = np.linspace(0, len(frames)-1, 8, dtype=int)
for i, idx in enumerate(indices):
    shutil.copy2(frames[idx], f'scaffold/images/{i:03d}.png')
"
```

#### Step 3: Prepare Final Scaffold

Create the final scaffold directory with images named `000.png` through `007.png`:

```bash
mkdir -p final_scaffold
cp scaffold/images/000.png final_scaffold/000.png
cp scaffold/images/001.png final_scaffold/001.png
# ... continue for all 8 images
```

Or use the helper script's `prepare_final_scaffold` function.

#### Step 4: Expand to 3D Scene

Use WorldExplorer's expand command:

```bash
python worldexplorer.py expand final_scaffold --translation-scaling 3.0
```

**Translation Scaling Factors:**
- **Indoor scenes**: 3.0 (default), range 2-8
- **Outdoor scenes**: 10.0, range 6-20

## Detailed Workflow Explanation

### Why 8 Images?

WorldExplorer's scene expansion expects 8 images representing a panoramic view:
- Images 000, 002, 004, 006: Cardinal directions (North, West, South, East)
- Images 001, 003, 005, 007: In-between views

When starting from a single image, we generate multiple views and extract 8 key frames that best represent different viewing angles.

### Generating the 3 Additional Views

You mentioned needing to generate 3 other views. Here's how:

1. **Using orbit trajectory**: Generates views rotating around the scene
2. **Using pan trajectories**: Generates views panning left/right/up/down
3. **Using move trajectories**: Generates views moving forward/backward

The `img2trajvid_s-prob` task generates a video sequence, from which you extract key frames.

### Recommended Trajectories for Different Scenes

- **Indoor scenes**: `orbit` or `spiral` (shows room structure)
- **Outdoor scenes**: `pan-left` or `pan-right` (shows landscape)
- **Close-up objects**: `orbit` with higher `camera_scale`
- **Architectural**: `move-forward` or `move-backward`

## Command Reference

### Helper Script Commands

```bash
# Generate from single image
python single_image_to_3d.py generate <image_path> [options]

Options:
  --traj-prior, -t      Camera trajectory (default: orbit)
  --num-targets, -n     Number of frames to generate (default: 80)
  --cfg                 CFG scale "first,second" (default: "4.0,2.0")
  --skip-expansion      Skip 3D scene expansion
  --translation-scaling, -s  Translation scaling (default: 3.0)
  --output, -o          Output directory (default: ./single_image_scenes)

# Expand existing scaffold
python single_image_to_3d.py expand <scaffold_dir> [options]

Options:
  --translation-scaling, -t  Translation scaling (default: 3.0)
```

### Direct SEVA Commands

```bash
# Basic single image to video
python model/stable-virtual-camera/demo.py \
    --data_path <folder_with_image> \
    --task img2trajvid_s-prob \
    --replace_or_include_input True \
    --traj_prior orbit \
    --cfg 4.0,2.0 \
    --guider 1,2 \
    --num_targets 80 \
    --L_short 576 \
    --use_traj_prior True \
    --chunk_strategy interp

# For panning motions (increase camera scale)
python model/stable-virtual-camera/demo.py \
    --data_path <folder_with_image> \
    --task img2trajvid_s-prob \
    --replace_or_include_input True \
    --traj_prior pan-left \
    --cfg 4.0,2.0 \
    --camera_scale 10.0 \
    --guider 1,2 \
    --num_targets 80 \
    --L_short 576 \
    --use_traj_prior True \
    --chunk_strategy interp
```

## Troubleshooting

### Issue: Generated views look blurry or inconsistent

**Solution**: Increase the CFG scale for the first pass:
```bash
--cfg 6.0,2.0  # Instead of 4.0,2.0
```

### Issue: Not enough camera motion

**Solution**: Increase camera scale:
```bash
--camera_scale 5.0  # Or higher for more motion
```

### Issue: Need more frames to choose from

**Solution**: Generate more frames:
```bash
--num_targets 120  # Instead of 80
```

### Issue: Scaffold expansion fails

**Solution**: Make sure you have exactly 8 images named `000.png` through `007.png` in the scaffold directory.

## Time Estimates

- **View generation**: ~2-5 minutes (depending on num_targets)
- **Scaffold preparation**: < 1 minute
- **3D scene expansion**: 6-7 hours

## Output Structure

After running the full pipeline:

```
single_image_scenes/
└── <scene_name>/
    ├── generated_views/          # SEVA output
    ├── scaffold/                 # Extracted key frames
    │   ├── images/
    │   │   ├── 000.png
    │   │   ├── ...
    │   │   └── 007.png
    │   └── transforms.json
    └── final/                    # Final scaffold (8 images)
        ├── 000.png
        ├── ...
        └── 007.png

scenes/
└── <scene_name>_<scaling>_<timestamp>/
    └── img2trajvid/              # Generated trajectory videos
        └── ...

nerfstudio_output/
└── <scene_id>/
    └── splatfacto/
        └── <timestamp>/
            ├── config.yml
            └── exports/
                └── splat/
                    └── splat.ply  # Final 3D model
```

## Viewing Your 3D Scene

After expansion completes, view your scene:

```bash
ns-viewer --load-config ./nerfstudio_output/<scene_id>/splatfacto/<timestamp>/config.yml
```

Or use the workaround command:
```bash
python -c "import sys, torch; sys.argv = ['ns-viewer', '--load-config', './nerfstudio_output/<scene_id>/splatfacto/<timestamp>/config.yml']; orig = torch.load; setattr(torch, 'load', lambda *a, **k: orig(*a, **{**k, 'weights_only': False})); from nerfstudio.scripts.viewer.run_viewer import entrypoint; entrypoint()"
```

## Tips for Best Results

1. **Image Quality**: Use high-resolution images (at least 576x576)
2. **Image Content**: Images with clear depth cues work best
3. **Trajectory Selection**: Choose trajectory that matches your scene type
4. **Translation Scaling**: Adjust based on scene scale (indoor vs outdoor)
5. **Manual Selection**: For best results, manually review and select the best 8 frames

## Example Workflow

```bash
# 1. Generate views from single image
python single_image_to_3d.py generate \
    my_room.jpg \
    --traj-prior orbit \
    --num-targets 100 \
    --skip-expansion

# 2. Review generated views in single_image_scenes/my_room/generated_views/
# 3. Manually select best 8 frames if needed

# 4. Expand to 3D scene
python single_image_to_3d.py expand \
    single_image_scenes/my_room/final \
    --translation-scaling 3.0
```

## Additional Resources

- [WorldExplorer README](README.md) - Main project documentation
- [Stable Virtual Camera CLI Usage](model/stable-virtual-camera/docs/CLI_USAGE.md) - Detailed SEVA documentation
- [Stable Virtual Camera Gradio Demo](model/stable-virtual-camera/docs/GR_USAGE.md) - Interactive demo guide
