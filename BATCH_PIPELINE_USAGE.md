# Batch Pipeline Usage Guide

## Overview

`run_batch_pipeline.py` is a batch processing script that runs the WorldExplorer pipeline on multiple scenes from a curated dataset. It reads metadata from a `metadata.json` file and processes each scene through the full pipeline (or selected stages), handling both scaffold generation and scene expansion.

The script automates the processing of multiple scenes, managing:
- **Stage 1**: Scaffold Generation - Creates panoramic images from text descriptions
- **Stage 2**: Video Generation & Processing - Generates trajectory videos, converts transforms, and generates point clouds
- **Stage 3**: NeRF Training & Export - Trains Gaussian Splatting model and exports to PLY format

## Prerequisites

1. **Dataset Structure**: Your dataset directory must contain:
   - `metadata.json` - JSON file with scene metadata (see format below)
   - `scaffolds/` - Optional directory containing pre-generated scaffold images (if skipping Stage 1)

2. **Dependencies**: All WorldExplorer dependencies must be installed (see main README.md)

3. **Hardware**: 
   - CUDA-capable GPU(s) with sufficient VRAM
   - For scene expansion: ~12GB+ VRAM recommended
   - For NeRF training: Additional VRAM required

4. **Checkpoints**: Ensure required checkpoints are downloaded:
   - Video-Depth-Anything checkpoint
   - Depth_Anything_V2 checkpoint

## Dataset Structure

### Directory Layout

```
curated_set/
├── metadata.json
└── scaffolds/          # Optional: pre-generated scaffolds
    ├── scene_0000/
    │   ├── 000.png
    │   ├── 001.png
    │   └── ... (000-007.png)
    └── scene_0005/
        └── ...
```

### metadata.json Format

The `metadata.json` file should contain a `samples` array with the following structure:

```json
{
  "samples": [
    {
      "index": 0,
      "scene_type": "indoor",
      "category": "living_room",
      "theme": "Modern Apartment",
      "scaffold": {
        "mode": "manual",
        "custom": false,
        "prompts": [
          "North view: A modern living room with minimalist furniture",
          "West view: Kitchen area with modern appliances",
          "South view: Dining area with contemporary design",
          "East view: Bedroom with clean lines"
        ]
      },
      "expansion": {
        "translation_scaling_factor": 3.0,
        "trajectory_order": ["in", "left", "right", "up"],
        "num_images_for_vggt": 40
      }
    },
    {
      "index": 5,
      "scene_type": "outdoor",
      "category": "landscape",
      "theme": "Mountain Vista",
      "scaffold": {
        "mode": "manual",
        "custom": true,
        "prompts": [
          "North view: Snow-capped mountain peaks",
          "West view: Dense forest with pine trees",
          "South view: Valley with a flowing river",
          "East view: Rocky cliffs and waterfalls"
        ]
      },
      "expansion": {
        "translation_scaling_factor": 10.0,
        "trajectory_order": ["in", "left", "right", "up"],
        "num_images_for_vggt": 40
      }
    },
    {
      "index": 10,
      "scene_type": "indoor",
      "category": "office",
      "scaffold": {
        "scaffold_path": "scaffolds/scene_0010"
      },
      "expansion": {
        "translation_scaling_factor": 3.0
      }
    }
  ]
}
```

#### Field Descriptions

- **`index`** (required): Unique identifier for the sample
- **`scene_type`** (optional): Type of scene (e.g., "indoor", "outdoor")
- **`category`** (optional): Scene category (e.g., "living_room", "landscape", "office")
- **`theme`** (optional): Theme name for scaffold generation
- **`scaffold`** (required): Scaffold generation configuration
  - **`mode`**: Generation mode - `fast`, `automatic`, or `manual` (default: `manual`)
  - **`custom`**: Whether to use custom prompts (default: `false`)
  - **`prompts`**: Array of 4 prompts for North, West, South, East views (required if `custom: true`)
  - **`scaffold_path`**: Path to pre-generated scaffold folder (if skipping Stage 1)
- **`expansion`** (optional): Scene expansion configuration
  - **`translation_scaling_factor`**: Movement scale factor (default: 3.0 for indoor, 10.0 for outdoor)
  - **`trajectory_order`**: Order to process trajectories (default: `["in", "left", "right", "up"]`)
  - **`num_images_for_vggt`**: Number of images for VGGT processing (default: 40)

### Translation Scaling Factors

The translation scaling factor controls movement scale in the 3D scene:
- **Indoor scenes**: Use 3.0 (default), recommended range 2.0 - 8.0
- **Outdoor scenes**: Use 10.0, recommended range 6.0 - 20.0

## Basic Usage

### Process All Scenes

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch
```

### Process Specific Indices

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch \
    --indices 0 5 14
```

### Run Only Specific Stages

```bash
# Only run Stage 1 (Scaffold Generation)
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --only_stages 1

# Only run Stages 1 and 2 (Scaffold + Video Generation & Processing)
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --only_stages 1 2

# Only run Stages 2-3 (Skip scaffold, start from existing scaffolds)
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --only_stages 2 3
```

### Continue on Error

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch \
    --continue_on_error
```

## Command-Line Arguments

### Required Arguments

- **`--dataset_dir`**: Path to curated_set directory containing `metadata.json` and optional scaffold folders

### Optional Arguments

#### Output Configuration

- **`--output_base`** (default: `output/batch`): Base output directory. Results will be organized as:
  ```
  output_base/
  ├── index_0000/
  │   ├── scaffold/          # Stage 1 outputs
  │   ├── scenes/            # Stage 2-3 outputs
  │   └── ...
  ├── index_0005/
  │   └── ...
  └── batch_summary.json
  ```

- **`--indices`**: Specific sample indices to process (e.g., `--indices 0 5 14`). If not specified, processes all samples.

- **`--only_stages`**: Only run specified stages (1, 2, or 3). Example: `--only_stages 1 2` runs only stages 1 and 2.

#### Stage 1: Scaffold Generation

- **`--scaffold_mode`** (default: `manual`): Generation mode - `fast`, `automatic`, or `manual`
- **`--default_theme`**: Default theme to use if not specified in metadata

#### Stage 2: Video Generation & Processing

- **`--translation_scaling_factor`** (default: `3.0`): Default translation scaling factor (overridden by metadata)
- **`--trajectory_order`**: Default trajectory order (overridden by metadata)
- **`--num_images_for_vggt`** (default: `40`): Number of images for VGGT processing
- **`--skip_vggt`** (flag): Skip VGGT processing (use existing point clouds if available)

#### Stage 3: NeRF Training & Export

- **`--nerf_steps`** (default: `30000`): Number of training steps
- **`--nerf_output_dir`** (default: `./nerfstudio_output`): Directory for NeRF outputs
- **`--skip_export`** (flag): Skip PLY export

#### General Settings

- **`--device`** (default: `cuda:0`): CUDA device for processing
- **`--continue_on_error`** (flag): Continue processing other scenes if one fails
- **`--root_dir`** (default: `./predefined_trajectories`): Directory containing predefined trajectories

## Output Structure

The script creates an organized output structure:

```
output_base/
├── index_0000/
│   ├── scaffold/
│   │   ├── panoramas/
│   │   │   └── [theme_name]/
│   │   │       └── [timestamp]/
│   │   │           ├── final/          # 8 images (000-007.png)
│   │   │           └── ...
│   │   └── theme_info.txt
│   ├── scenes/
│   │   └── [scene_id]/
│   │       ├── img2trajvid/           # Stage 2 outputs
│   │       │   ├── transforms.json
│   │       │   ├── vggt_pcl.ply
│   │       │   └── ...
│   │       └── ...
│   └── ...
├── index_0005/
│   └── ...
└── batch_summary.json
```

### batch_summary.json

After processing completes, a summary file is created with:
- Total samples processed
- Successfully processed scenes
- Failed scenes
- Elapsed time
- Configuration used
- Detailed results per sample

Example:
```json
{
  "total_samples": 10,
  "total_processed": 8,
  "total_failed": 2,
  "elapsed_time_seconds": 14400.5,
  "configuration": {
    "only_stages": null,
    "translation_scaling_factor": 3.0,
    "device": "cuda:0",
    ...
  },
  "results": [
    {
      "index": 0,
      "scene_type": "indoor",
      "category": "living_room",
      "stages_completed": [1, 2, 3],
      "scaffold_path": "output/batch/index_0000/scaffold/panoramas/.../final",
      "scene_path": "output/batch/index_0000/scenes/...",
      "status": "success"
    },
    ...
  ]
}
```

## Advanced Examples

### High-Quality Processing with Custom Prompts

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch_hq \
    --scaffold_mode manual \
    --translation_scaling_factor 3.0
```

### Process Only Outdoor Scenes

Filter your metadata.json to include only outdoor scenes, or use `--indices` to select specific samples.

### Skip Scaffold Generation (Use Pre-generated Scaffolds)

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch \
    --only_stages 2 3
```

Ensure your metadata.json includes `scaffold_path` for each sample.

### Custom Device and Trajectory Order

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch \
    --device cuda:1 \
    --trajectory_order in left right up
```

### Fast Processing (Skip NeRF Training)

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch_fast \
    --only_stages 1 2
```

### Resume Processing

If processing fails partway through, you can resume by:
1. Checking `batch_summary.json` to see which samples were processed
2. Using `--indices` to process only the remaining samples:

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch \
    --indices 5 14 23 \
    --continue_on_error
```

## Prompt Enhancement

The script automatically enhances prompts with metadata information:

- **Scene Type**: Added as "Scene type: [Scene Type]"
- **Category**: Added as "Category: [Category]"
- **Theme**: Used as context for generation

Example:
- Base prompt: `"A modern living room"`
- Enhanced prompt: `"Scene type: Indoor, Category: Living Room. A modern living room"`

This helps the pipeline better understand the scene context.

## Tips and Best Practices

1. **Start Small**: Test with `--indices` on a few samples first to verify your setup
2. **Monitor VRAM**: Scene expansion requires significant GPU memory; monitor usage during processing
3. **Stage-by-Stage**: Use `--only_stages` to process in stages and verify intermediate results
4. **Error Handling**: Use `--continue_on_error` for batch processing to avoid stopping on single failures
5. **Check Summary**: Always review `batch_summary.json` after processing to identify any failures
6. **Manual Mode**: For best quality, use `manual` mode for scaffold generation and curate results
7. **Translation Scaling**: Adjust `translation_scaling_factor` based on scene type (indoor vs outdoor)
8. **Pre-generated Scaffolds**: For faster iteration, generate scaffolds separately and use `scaffold_path` in metadata

## Troubleshooting

### Error: Dataset directory not found
- Verify the `--dataset_dir` path is correct
- Ensure the directory contains `metadata.json`

### Error: metadata.json not found
- Check that `metadata.json` exists in the dataset directory
- Verify the JSON format is valid

### Error: Scaffold images not found
- If using `scaffold_path`, verify the path exists and contains 8 images (000-007.png)
- Check that filenames match exactly (case-sensitive)

### Out of Memory Errors
- Reduce `--num_images_for_vggt` (default: 40, try 20 or 30)
- Process fewer samples at once using `--indices`
- Ensure sufficient VRAM for NeRF training stage

### Processing Takes Too Long
- Use `--only_stages` to skip unnecessary stages
- Skip NeRF training if you only need video outputs: `--only_stages 1 2`
- Process in smaller batches using `--indices`
- Use `fast` mode for scaffold generation (faster but lower quality)

### NeRF Training Fails
- Check CUDA toolkit installation (required for gsplat compilation)
- Verify `CUDA_HOME` environment variable is set
- Check GPU memory availability
- Review training logs in `nerfstudio_output/`

### Scaffold Generation Fails
- Verify Hugging Face authentication for FLUX.1-dev model
- Check that prompts are provided correctly in metadata
- Ensure sufficient disk space for generated images

## Stage Details

### Stage 1: Scaffold Generation
- Generates 8 panoramic images (000-007.png) from text descriptions
- Supports three modes: `fast`, `automatic`, `manual`
- Custom mode allows 4 separate prompts for each cardinal direction
- Output: Panorama folder with 8 images

### Stage 2: Video Generation & Processing
- Generates trajectory videos for scene expansion
- Processes multiple trajectory types (in, left, right, up)
- Creates video sequences for each of the 8 scaffold images
- Converts video transforms to NeRF format
- Samples images for VGGT point cloud generation
- Aligns and processes point clouds
- Output: Video frames, `transforms.json`, and `vggt_pcl.ply` in `img2trajvid/`

### Stage 3: NeRF Training & Export
- Trains Gaussian Splatting model using splatfacto
- Uses generated videos and point clouds
- Longest stage (several hours)
- Exports trained model to PLY format
- Rotates coordinate system for standard viewers
- Output: Trained model checkpoints, `splat.ply` and `splat_rotated.ply`

## See Also

- `worldexplorer.py` - Main CLI for single scene processing
- `README.md` - Main project documentation
- `SINGLE_IMAGE_GUIDE.md` - Guide for single image to 3D conversion
