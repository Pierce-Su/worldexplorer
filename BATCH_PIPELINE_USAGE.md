# Batch Pipeline Usage Guide

## Overview

`run_batch_pipeline.py` runs the WorldExplorer pipeline on multiple scenes from a curated dataset. It reads `metadata.json`, resolves **photorealistic** and **stylized** image paths per sample, and for each variant runs up to three stages: scaffold → trajectory + VGGT → 3DGS. Output is organized as **`output_base/photorealistic/index_XXXX/`** and **`output_base/stylized/index_XXXX/`**.

The script uses:
- **Stage 1**: Scaffold generation (default: **single_image** = SEVA from one image, so all views stay in context). Other methods: **single_image_hybrid** (SEVA + inpainting), **scaffold_gen** (text prompts + image as 000), **two_images** (0° + 180° guidance images).
- **Stage 2**: Trajectory generation + VGGT (no 3DGS).
- **Stage 3**: 3DGS training and PLY export.

You can run all stages or select stages with `--only_stages`.

## Prerequisites

1. **Dataset**: A directory (e.g. `data/curated_set`) containing:
   - `metadata.json` with a `samples` array (see format below).
   - Image files at paths referenced in the metadata (e.g. under `photorealistic/`, `stylized/`, or `images/`).

2. **Environment**: WorldExplorer dependencies and checkpoints (Video-Depth-Anything, Depth_Anything_V2). Use `--check_checkpoints` to run the usual checkpoint check before processing.

3. **Hardware**: CUDA GPU; ~12GB+ VRAM recommended for scene expansion and 3DGS.

## Dataset Structure

### Directory layout

```
curated_set/
├── metadata.json
├── photorealistic/       # optional: variant images
│   ├── image0.png
│   └── ...
├── stylized/
│   └── ...
└── images/               # optional: fallback images (e.g. 0.png, index_0000.png)
    └── ...
```

### metadata.json format

The file must contain a **`samples`** array. Each sample can define **photorealistic** and/or **stylized** image paths. Paths are relative to the dataset directory unless absolute.

**Variant images** (for `output_base/photorealistic/` and `output_base/stylized/`):

- **`photorealistic`**: object with **`filename`** — image path is `photorealistic/<filename>` (relative to dataset dir).
- **`stylized`**: same: **`filename`** with path `stylized/<filename>`.

If a sample has no variant entries but has another resolvable image (e.g. `scaffold.input_image` or a file under `images/`), the script uses that as a single job under variant **`default`** (when applicable).

Example with `filename` (path = `variant/filename`):

```json
{
  "samples": [
    {
      "index": 0,
      "scene_type": "indoor",
      "category": "living_room",
      "photorealistic": {
        "filename": "scene0_photo.png"
      },
      "stylized": {
        "filename": "scene0_style.png"
      },
      "scaffold": {
        "mode": "manual",
        "custom": true,
        "prompts": [
          "North: modern living room",
          "West: kitchen area",
          "South: dining area",
          "East: bedroom"
        ]
      },
      "expansion": {
        "translation_scaling_factor": 3.0,
        "num_images_for_vggt": 40
      }
    }
  ]
}
```

Example with pre-generated scaffold (skip Stage 1):

```json
{
  "index": 5,
  "scaffold": {
    "scaffold_path": "scaffolds/scene_0005"
  },
  "expansion": {
    "translation_scaling_factor": 10.0
  }
}
```

#### Field summary

| Field | Required | Description |
|-------|----------|-------------|
| `index` | Yes | Unique sample id. |
| `photorealistic` | No* | Object with `filename`; image at `photorealistic/<filename>` (relative to dataset_dir). |
| `stylized` | No* | Object with `filename`; image at `stylized/<filename>`. *At least one variant (or fallback image) must resolve. |
| `scene_type` | No | e.g. `"indoor"`, `"outdoor"` (can affect default translation scaling). |
| `category` | No | e.g. `"living_room"`. |
| `theme` | No | Used when using `scaffold_gen` with theme-based prompts. |
| `scaffold` | No | Config for Stage 1: `mode`, `custom`, `prompts` (4), or `scaffold_path` to use pre-generated scaffold. |
| `expansion` | No | `translation_scaling_factor`, `trajectory_order`, `num_images_for_vggt`. |

### Translation scaling

- **Indoor**: typically `3.0` (default).
- **Outdoor**: typically `10.0`.

## Stage definitions

| Stage | Name | What it does |
|-------|------|--------------|
| **1** | Scaffold | Produces 8 images (000–007.png). **single_image** (default): SEVA from one image → key-frame extraction → final scaffold. **single_image_hybrid**: SEVA for 0,2,4,6 + inpainting for 1,3,5,7. **scaffold_gen**: text prompts + optional image as 000. **two_images**: SEVA with 0° + 180° guidance images and defined poses (requires pre-generated 180° views). |
| **2** | Trajectory + VGGT | From the 8 scaffold images: trajectory/video generation (Stable Virtual Camera), transform conversion, VGGT alignment. Output: `scenes/<scene_id>/img2trajvid/` ready for 3DGS. |
| **3** | 3DGS | Trains NeRFstudio splatfacto-big and exports PLY (`splat.ply`, `splat_rotated.ply`). Requires Stage 2. |

## Basic usage

### Process all samples (all stages)

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --output_base output/batch
```

Default: **single_image** scaffold, then Stage 2 and Stage 3. One job per variant per sample; output under `output/batch/photorealistic/index_XXXX/` and `output/batch/stylized/index_XXXX/`.

### Only Stage 1 (scaffold)

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --only_stages 1
```

### Only Stage 2 (trajectory + VGGT)

Requires existing scaffold (e.g. from a previous Stage 1 run):

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --only_stages 2
```

### Only Stage 3 (3DGS)

Requires existing Stage 2 output under `output_base/<variant>/index_XXXX/scenes/<scene_id>/`:

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --only_stages 3
```

### Specific sample indices and ranges

Limit which samples are processed with `--indices`. Each argument can be a single index or an **inclusive range** (`start-end` or `start:end`), so you can split work across parallel processes:

```bash
# Single indices
python run_batch_pipeline.py --dataset_dir data/curated_set --indices 0 2 5

# Range (e.g. worker 1: 0–24, worker 2: 25–49)
python run_batch_pipeline.py --dataset_dir data/curated_set --indices 0-24
python run_batch_pipeline.py --dataset_dir data/curated_set --indices 25-49

# Mix of ranges and single indices
python run_batch_pipeline.py --dataset_dir data/curated_set --indices 0-9 20 30-32
```

### Hybrid scaffold (SEVA + inpainting)

Often better quality than SEVA-only: SEVA for 0,2,4,6 and inpainting for 1,3,5,7:

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --scaffold_method single_image_hybrid
```

### Text-guided scaffold (optional)

To use FLUX + inpainting with your image only as 000 (instead of SEVA):

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --scaffold_method scaffold_gen
```

### Two-images scaffold (0° + 180°)

Requires 180° counterpart images. Generate them first with `generate_180_views.py`, then:

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --scaffold_method two_images
# Or specify 180° directory explicitly:
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --scaffold_method two_images \
    --views_180_dir data/curated_set/views_180
```

### Continue on error

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --continue_on_error
```

### Check checkpoints before running

```bash
python run_batch_pipeline.py \
    --dataset_dir data/curated_set \
    --check_checkpoints
```

## Command-line arguments

### Required

- **`--dataset_dir`**: Path to the curated_set directory (must contain `metadata.json`).

### Output and filtering

- **`--output_base`** (default: `output/batch`): Base output directory. Results go under `output_base/photorealistic/index_XXXX/` and `output_base/stylized/index_XXXX/` (and `output_base/default/` when using the fallback).
- **`--indices`**: Only process these sample indices. Each item can be an integer or an inclusive range (`start-end` or `start:end`), e.g. `--indices 0 5 14`, `--indices 0-24`, `--indices 0-9 20 30-32`. Use ranges to split work across parallel runs.
- **`--only_stages`**: Run only the given stages, e.g. `--only_stages 1 2` or `--only_stages 3`. Omit to run 1, 2, and 3.

### Stage 1: Scaffold

- **`--scaffold_method`** (default: `single_image`):  
  - **`single_image`**: SEVA from one image (default; context-preserving).  
  - **`single_image_hybrid`**: SEVA for views 0,2,4,6 + inpainting for 1,3,5,7 (often better quality).  
  - **`scaffold_gen`**: Text prompts + optional image as 000.  
  - **`two_images`**: SEVA img2trajvid with 0° + 180° guidance images and defined poses; run `generate_180_views.py` first and set **`--views_180_dir`** (or use default `dataset_dir/views_180`).
- **`--views_180_dir`**: Directory with 180° images for **two_images** scaffold. Expects `<views_180_dir>/<variant>/index_XXXX_180.png`. Default: `<dataset_dir>/views_180`.
- **`--scaffold_mode`** (default: `manual`): For scaffold_gen only: `fast`, `automatic`, or `manual`.
- **`--default_theme`**: Default theme when not set in metadata (scaffold_gen).
- **`--traj_prior`** (default: `pan-in-place`): For single_image: SEVA trajectory. **`pan-in-place`** = rotate view only (avoids wall collision). **`orbit`** = move camera around scene.
- **`--num_targets`** (default: `80`): For single_image: number of frames.
- **`--cfg`** (default: `4.0,2.0`): For single_image: CFG scale.
- **`--output_all_seva_frames`**: Copy all SEVA-generated frames to `output_base/.../index_XXXX/all_seva_frames/` (000.png, 001.png, …) so you can inspect the full trajectory and verify the trajectory prior is followed.
- **`--num_seva_frames`** (default: 80): Number of frames for SEVA to generate. Use a value divisible by 8 (e.g. 40 or 80) so that the 8 scaffold keyframes are evenly spaced. Default 80.

### Stage 2: Trajectory + VGGT

- **`--translation_scaling_factor`**: Overridden by `expansion.translation_scaling_factor` in metadata when set.
- **`--trajectory_order`**: e.g. `in left right up`.
- **`--num_images_for_vggt`** (default: `40`).
- **`--root_dir`**: Directory containing predefined trajectories (default from scene_expansion).

### Stage 3: 3DGS

- **`--nerf_folder`**: Base output dir for nerfstudio (default: `./nerfstudio_output`).

### General

- **`--continue_on_error`**: Do not stop on first failure.
- **`--check_checkpoints`**: Run checkpoint check before processing.

## Output structure

```
output_base/
├── photorealistic/
│   ├── index_0000/
│   │   ├── scaffold/
│   │   │   ├── generated_views/   # single_image: SEVA output
│   │   │   ├── scaffold/
│   │   │   └── final/             # 000.png–007.png
│   │   └── scenes/
│   │       └── <scene_id>/
│   │           └── img2trajvid/
│   ├── index_0001/
│   └── ...
├── stylized/
│   ├── index_0000/
│   └── ...
├── default/                       # when using fallback image only
│   └── ...
├── batch_summary.json
└── failed_indices_YYYYMMDD_HHMMSS.json   # written when any job fails
```

### batch_summary.json

Written to `output_base/batch_summary.json`. Contains:

- **total**, **success**, **failed**: Job counts.
- **elapsed_seconds**, **only_stages**, **scaffold_method**.
- **results**: List of per-job entries with `index`, `variant`, `status`, `stages_completed`, `scaffold_path`, `scene_path`, `export_path`.

### failed_indices_&lt;timestamp&gt;.json

Written to `output_base/failed_indices_YYYYMMDD_HHMMSS.json` when any job fails. Contains **run_timestamp**, **total_failed_jobs**, **failed_indices** (unique indices), and **failed_entries** (per-job details including variant and optional error message). Useful for re-runs with `--indices` or debugging.

## Tips

1. Start with `--indices 0` or a few indices to verify paths and stages.
2. Use `--only_stages 1` to generate scaffolds first, then run 2 and 3.
3. Use `--continue_on_error` for large batches.
4. Default **single_image** keeps the full scaffold consistent with the input image; **single_image_hybrid** often improves quality; use **scaffold_gen** for text-guided variation; use **two_images** when you have 0° and 180° views (run `generate_180_views.py` first).

## Troubleshooting

- **No jobs to run**: Ensure each sample has at least one variant with **`filename`** and that the file exists at `dataset_dir/<variant>/<filename>` (e.g. `photorealistic/scene0.png`). Alternatively, `scaffold.input_image` or files under `dataset_dir/images/` can be used when running with a single fallback job.
- **Dataset directory not found**: Check `--dataset_dir` and that it contains `metadata.json`.
- **Stage 3 requires Stage 2**: Run Stage 2 first so that `output_base/<variant>/index_XXXX/scenes/<scene_id>/` exists with `img2trajvid/`.
- **Out of memory**: Reduce `--num_images_for_vggt` or process fewer samples with `--indices`.
- **3DGS/NeRF failures**: Check CUDA, `CUDA_HOME`, and gsplat/nerfstudio installation; see main README and scene_expansion logs.

## See also

- **worldexplorer.py** – Single-scene CLI (generate, expand, scaffold).
- **single_image_to_3d.py** – Single image → 3D (SEVA + key frames + optional expansion).
- **README.md** – Project overview and setup.
