# Conversation Summary

Summary of the discussion and changes made in this session.

---

## 1. Creation of `run_batch_pipeline.py`

You asked for a batch script that:

- Processes each sample from **`data/curated_set`**
- Uses functions from **`single_image_to_3d.py`** and the workflow in **`worldexplorer.py`**
- Generates scenes from **prompts in the dataset JSON** and **images as index 0** for the scaffold
- Supports **stage-wise execution**

We added **`run_batch_pipeline.py`** with:

- **Stage 1 – Scaffold**: Two methods — **scaffold_gen** (WorldExplorer text prompts + optional image as 000) or **single_image** (SEVA from one image → key frames → final scaffold).
- **Stage 2 – Scene expansion**: At first this ran the full expansion (trajectory + VGGT + 3DGS). Later we split out 3DGS (see below).
- **Stage 3 – 3DGS**: Added as a separate stage (see below).
- CLI: `--dataset_dir`, `--output_base`, `--only_stages`, `--scaffold_method`, plus stage-specific options.

---

## 2. What the stages mean / 3DGS as a separate stage

You asked what the stages mean and whether **3DGS optimization is included in Stage 2**.

We clarified:

- **Stage 1**: Build the scaffold (8 images 000–007).
- **Stage 2** (at that time): Full scene expansion: trajectory generation, VGGT, **and** 3DGS training + PLY export — so yes, 3DGS was inside Stage 2.

You then asked to **make 3DGS a separate stage**. We did:

- **`model/scene_expansion.py`**  
  - `TrajectoryTransformer.run(skip_3dgs=False)`: when `skip_3dgs=True`, it stops after trajectory + VGGT.  
  - `run_scene_expansion(..., skip_3dgs=False)`: passes `skip_3dgs` through.  
  - **`run_3dgs_only(work_dir, scene_id=None, nerf_folder=None)`**: runs only 3DGS training and PLY export on an existing work dir that already has `img2trajvid/` (Stage 2 output).

- **`run_batch_pipeline.py`**  
  - **Stage 2**: Calls `run_scene_expansion(..., skip_3dgs=True)` → trajectory + VGGT only.  
  - **Stage 3**: Calls `run_3dgs_only(work_dir)` for each sample’s work dir.  
  - `--only_stages` can be `1`, `2`, `3`, or any combination.  
  - When running only Stage 3, the script discovers the work dir from `output_base/<variant>/index_XXXX/scenes/`.

---

## 3. Photorealistic and stylized output structure

You asked for:

- Separate handling of **photorealistic** and **stylized** samples
- Output layout: **`output/batch/{photorealistic or stylized}/index_xxxx`**

We implemented:

- **`get_variants_for_sample(sample, dataset_dir)`**: Returns a list of `(variant_name, image_path)` for each variant that has an existing image, using **photorealistic** and **stylized** (e.g. `original_path` or `path`).
- Main loop: For each sample, for each variant, we run the pipeline with **`output_base = output_base / variant_name`** and **`input_image_path = variant_image_path`**.
- Output is written under **`output_base/photorealistic/index_XXXX/`** and **`output_base/stylized/index_XXXX/`** (and **`output_base/default/`** when using the fallback).

---

## 4. “No jobs to run” error

You ran the batch and got:

`ERROR: No jobs to run. Each sample must have at least one of photorealistic.original_path or stylized.original_path pointing to an existing file.`

We fixed it by:

- **Fallback in `get_variants_for_sample`**: If a sample has no photorealistic/stylized paths, we call **`_resolve_input_image(sample, dataset_dir)`** (same logic as elsewhere: `scaffold.input_image`, `images/index_XXXX.png`, etc.). If an image is found, we add one job with variant **`"default"`** so the pipeline still runs.
- **Variant path keys**: Supporting **`path`** in addition to **`original_path`** for `photorealistic` and `stylized`.
- **Error message**: Clarifying what the script looks for (variant paths, scaffold image, or `images/` convention) when no jobs can be created.

---

## 5. Scaffold generation process and parameters (e.g. “manual”)

You asked for an explanation of **how scaffold generation works** and what parameters like **manual** mean.

We summarized:

- **Scaffold** = 8 images (000–007) forming a 360° panorama: 4 cardinal views (000, 002, 004, 006) plus 4 inpainted “in-between” views (001, 003, 005, 007).
- **Flow**: (1) Generate or supply 000, 002, 004, 006 from text and/or one input image. (2) Depth/point clouds. (3) Inpaint 001, 003, 005, 007. (4) Choose which inpainting results go into `final/`.
- **Modes**:
  - **fast**: One inpainting per gap → copy directly to `final/`. Fastest; no selection.
  - **automatic**: Many inpainting variants per gap → **CLIP** scores each against neighboring prompts → best copy to `final/`.
  - **manual**: Many variants generated → **user** copies chosen 001, 003, 005, 007 into `final/` (script only copies 000, 002, 004, 006).

Other parameters (theme, custom prompts, `input_image_path`, `parent_folder`) were briefly explained in that discussion.

---

## 6. Default scaffold method: single_image (SEVA)

You reported that **later scaffold images were not guided by the input image** and felt out of context. You wanted the **image-based** method from **`single_image_to_3d.py`** as the default.

We set:

- **Default `--scaffold_method`** to **`single_image`** (was `scaffold_gen`).
- With **single_image**, the pipeline uses SEVA to generate views from **one** input image, then extracts key frames and builds the final scaffold — so all 8 images are derived from that image and stay in context.
- **`scaffold_gen`** remains available for text-guided generation (FLUX + inpainting) with the image only as 000; you can use it explicitly with `--scaffold_method scaffold_gen`.

Docstring and help text in `run_batch_pipeline.py` were updated to state that the default is image-based and context-preserving.

---

## 7. BATCH_PIPELINE_USAGE.md rewrite

You asked to **replace the content of BATCH_PIPELINE_USAGE.md** with instructions for the **latest** `run_batch_pipeline.py`.

We rewrote **BATCH_PIPELINE_USAGE.md** to document:

- Overview: batch over curated_set, photorealistic/stylized variants, three stages.
- Prerequisites and dataset layout.
- **metadata.json** format: `samples`, variant paths (`photorealistic` / `stylized` with `original_path` or `path`), fallback behavior, `scaffold` and `expansion` fields.
- **Stage definitions**: 1 = Scaffold (default single_image), 2 = Trajectory + VGGT, 3 = 3DGS.
- **Basic usage**: all stages, only Stage 1/2/3, `--indices`, `--scaffold_method scaffold_gen`, `--continue_on_error`, `--check_checkpoints`.
- **CLI arguments**: required, output/filtering, Stage 1 (scaffold method and SEVA/scaffold_gen options), Stage 2, Stage 3, general.
- **Output structure**: `output_base/photorealistic/`, `stylized/`, `default/`, and `batch_summary.json`.
- **Tips and troubleshooting** (no jobs, missing Stage 2, OOM, 3DGS).
- **See also**: worldexplorer.py, single_image_to_3d.py, README.md.

---

## Files touched in this conversation

| File | Changes |
|------|--------|
| **run_batch_pipeline.py** | New script; stages 1–3; variants (photorealistic/stylized/default); default single_image; fallback and path keys. |
| **model/scene_expansion.py** | `skip_3dgs` in `run()` and `run_scene_expansion()`; new `run_3dgs_only()`. |
| **BATCH_PIPELINE_USAGE.md** | Fully replaced with current batch pipeline usage. |
| **CONVERSATION_SUMMARY.md** | This summary. |

---

## Quick reference: current batch pipeline

- **Command**: `python run_batch_pipeline.py --dataset_dir data/curated_set [options]`
- **Stages**: 1 = Scaffold, 2 = Trajectory + VGGT, 3 = 3DGS (`--only_stages 1`, `2`, `3`, or omit for all).
- **Default scaffold**: **single_image** (SEVA from one image).
- **Output**: `output_base/photorealistic/index_XXXX/` and `output_base/stylized/index_XXXX/` (and `default/` when using fallback).
- **Metadata**: `samples[]` with `photorealistic` / `stylized` (e.g. `original_path` or `path`); fallback uses other resolvable image paths.
