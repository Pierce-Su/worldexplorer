#!/usr/bin/env python3
"""
Batch pipeline: Process each sample from data/curated_set using functions from
single_image_to_3d.py and the workflow in worldexplorer.py.

Stage definitions:
  Stage 1 — Scaffold: Produce 8 panorama/keyframe images (000.png–007.png). Output:
    output_base/index_XXXX/scaffold/final/.

  Stage 2 — Trajectory + VGGT: From 8 images, run Stable Virtual Camera (trajectory/video
    generation), transform conversion, and VGGT alignment (pose/depth). Output:
    output_base/index_XXXX/scenes/<scene_id>/ with img2trajvid/ (ready for 3DGS).

  Stage 3 — 3DGS: Train NeRFstudio splatfacto-big (Gaussian Splatting) and export to
    .ply (splat.ply, splat_rotated.ply). Requires Stage 2 output. Export under
    nerfstudio_output/ and optionally referenced in result["export_path"].

- Default scaffold method is single_image (SEVA): views are generated from one input image.
  single_image_hybrid: SEVA for views 0,2,4,6 + inpainting for 1,3,5,7 (often better quality).
  scaffold_gen: text prompts + image as 000.
- Each sample can have photorealistic and/or stylized image paths; a full scene is generated
  for each variant. Output: output_base/photorealistic/index_XXXX/ and
  output_base/stylized/index_XXXX/.
- Uses the variant image as scaffold index 0 (000.png).
- Stage-wise execution: --only_stages 1, 2, 3, or any combination.

Usage:
  python run_batch_pipeline.py --dataset_dir data/curated_set --output_base output/batch
  python run_batch_pipeline.py --dataset_dir data/curated_set --only_stages 1
  python run_batch_pipeline.py --dataset_dir data/curated_set --scaffold_method single_image
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _parse_indices(specs: List[str]) -> Set[int]:
    """Parse --indices specs into a set of integers. Each spec can be an int or inclusive range (e.g. 0-24 or 0:24)."""
    out: Set[int] = set()
    range_re = re.compile(r"^(-?\d+)[-:](-?\d+)$")
    for spec in specs:
        s = str(spec).strip()
        m = range_re.match(s)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo <= hi:
                out.update(range(lo, hi + 1))
            else:
                out.update(range(hi, lo + 1))
        else:
            out.add(int(s))
    return out


# Optional: worldexplorer checkpoint check (can be called before heavy stages)
def _check_checkpoints():
    """Optionally check for required checkpoints (mirrors worldexplorer)."""
    try:
        import importlib.util
        we_path = Path(__file__).resolve().parent / "worldexplorer.py"
        spec = importlib.util.spec_from_file_location("worldexplorer_script", we_path)
        we = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(we)
        we.check_and_download_checkpoints()
    except Exception as e:
        print(f"Checkpoint check skipped: {e}")


def _resolve_input_image(sample: dict, dataset_dir: Path) -> Optional[Path]:
    """Resolve the dataset image path to use as scaffold index 0 (000.png)."""
    # Explicit scaffold image
    scaffold = sample.get("scaffold", {})
    for key in ("input_image", "imag_0", "image"):
        path_val = scaffold.get(key) or sample.get(key)
        if path_val:
            p = dataset_dir / path_val if not os.path.isabs(path_val) else Path(path_val)
            if p.exists():
                return p
    # Legacy-style variant paths (used by get_variants_for_sample; not returned as single here)
    for variant in ("photorealistic", "stylized"):
        if variant in sample:
            path_val = sample[variant].get("original_path")
            if path_val:
                p = dataset_dir / path_val if not os.path.isabs(path_val) else Path(path_val)
                if p.exists():
                    return p
    # Convention: images/{index}.png or images/index_XXXX.png
    index = sample.get("index", 0)
    for name in (f"{index}.png", f"index_{index:04d}.png", "0.png"):
        p = dataset_dir / "images" / name
        if p.exists():
            return p
    return None


def get_variants_for_sample(sample: dict, dataset_dir: Path) -> List[Tuple[str, Path]]:
    """
    Return (variant_name, image_path) for each variant that has an existing image.
    Output is written to output_base/{variant_name}/index_XXXX/.
    """
    out: List[Tuple[str, Path]] = []
    for variant in ("photorealistic", "stylized"):
        if variant not in sample:
            continue
        '''path_val = sample[variant].get("original_path")
        if not path_val:
            continue
        '''
        path_val = f"{variant}/{sample[variant]['filename']}"
        p = dataset_dir / path_val if not os.path.isabs(path_val) else Path(path_val)
        if p.exists():
            out.append((variant, p))
    return out


def _get_prompts(sample: dict, dataset_dir: Path) -> Optional[List[str]]:
    """Get 4 prompts for N/W/S/E from sample (scaffold.prompts or theme-based)."""
    scaffold = sample.get("scaffold", {})
    prompts = scaffold.get("prompts")
    if prompts and len(prompts) >= 4:
        return list(prompts)[:4]
    theme = sample.get("theme") or scaffold.get("theme")
    if theme:
        return [
            f"kitchen of a {theme}",
            f"office of a {theme}",
            f"bedroom of a {theme}",
            f"living room of a {theme}",
        ]
    return None


def _enhance_prompts(
    prompts: List[str],
    scene_type: Optional[str] = None,
    category: Optional[str] = None,
) -> List[str]:
    """Optionally prefix prompts with scene_type/category."""
    if not scene_type and not category:
        return prompts
    parts = []
    if scene_type and scene_type != "unknown":
        parts.append(f"Scene type: {scene_type.replace('_', ' ').title()}")
    if category and category != "unknown":
        parts.append(f"Category: {category.replace('_', ' ').title()}")
    if not parts:
        return prompts
    prefix = ", ".join(parts) + ". "
    return [prefix + p for p in prompts]


# ----- Stage 1: Scaffold -----

def _run_stage1_scaffold_gen(
    sample: dict,
    dataset_dir: Path,
    output_base: Path,
    common_args: dict,
    input_image_path: Optional[Path],
) -> Optional[str]:
    """Stage 1 using worldexplorer scaffold_generation (prompts + optional image as 000)."""
    from model.scaffold_generation import GenerationMode, run_scaffold_generation

    index = sample["index"]
    scaffold = sample.get("scaffold", {})

    # Pre-generated scaffold: copy to output and return final path
    if "scaffold_path" in scaffold:
        raw = scaffold["scaffold_path"]
        src = Path(raw) if os.path.isabs(raw) else dataset_dir / raw
        if src.exists():
            output_scaffold_dir = output_base / f"index_{index:04d}" / "scaffold"
            output_scaffold_dir.mkdir(parents=True, exist_ok=True)
            final_folder = output_scaffold_dir / "final"
            if (src / "final").exists():
                shutil.copytree(src / "final", final_folder, dirs_exist_ok=True)
            else:
                shutil.copytree(src, final_folder, dirs_exist_ok=True)
            print(f"  Using pre-generated scaffold: {src}")
            return str(final_folder)
        print(f"  WARNING: scaffold_path not found: {src}")
    mode_str = scaffold.get("mode", common_args.get("scaffold_mode", "manual"))
    try:
        mode = GenerationMode[mode_str]
    except KeyError:
        mode = GenerationMode.manual

    custom = scaffold.get("custom", True)  # Prefer custom when we have prompts
    prompts = _get_prompts(sample, dataset_dir)
    if custom and prompts and len(prompts) >= 4:
        enhanced = _enhance_prompts(
            prompts[:4],
            sample.get("scene_type"),
            sample.get("category"),
        )
    else:
        enhanced = None
        custom = False
        theme = sample.get("theme") or common_args.get("default_theme") or sample.get("content")
        if not theme:
            print("  ERROR: No theme and no 4 prompts for scaffold_gen")
            return None

    output_scaffold_dir = output_base / f"index_{index:04d}" / "scaffold"
    output_scaffold_dir.mkdir(parents=True, exist_ok=True)
    parent_folder = str(output_scaffold_dir / "panoramas" / f"scene_{index:04d}")

    try:
        if custom and enhanced:
            parent_folder, output_folder, final_folder = run_scaffold_generation(
                theme="custom",
                mode=mode,
                parent_folder=parent_folder,
                custom=True,
                custom_prompts=enhanced,
                input_image_path=str(input_image_path) if input_image_path else None,
            )
        else:
            parent_folder, output_folder, final_folder = run_scaffold_generation(
                theme=theme,
                mode=mode,
                parent_folder=parent_folder,
                custom=False,
                custom_prompts=None,
                input_image_path=str(input_image_path) if input_image_path else None,
            )
        return final_folder
    except Exception as e:
        print(f"  ERROR scaffold_gen: {e}")
        import traceback
        traceback.print_exc()
        return None


def _run_stage1_single_image(
    sample: dict,
    dataset_dir: Path,
    output_base: Path,
    common_args: dict,
    input_image_path: Optional[Path],
) -> Optional[str]:
    """Stage 1 using single_image_to_3d: SEVA views → key frames → final scaffold."""
    from single_image_to_3d import (
        generate_views_from_single_image,
        extract_key_frames_for_scaffold,
        prepare_final_scaffold,
    )

    index = sample["index"]
    if not input_image_path or not input_image_path.exists():
        print("  ERROR: single_image method requires an input image for this sample")
        return None

    output_scaffold_dir = output_base / f"index_{index:04d}" / "scaffold"
    output_scaffold_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_scaffold_dir / "generated_views"
    scaffold_dir = output_scaffold_dir / "scaffold"
    final_dir = output_scaffold_dir / "final"

    # Copy image with a unique name so SEVA output dir is unique per sample
    views_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"index_{index:04d}.png"
    staged_image = views_dir / unique_name
    shutil.copy2(str(input_image_path), str(staged_image))

    traj_prior = common_args.get("traj_prior", "orbit")
    num_targets = common_args.get("num_targets", 80)
    cfg = common_args.get("cfg", "4.0,2.0")
    translation_scaling_factor = sample.get("expansion", {}).get(
        "translation_scaling_factor",
        common_args.get("translation_scaling_factor", 3.0),
    )

    num_seva_frames = common_args.get("num_seva_frames", 80)
    try:
        generated_scene_dir = generate_views_from_single_image(
            str(staged_image),
            str(views_dir),
            traj_prior=traj_prior,
            num_targets=num_targets,
            cfg=cfg,
            translation_scaling_factor=translation_scaling_factor,
            num_seva_frames=num_seva_frames,
        )
        if common_args.get("output_all_seva_frames"):
            from single_image_to_3d import copy_all_seva_frames_to
            copy_all_seva_frames_to(str(generated_scene_dir), str(output_base / f"index_{index:04d}" / "all_seva_frames"))
        extract_key_frames_for_scaffold(str(generated_scene_dir), str(scaffold_dir))
        prepare_final_scaffold(str(scaffold_dir), str(final_dir))
        # Save prompts as metadata if present
        prompts = _get_prompts(sample, dataset_dir)
        if prompts:
            meta_path = output_scaffold_dir / "theme_info.txt"
            with open(meta_path, "w") as f:
                f.write("Prompts from dataset:\n\n")
                for i, p in enumerate(prompts[:4]):
                    f.write(f"  {i}: {p}\n")
        return str(final_dir)
    except Exception as e:
        print(f"  ERROR single_image: {e}")
        import traceback
        traceback.print_exc()
        return None


def _run_stage1_single_image_hybrid(
    sample: dict,
    dataset_dir: Path,
    output_base: Path,
    common_args: dict,
    input_image_path: Optional[Path],
) -> Optional[str]:
    """Stage 1 hybrid: SEVA for 000, 002, 004, 006; inpainting for 001, 003, 005, 007."""
    from single_image_to_3d import run_hybrid_seva_inpaint

    index = sample["index"]
    if not input_image_path or not input_image_path.exists():
        print("  ERROR: single_image_hybrid requires an input image for this sample")
        return None

    output_scaffold_dir = output_base / f"index_{index:04d}" / "scaffold"
    output_scaffold_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_scaffold_dir / "generated_views_hybrid"
    final_dir = output_scaffold_dir / "final"

    traj_prior = common_args.get("traj_prior", "pan-in-place")
    num_targets = common_args.get("num_targets", 80)
    cfg = common_args.get("cfg", "4.0,2.0")
    translation_scaling_factor = sample.get("expansion", {}).get(
        "translation_scaling_factor",
        common_args.get("translation_scaling_factor", 3.0),
    )
    num_seva_frames = common_args.get("num_seva_frames", 80)

    try:
        all_seva_dir = str(output_base / f"index_{index:04d}" / "all_seva_frames") if common_args.get("output_all_seva_frames") else None
        return run_hybrid_seva_inpaint(
            str(input_image_path),
            str(views_dir),
            traj_prior=traj_prior,
            num_targets=num_targets,
            cfg=cfg,
            translation_scaling_factor=translation_scaling_factor,
            keyframe_span_ratio=0.875,
            final_output_dir=str(final_dir),
            output_all_seva_frames_to=all_seva_dir,
            num_seva_frames=num_seva_frames,
        )
    except Exception as e:
        print(f"  ERROR single_image_hybrid: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_stage1(
    sample: dict,
    dataset_dir: Path,
    output_base: Path,
    common_args: dict,
    scaffold_method: str,
    input_image_path: Optional[Path] = None,
) -> Optional[str]:
    """Run Stage 1 (scaffold). Returns path to final folder (000–007.png) or None."""
    index = sample["index"]
    if input_image_path is None:
        input_image_path = _resolve_input_image(sample, dataset_dir)
    if input_image_path:
        print(f"  Using dataset image as scaffold index 0: {input_image_path}")
    elif scaffold_method in ("single_image", "single_image_hybrid"):
        print("  Skipping: no input image for single-image method")
        return None

    if scaffold_method == "scaffold_gen":
        return _run_stage1_scaffold_gen(
            sample, dataset_dir, output_base, common_args, input_image_path
        )
    if scaffold_method == "single_image":
        return _run_stage1_single_image(
            sample, dataset_dir, output_base, common_args, input_image_path
        )
    if scaffold_method == "single_image_hybrid":
        return _run_stage1_single_image_hybrid(
            sample, dataset_dir, output_base, common_args, input_image_path
        )
    print(f"  Unknown scaffold_method: {scaffold_method}")
    return None


# ----- Stage 2: Trajectory + VGGT (no 3DGS) -----

def run_stage2(
    sample: dict,
    scaffold_path: str,
    output_base: Path,
    common_args: dict,
) -> Optional[str]:
    """Run Stage 2 (trajectory generation + VGGT). Returns work_dir path or None."""
    from model.scene_expansion import run_scene_expansion

    index = sample["index"]
    expansion = sample.get("expansion", {})
    translation_scaling_factor = expansion.get(
        "translation_scaling_factor",
        common_args.get("translation_scaling_factor", 3.0),
    )
    if translation_scaling_factor is None:
        translation_scaling_factor = 10.0 if sample.get("scene_type") == "outdoor" else 3.0
    trajectory_order = expansion.get(
        "trajectory_order",
        common_args.get("trajectory_order"),
    )
    num_images_for_vggt = expansion.get(
        "num_images_for_vggt",
        common_args.get("num_images_for_vggt", 40),
    )
    root_dir = common_args.get("root_dir")

    output_scenes_dir = output_base / f"index_{index:04d}" / "scenes"
    try:
        scene_work_dir = run_scene_expansion(
            scaffold_path,
            translation_scaling_factor=translation_scaling_factor,
            root_dir=root_dir,
            trajectory_order=trajectory_order,
            num_images_for_vggt=num_images_for_vggt,
            skip_3dgs=True,
        )
        if scene_work_dir and not str(scene_work_dir).startswith(str(output_scenes_dir)):
            output_scenes_dir.mkdir(parents=True, exist_ok=True)
            scene_name = os.path.basename(scene_work_dir)
            target_dir = output_scenes_dir / scene_name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(scene_work_dir, target_dir)
            scene_work_dir = str(target_dir)
        return scene_work_dir
    except Exception as e:
        print(f"  ERROR Stage 2 (trajectory + VGGT): {e}")
        import traceback
        traceback.print_exc()
        return None


# ----- Stage 3: 3DGS training + export -----

def run_stage3(work_dir: str, common_args: dict) -> Optional[str]:
    """Run Stage 3 (3DGS training and PLY export). Returns export dir path or None."""
    from model.scene_expansion import run_3dgs_only

    nerf_folder = common_args.get("nerf_folder")
    try:
        export_path = run_3dgs_only(work_dir, scene_id=None, nerf_folder=nerf_folder)
        return export_path
    except Exception as e:
        print(f"  ERROR Stage 3 (3DGS): {e}")
        import traceback
        traceback.print_exc()
        return None


def _find_stage2_work_dir(output_base: Path, index: int) -> Optional[Path]:
    """Find existing Stage 2 work dir under output_base/index_XXXX/scenes/ (one subdir)."""
    scenes_dir = output_base / f"index_{index:04d}" / "scenes"
    if not scenes_dir.is_dir():
        return None
    subdirs = [p for p in scenes_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    # Prefer dir that contains img2trajvid
    for d in subdirs:
        if (d / "img2trajvid").is_dir():
            return d
    return subdirs[0]


# ----- Per-sample pipeline -----

def process_sample(
    sample: dict,
    dataset_dir: Path,
    output_base: Path,
    only_stages: Optional[List[int]],
    common_args: dict,
    scaffold_method: str,
    input_image_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Process one sample through requested stages. Returns result dict."""
    index = sample["index"]
    result = {
        "index": index,
        "scene_type": sample.get("scene_type", "unknown"),
        "category": sample.get("category", "unknown"),
        "stages_completed": [],
        "scaffold_path": None,
        "scene_path": None,
        "export_path": None,
        "status": "pending",
    }

    # Stage 1: Scaffold
    run_stage1_flag = only_stages is None or 1 in only_stages
    if run_stage1_flag:
        print(f"\n  Stage 1: Scaffold ({scaffold_method})")
        scaffold_path = run_stage1(
            sample, dataset_dir, output_base, common_args, scaffold_method,
            input_image_path=input_image_path,
        )
        if scaffold_path:
            result["scaffold_path"] = scaffold_path
            result["stages_completed"].append(1)
        else:
            existing = output_base / f"index_{index:04d}" / "scaffold" / "final"
            if existing.exists():
                result["scaffold_path"] = str(existing)
                print(f"  Using existing scaffold: {existing}")
            else:
                result["status"] = "failed"
                return result
    else:
        # Stage 1 skipped: need scaffold only if running Stage 2
        if only_stages is None or 2 in only_stages:
            existing = output_base / f"index_{index:04d}" / "scaffold" / "final"
            if not existing.exists():
                result["status"] = "failed"
                print("  ERROR: Stage 1 skipped but no scaffold/final found")
                return result
            result["scaffold_path"] = str(existing)
        # If only Stage 3, scaffold_path can stay None

    scaffold_path = result.get("scaffold_path")

    # Stage 2: Trajectory + VGGT
    run_stage2_flag = only_stages is None or 2 in only_stages
    if run_stage2_flag:
        print(f"\n  Stage 2: Trajectory + VGGT")
        scene_path = run_stage2(sample, scaffold_path, output_base, common_args)
        if scene_path:
            result["scene_path"] = scene_path
            result["stages_completed"].append(2)
        else:
            result["status"] = "failed"
            return result
    elif only_stages and 3 in only_stages:
        work_dir = _find_stage2_work_dir(output_base, index)
        if work_dir:
            result["scene_path"] = str(work_dir)
        else:
            result["status"] = "failed"
            print("  ERROR: Stage 3 requires Stage 2 output; no work_dir found under scenes/")
            return result

    # Stage 3: 3DGS
    run_stage3_flag = only_stages is None or 3 in only_stages
    if run_stage3_flag:
        work_dir = result.get("scene_path")
        if not work_dir:
            work_dir = _find_stage2_work_dir(output_base, index)
            if work_dir:
                result["scene_path"] = str(work_dir)
        if work_dir:
            print(f"\n  Stage 3: 3DGS")
            export_path = run_stage3(work_dir, common_args)
            if export_path:
                result["export_path"] = export_path
                result["stages_completed"].append(3)
            else:
                result["status"] = "failed"
                return result
        else:
            result["status"] = "failed"
            print("  ERROR: Stage 3 requires Stage 2 work_dir (scenes/<scene_id>)")
            return result

    result["status"] = "success"
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch pipeline for data/curated_set using single_image_to_3d and worldexplorer workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all samples (scaffold + expansion)
  python run_batch_pipeline.py --dataset_dir data/curated_set --output_base output/batch

  # Only Stage 1 (scaffold) using prompts + image as 000
  python run_batch_pipeline.py --dataset_dir data/curated_set --only_stages 1

  # Only Stage 2 (trajectory + VGGT from existing scaffold)
  python run_batch_pipeline.py --dataset_dir data/curated_set --only_stages 2

  # Only Stage 3 (3DGS from existing Stage 2 output)
  python run_batch_pipeline.py --dataset_dir data/curated_set --only_stages 3

  # Use single-image flow (SEVA) with dataset image as 000
  python run_batch_pipeline.py --dataset_dir data/curated_set --scaffold_method single_image

  # Specific indices
  python run_batch_pipeline.py --dataset_dir data/curated_set --indices 0 2 5

  # Range (for parallel workers: worker 1 does 0-24, worker 2 does 25-49, etc.)
  python run_batch_pipeline.py --dataset_dir data/curated_set --indices 0-24
  python run_batch_pipeline.py --dataset_dir data/curated_set --indices 25-49
  python run_batch_pipeline.py --dataset_dir data/curated_set --indices 0-9 20 30-32
        """,
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to curated_set (metadata.json + images)")
    parser.add_argument("--output_base", type=str, default="output/batch",
                        help="Base output directory. Output: output_base/photorealistic/index_XXXX/ and output_base/stylized/index_XXXX/")
    parser.add_argument("--indices", type=str, nargs="+", default=None,
                        help="Only process these sample indices. Each item can be an integer (e.g. 0, 5) or an inclusive range (e.g. 0-24 or 0:24). Examples: --indices 0 2 5, --indices 0-9, --indices 0-24 30-39")
    parser.add_argument("--only_stages", type=int, nargs="+", choices=[1, 2, 3], default=None,
                        help="Run only these stages (e.g. --only_stages 1 2 3)")
    parser.add_argument("--scaffold_method", type=str, default="single_image",
                        choices=["scaffold_gen", "single_image", "single_image_hybrid"],
                        help="single_image (default): SEVA only. single_image_hybrid: SEVA for 0,2,4,6 + inpainting for 1,3,5,7. scaffold_gen: text prompts + image as 000.")
    # Stage 1
    g1 = parser.add_argument_group("Stage 1: Scaffold")
    g1.add_argument("--scaffold_mode", type=str, default="manual",
                     choices=["fast", "automatic", "manual"])
    g1.add_argument("--default_theme", type=str, default=None)
    g1.add_argument("--traj_prior", type=str, default="pan-in-place",
                     help="For single_image: SEVA trajectory. pan-in-place (default) = rotate view only, no position change (avoids wall collision). orbit = move camera around scene.")
    g1.add_argument("--num_targets", type=int, default=80,
                     help="For single_image: number of frames")
    g1.add_argument("--cfg", type=str, default="4.0,2.0", help="For single_image: CFG scale")
    g1.add_argument("--output_all_seva_frames", action="store_true",
                    help="Copy all SEVA-generated frames to output_base/.../index_XXXX/all_seva_frames for trajectory inspection.")
    g1.add_argument("--num_seva_frames", type=int, default=80,
                    help="Number of frames for SEVA to generate (use divisible by 8, e.g. 40 or 80). Default 80.")
    # Stage 2
    g2 = parser.add_argument_group("Stage 2: Scene expansion")
    g2.add_argument("--translation_scaling_factor", type=float, default=None)
    g2.add_argument("--trajectory_order", type=str, nargs="+", default=None)
    g2.add_argument("--num_images_for_vggt", type=int, default=40)
    g2.add_argument("--root_dir", type=str, default=None)
    # Stage 3: 3DGS
    g3 = parser.add_argument_group("Stage 3: 3DGS")
    g3.add_argument("--nerf_folder", type=str, default=None,
                    help="Base output dir for nerfstudio (default: ./nerfstudio_output)")
    # General
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--check_checkpoints", action="store_true",
                        help="Run worldexplorer checkpoint check before processing")

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / "metadata.json"

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    samples = metadata.get("samples", [])
    if not samples:
        print("ERROR: No samples in metadata.json", file=sys.stderr)
        sys.exit(1)

    if args.indices is not None:
        idx_set = _parse_indices(args.indices)
        samples = [s for s in samples if s.get("index") in idx_set]
        print(f"Filtered to {len(samples)} samples (indices: {sorted(idx_set)})")
    if not samples:
        print("ERROR: No samples to process after filtering", file=sys.stderr)
        sys.exit(1)

    common_args = {
        "scaffold_mode": args.scaffold_mode,
        "default_theme": args.default_theme,
        "translation_scaling_factor": args.translation_scaling_factor,
        "trajectory_order": args.trajectory_order,
        "num_images_for_vggt": args.num_images_for_vggt,
        "root_dir": args.root_dir,
        "nerf_folder": args.nerf_folder,
        "traj_prior": args.traj_prior,
        "num_targets": args.num_targets,
        "cfg": args.cfg,
        "output_all_seva_frames": args.output_all_seva_frames,
        "num_seva_frames": args.num_seva_frames,
    }

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Build list of (sample, variant_name, variant_image_path) jobs
    jobs: List[Tuple[dict, str, Path]] = []
    for sample in samples:
        variants = get_variants_for_sample(sample, dataset_dir)
        for variant_name, variant_image_path in variants:
            jobs.append((sample, variant_name, variant_image_path))
    if not jobs:
        print("ERROR: No jobs to run. Each sample must have at least one of "
              "photorealistic.original_path or stylized.original_path pointing to an existing file.",
              file=sys.stderr)
        sys.exit(1)

    if args.check_checkpoints:
        _check_checkpoints()

    print(f"\n{'='*80}")
    print("Batch pipeline (curated_set)")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_base} (photorealistic/ and stylized/ per sample)")
    print(f"Samples: {len(samples)} → {len(jobs)} jobs (one per variant per sample)")
    print(f"Scaffold method: {args.scaffold_method}")
    print(f"Stages: {args.only_stages if args.only_stages else [1, 2, 3]}")
    print(f"{'='*80}\n")

    all_results = []
    start = time.time()
    for job_num, (sample, variant_name, variant_image_path) in enumerate(jobs, 1):
        index = sample.get("index", 0)
        variant_output_base = output_base / variant_name
        variant_output_base.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*80}")
        print(f"Job {job_num}/{len(jobs)}: index {index} [{variant_name}]")
        print(f"{'='*80}")
        try:
            res = process_sample(
                sample=sample,
                dataset_dir=dataset_dir,
                output_base=variant_output_base,
                only_stages=args.only_stages,
                common_args=common_args,
                scaffold_method=args.scaffold_method,
                input_image_path=variant_image_path,
            )
            res["variant"] = variant_name
            all_results.append(res)
            if res["status"] == "success":
                print(f"  ✓ Done. Stages: {res['stages_completed']}")
            else:
                print("  ✗ Failed")
                if not args.continue_on_error:
                    break
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "index": index,
                "variant": variant_name,
                "status": "failed",
                "stages_completed": [],
                "scaffold_path": None,
                "scene_path": None,
                "export_path": None,
            })
            if not args.continue_on_error:
                break

    elapsed = time.time() - start
    success = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "failed")

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Jobs: {len(all_results)} | Success: {success} | Failed: {failed}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"{'='*80}\n")

    summary_path = output_base / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total": len(all_results),
            "success": success,
            "failed": failed,
            "elapsed_seconds": elapsed,
            "only_stages": args.only_stages,
            "scaffold_method": args.scaffold_method,
            "results": all_results,
        }, f, indent=2)
    print(f"Summary written to {summary_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
