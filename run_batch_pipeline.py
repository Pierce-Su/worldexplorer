#!/usr/bin/env python3
"""
Batch pipeline: Run the WorldExplorer pipeline on a whole dataset from curated_set.

Reads metadata.json to get scenes, prompts, and configurations, then processes each scene
through the pipeline stages. Supports optional stage filtering.

Pipeline stages:
    Stage 1: Scaffold Generation - Creates panoramic images from text
    Stage 2: Video Generation & Processing - Generates videos, converts transforms, processes point clouds
    Stage 3: NeRF Training & Export - Trains Gaussian Splatting model and exports to PLY

Output structure:
    output_base/
        index_0000/
            scaffold/
            scenes/
        index_0005/
            ...

Usage:
    python run_batch_pipeline.py --dataset_dir data/curated_set --output_base output/batch
    python run_batch_pipeline.py --dataset_dir data/curated_set --only_stages 1 2  # Only run stages 1 and 2
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil
import os

from model.scaffold_generation import GenerationMode, run_scaffold_generation
from model.scene_expansion import run_scene_expansion


def enhance_prompt_with_metadata(
    base_prompt: Optional[str],
    scene_type: Optional[str] = None,
    category: Optional[str] = None,
) -> Optional[str]:
    """Enhance prompt with scene_type and category metadata."""
    if not base_prompt:
        return None
    
    # Build metadata prefix
    metadata_parts = []
    
    # Add scene type and category context
    if scene_type and scene_type != "unknown":
        scene_formatted = scene_type.replace("_", " ").title()
        metadata_parts.append(f"Scene type: {scene_formatted}")
    
    if category and category != "unknown":
        category_formatted = category.replace("_", " ").title()
        metadata_parts.append(f"Category: {category_formatted}")
    
    # Combine metadata with base prompt
    if metadata_parts:
        metadata_prefix = ", ".join(metadata_parts)
        enhanced_prompt = f"{metadata_prefix}. {base_prompt}"
        return enhanced_prompt
    else:
        return base_prompt


def process_scaffold_stage(
    sample: dict,
    dataset_dir: Path,
    output_base: Path,
    common_args: dict,
) -> Optional[str]:
    """Process Stage 1: Scaffold Generation."""
    index = sample["index"]
    scaffold_config = sample.get("scaffold", {})
    
    # Check if scaffold_path is provided (pre-generated scaffold)
    if "scaffold_path" in scaffold_config:
        scaffold_path = dataset_dir / scaffold_config["scaffold_path"]
        if scaffold_path.exists():
            # Copy to output location
            output_scaffold_dir = output_base / f"index_{index:04d}" / "scaffold"
            output_scaffold_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy scaffold folder
            final_folder = output_scaffold_dir / "final"
            if scaffold_path.is_dir():
                if (scaffold_path / "final").exists():
                    shutil.copytree(scaffold_path / "final", final_folder, dirs_exist_ok=True)
                else:
                    # Assume scaffold_path points directly to final folder
                    shutil.copytree(scaffold_path, final_folder, dirs_exist_ok=True)
            else:
                raise ValueError(f"Scaffold path {scaffold_path} is not a directory")
            
            print(f"  Using pre-generated scaffold from: {scaffold_path}")
            return str(final_folder)
        else:
            print(f"  WARNING: Scaffold path not found: {scaffold_path}")
            return None
    
    # Generate scaffold
    mode_str = scaffold_config.get("mode", common_args.get("scaffold_mode", "manual"))
    try:
        mode = GenerationMode[mode_str]
    except KeyError:
        print(f"  WARNING: Invalid mode '{mode_str}', using 'manual'")
        mode = GenerationMode.manual
    
    custom = scaffold_config.get("custom", False)
    theme = sample.get("theme", common_args.get("default_theme"))
    
    # Fallback: use content as theme if theme is not provided
    if not theme and not custom:
        theme = sample.get("content")
        if theme:
            print(f"  Using 'content' field as theme: {theme}")
    
    # Prepare prompts
    prompts = scaffold_config.get("prompts", [])
    
    # Enhance prompts with metadata
    scene_type = sample.get("scene_type")
    category = sample.get("category")
    if prompts:
        enhanced_prompts = [
            enhance_prompt_with_metadata(p, scene_type, category) or p
            for p in prompts
        ]
    else:
        enhanced_prompts = None
    
    # Create output directory for scaffold
    output_scaffold_dir = output_base / f"index_{index:04d}" / "scaffold"
    output_scaffold_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if custom:
            if not enhanced_prompts or len(enhanced_prompts) != 4:
                print(f"  ERROR: Custom mode requires exactly 4 prompts")
                return None
            
            parent_folder, output_folder, final_folder = run_scaffold_generation(
                theme="custom",
                mode=mode,
                parent_folder=str(output_scaffold_dir / "panoramas" / f"scene_{index:04d}"),
                custom=True,
                custom_prompts=enhanced_prompts
            )
        else:
            if not theme:
                print(f"  ERROR: Theme required for non-custom scaffold generation")
                return None
            
            parent_folder, output_folder, final_folder = run_scaffold_generation(
                theme=theme,
                mode=mode,
                parent_folder=str(output_scaffold_dir / "panoramas" / f"scene_{index:04d}"),
                custom=False,
                custom_prompts=None
            )
        
        print(f"  ✓ Scaffold generated: {final_folder}")
        return final_folder
    
    except Exception as e:
        print(f"  ERROR processing scaffold: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_expansion_stages(
    sample: dict,
    scaffold_path: str,
    output_base: Path,
    only_stages: Optional[List[int]],
    common_args: dict,
) -> Optional[str]:
    """Process Stages 2-3: Scene Expansion."""
    index = sample["index"]
    expansion_config = sample.get("expansion", {})
    
    # Check if we should skip expansion stages
    if only_stages is not None:
        expansion_stages = [2, 3]
        if not any(s in only_stages for s in expansion_stages):
            print(f"  Skipping expansion stages (not in --only_stages)")
            return None
    
    # Get expansion parameters
    translation_scaling_factor = expansion_config.get(
        "translation_scaling_factor",
        common_args.get("translation_scaling_factor", 3.0)
    )
    
    # Determine scene type for default translation scaling
    scene_type = sample.get("scene_type", "indoor")
    if translation_scaling_factor is None:
        translation_scaling_factor = 10.0 if scene_type == "outdoor" else 3.0
    
    trajectory_order = expansion_config.get(
        "trajectory_order",
        common_args.get("trajectory_order")
    )
    
    num_images_for_vggt = expansion_config.get(
        "num_images_for_vggt",
        common_args.get("num_images_for_vggt", 40)
    )
    
    root_dir = common_args.get("root_dir")
    
    # Create output directory for scenes
    output_scenes_dir = output_base / f"index_{index:04d}" / "scenes"
    
    try:
        scene_work_dir = run_scene_expansion(
            scaffold_path,
            translation_scaling_factor=translation_scaling_factor,
            root_dir=root_dir,
            trajectory_order=trajectory_order,
            num_images_for_vggt=num_images_for_vggt
        )
        
        # Move scene to output location if needed
        if not str(scene_work_dir).startswith(str(output_scenes_dir)):
            # Scene was created in default location, move it
            output_scenes_dir.mkdir(parents=True, exist_ok=True)
            scene_name = os.path.basename(scene_work_dir)
            target_dir = output_scenes_dir / scene_name
            if os.path.exists(scene_work_dir):
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(scene_work_dir, target_dir)
                scene_work_dir = str(target_dir)
        
        print(f"  ✓ Scene expansion completed: {scene_work_dir}")
        return scene_work_dir
    
    except Exception as e:
        print(f"  ERROR processing scene expansion: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_sample(
    sample: dict,
    dataset_dir: Path,
    output_base: Path,
    only_stages: Optional[List[int]],
    common_args: dict,
) -> dict:
    """Process a single sample from metadata."""
    index = sample["index"]
    scene_type = sample.get("scene_type", "unknown")
    category = sample.get("category", "unknown")
    
    results = {
        "index": index,
        "scene_type": scene_type,
        "category": category,
        "stages_completed": [],
        "scaffold_path": None,
        "scene_path": None,
        "status": "pending",
    }
    
    # Stage 1: Scaffold Generation
    if only_stages is None or 1 in only_stages:
        print(f"\n  Stage 1: Scaffold Generation")
        scaffold_path = process_scaffold_stage(
            sample=sample,
            dataset_dir=dataset_dir,
            output_base=output_base,
            common_args=common_args,
        )
        
        if scaffold_path:
            results["scaffold_path"] = scaffold_path
            results["stages_completed"].append(1)
        else:
            results["status"] = "failed"
            return results
    else:
        # Check if scaffold already exists
        scaffold_dir = output_base / f"index_{index:04d}" / "scaffold" / "final"
        if scaffold_dir.exists():
            scaffold_path = str(scaffold_dir)
            results["scaffold_path"] = scaffold_path
            print(f"  Using existing scaffold: {scaffold_path}")
        else:
            print(f"  ERROR: Stage 1 skipped but no scaffold found")
            results["status"] = "failed"
            return results
    
    # Stages 2-3: Scene Expansion
    if only_stages is None or any(s in only_stages for s in [2, 3]):
        print(f"\n  Stages 2-3: Scene Expansion")
        scene_path = process_expansion_stages(
            sample=sample,
            scaffold_path=scaffold_path,
            output_base=output_base,
            only_stages=only_stages,
            common_args=common_args,
        )
        
        if scene_path:
            results["scene_path"] = scene_path
            # Determine which expansion stages were completed
            # This is a simplification - in practice, you'd check for specific outputs
            if only_stages:
                expansion_stages = [s for s in only_stages if s in [2, 3]]
                results["stages_completed"].extend(expansion_stages)
            else:
                results["stages_completed"].extend([2, 3])
        else:
            if only_stages and any(s in only_stages for s in [2, 3]):
                results["status"] = "failed"
                return results
    
    results["status"] = "success"
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run WorldExplorer pipeline on a batch dataset from curated_set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all scenes in the dataset
  python run_batch_pipeline.py --dataset_dir data/curated_set --output_base output/batch

  # Only run stages 1 and 2 (skip NeRF training & export)
  python run_batch_pipeline.py --dataset_dir data/curated_set --only_stages 1 2

  # Only run stage 1 (scaffold generation)
  python run_batch_pipeline.py --dataset_dir data/curated_set --only_stages 1

  # Process specific indices
  python run_batch_pipeline.py --dataset_dir data/curated_set --indices 0 5 14

  # Custom translation scaling and trajectory order
  python run_batch_pipeline.py --dataset_dir data/curated_set \\
      --translation_scaling_factor 10.0 --trajectory_order in left right up
        """,
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to curated_set directory containing metadata.json and optional scaffold folders",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="output/batch",
        help="Base output directory. Output structure: output_base/index_XXXX/",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample indices to process (default: all)",
    )
    
    # Stage filtering
    parser.add_argument(
        "--only_stages",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        default=None,
        help="Only run these stages (e.g., --only_stages 1 2). If not specified, runs all stages.",
    )
    
    # Stage 1: Scaffold params
    g1 = parser.add_argument_group("Stage 1: Scaffold Generation")
    g1.add_argument("--scaffold_mode", type=str, default="manual", choices=["fast", "automatic", "manual"],
                    help="Scaffold generation mode")
    g1.add_argument("--default_theme", type=str, default=None,
                    help="Default theme if not specified in metadata")
    
    # Stage 2-3: Expansion params
    g2 = parser.add_argument_group("Stages 2-3: Scene Expansion")
    g2.add_argument("--translation_scaling_factor", type=float, default=None,
                    help="Default translation scaling factor (overridden by metadata)")
    g2.add_argument("--trajectory_order", type=str, nargs="+", default=None,
                    help="Default trajectory order (overridden by metadata)")
    g2.add_argument("--num_images_for_vggt", type=int, default=40,
                    help="Number of images for VGGT processing")
    g2.add_argument("--root_dir", type=str, default=None,
                    help="Directory containing predefined trajectories")
    
    # General
    g3 = parser.add_argument_group("General")
    g3.add_argument("--device", type=str, default="cuda:0",
                    help="CUDA device for processing")
    g3.add_argument("--continue_on_error", action="store_true",
                    help="Continue processing other scenes if one fails")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found: {metadata_path}")
        sys.exit(1)
    
    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    samples = metadata.get("samples", [])
    total_samples = len(samples)
    
    if args.indices is not None:
        # Filter samples by indices
        indices_set = set(args.indices)
        samples = [s for s in samples if s["index"] in indices_set]
        print(f"Filtered to {len(samples)} samples matching indices: {args.indices}")
    
    if not samples:
        print("ERROR: No samples to process")
        sys.exit(1)
    
    # Prepare common arguments
    common_args = {
        "scaffold_mode": args.scaffold_mode,
        "default_theme": args.default_theme,
        "translation_scaling_factor": args.translation_scaling_factor,
        "trajectory_order": args.trajectory_order,
        "num_images_for_vggt": args.num_images_for_vggt,
        "root_dir": args.root_dir,
        "device": args.device,
    }
    
    # Create output directory
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"\n{'='*80}")
    print("Batch Pipeline Configuration")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output base: {output_base}")
    print(f"Total samples: {len(samples)}")
    if args.only_stages:
        print(f"Only running stages: {args.only_stages}")
    else:
        print("Running all stages: 1 (Scaffold), 2 (Video & Processing), 3 (Training & Export)")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Process each sample
    all_results = []
    start_time = time.time()
    
    for i, sample in enumerate(samples, 1):
        index = sample["index"]
        print(f"\n{'='*80}")
        print(f"Processing sample {i}/{len(samples)}: Index {index}")
        print(f"{'='*80}")
        
        try:
            result = process_sample(
                sample=sample,
                dataset_dir=dataset_dir,
                output_base=output_base,
                only_stages=args.only_stages,
                common_args=common_args,
            )
            all_results.append(result)
            
            if result["status"] == "success":
                print(f"✓ Successfully processed sample {index}")
                print(f"  Stages completed: {result['stages_completed']}")
            else:
                print(f"✗ Failed to process sample {index}")
                if not args.continue_on_error:
                    print("Stopping due to error (use --continue_on_error to continue)")
                    break
        
        except Exception as e:
            print(f"ERROR processing sample {index}: {e}")
            import traceback
            traceback.print_exc()
            if not args.continue_on_error:
                print("Stopping due to error (use --continue_on_error to continue)")
                break
    
    # Summary
    elapsed = time.time() - start_time
    total_success = sum(1 for r in all_results if r["status"] == "success")
    total_failed = sum(1 for r in all_results if r["status"] == "failed")
    
    print(f"\n{'='*80}")
    print("Batch Processing Summary")
    print(f"{'='*80}")
    print(f"Total samples processed: {len(all_results)}")
    print(f"Successfully processed: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
    print(f"{'='*80}\n")
    
    # Save results summary
    summary_path = output_base / "batch_summary.json"
    summary = {
        "total_samples": len(all_results),
        "total_processed": total_success,
        "total_failed": total_failed,
        "elapsed_time_seconds": elapsed,
        "configuration": {
            "only_stages": args.only_stages,
            "device": args.device,
            **common_args,
        },
        "results": all_results,
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
