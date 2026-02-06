#!/usr/bin/env python
"""
Helper script to continue scene expansion from the training step.
Use this if video generation, transform conversion, and VGGT are already done.
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_cuda_env():
    """Set up CUDA environment variables for gsplat compilation.
    This fixes the 'cuda_runtime.h: No such file or directory' error
    when CUDA is installed via conda.
    """
    env = os.environ.copy()
    conda_env = os.environ.get("CONDA_PREFIX")
    if conda_env:
        env["CUDA_HOME"] = conda_env
        env["CUDA_PATH"] = conda_env
        # Add CUDA headers path for conda-installed CUDA
        cuda_include_path = os.path.join(conda_env, "targets", "x86_64-linux", "include")
        if os.path.exists(cuda_include_path):
            current_cplus_include = env.get("CPLUS_INCLUDE_PATH", "")
            if current_cplus_include:
                env["CPLUS_INCLUDE_PATH"] = f"{cuda_include_path}:{current_cplus_include}"
            else:
                env["CPLUS_INCLUDE_PATH"] = cuda_include_path
    return env

def continue_from_training(work_dir, scene_id):
    """Continue scene expansion from the training step."""
    
    print(f"\n{'='*80}")
    print("CONTINUING SCENE EXPANSION FROM TRAINING STEP")
    print(f"{'='*80}\n")
    
    # Check prerequisites
    img2trajvid_dir = os.path.join(work_dir, "img2trajvid")
    merged_transforms = os.path.join(img2trajvid_dir, "transforms.json")
    vggt_pcl = os.path.join(img2trajvid_dir, "vggt_pcl.ply")
    
    if not os.path.exists(merged_transforms):
        print(f"❌ Error: Transform conversion not found at {merged_transforms}")
        print("   Please run the full expansion pipeline first.")
        return 1
    
    if not os.path.exists(vggt_pcl):
        print(f"❌ Error: VGGT processing not found at {vggt_pcl}")
        print("   Please run the full expansion pipeline first.")
        return 1
    
    print(f"✅ Found transform conversion: {merged_transforms}")
    print(f"✅ Found VGGT point cloud: {vggt_pcl}")
    print(f"\n📁 Work directory: {work_dir}")
    print(f"🆔 Scene ID: {scene_id}\n")
    
    # Check if training already done
    from model.scene_expansion import NERF_FOLDER
    scene_root = os.path.join(NERF_FOLDER, scene_id, "splatfacto")
    training_done = False
    
    if os.path.exists(scene_root):
        runs = sorted([d for d in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, d))])
        if runs:
            latest_run = runs[-1]
            checkpoint_dir = os.path.join(scene_root, latest_run, "nerfstudio_models")
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
                if checkpoints:
                    print(f"✅ Training already completed (found {len(checkpoints)} checkpoint(s))")
                    print(f"   Latest run: {latest_run}")
                    training_done = True
                    config_path = os.path.join(scene_root, latest_run, "config.yml")
                else:
                    print(f"⚠️  Training directory exists but no checkpoints found")
                    print(f"   Will attempt to resume training...")
                    config_path = os.path.join(scene_root, latest_run, "config.yml")
                    if os.path.exists(config_path):
                        print(f"   Found config: {config_path}")
    
    if not training_done:
        print(f"\n🚀 Starting 3DGS training...")
        print(f"   This will take 1-4 hours depending on your GPU\n")
        
        # Set CUDA environment variables for gsplat compilation
        env = setup_cuda_env()
        
        '''
        command = [
            "ns-train", "splatfacto-big",
            "--data", img2trajvid_dir,
            "--output-dir", NERF_FOLDER,
            "--experiment-name", scene_id,
            # "--vis", "tensorboard",
            "--vis", "viewer_legacy",
            "--steps_per_eval_image", "5000",
            "--steps_per_eval_all_images", "1000000",
            "--pipeline.model.camera-optimizer.mode", "off",
            # "--pipeline.model.strategy", "mcmc",
            "--pipeline.model.strategy", "default",
            "--pipeline.model.rasterize_mode", "antialiased",
            "--pipeline.model.use_scale_regularization", "True",
            "--pipeline.model.max_gauss_ratio", "5.0",
            "--pipeline.model.cull_scale_thresh", "0.1",
            # "--pipeline.model.cull_screen_size", "0.1",
            "--pipeline.model.cull_screen_size", "0.05",
            # "--pipeline.model.max_screen_size", "0.05",  # additional parameter
            "--pipeline.model.use_bilateral_grid", "True", # additional parameter
            "--pipeline.model.color_corrected_metrics", "True", # additional parameter
            "--pipeline.model.refine_every", "100", # additional parameter
            "--pipeline.model.reset_alpha_every", "30", # additional parameter
            "--pipeline.model.warmup_length", "500", # additional parameter
            # "--pipeline.model.position_lr_init", "0.00016", # additional parameter
            "nerfstudio-data",
            "--eval_mode", "all",
        ]
        '''

        command = [
            "ns-train", "splatfacto", # Use base splatfacto; 'big' is more prone to exploding
            "--data", img2trajvid_dir,
            "--output-dir", NERF_FOLDER,
            "--experiment-name", scene_id,
            "--vis", "viewer_legacy",
            "--max-num-iterations", "300000",        # Total training time
            # "--pipeline.model.stop-split-at", "20000", # Stop making new points early
            # "--pipeline.model.stop-screen-size-at", "20000",
            "--pipeline.model.strategy", "default", # KEEP default, stay away from MCMC
            "--pipeline.model.sh-degree", "1",      # Lower this! Stops the B&W color explosion
            "--pipeline.model.use-bilateral-grid", "False", # Set to False to stop the AssertionError
            "--pipeline.model.camera-optimizer.mode", "off",
            
            # Aggressive Culling (This kills the "Blobs")
            "--pipeline.model.cull-alpha-thresh", "0.01", 
            "--pipeline.model.cull-scale-thresh", "0.5",
            "--pipeline.model.reset-alpha-every", "30",
            
            # Limit Gaussian growth
            "--pipeline.model.densify-grad-thresh", "0.0006",
            
            "nerfstudio-data",
            "--eval-mode", "all",
            "--load-3D-points", "True", # Force it to use your "plausible" point cloud
        ]
        
        result = subprocess.run(command, env=env)
        if result.returncode != 0:
            print(f"\n❌ Training failed (exit code {result.returncode})")
            print("\nCommon causes:")
            print("  1. gsplat CUDA extensions not compiled (need CUDA toolkit)")
            print("  2. GPU out of memory")
            print("  3. Missing dependencies")
            print("\nTo fix gsplat CUDA issue:")
            print("  1. Install CUDA toolkit: conda install -c conda-forge cuda-toolkit")
            print("  2. Or ask your admin to install CUDA toolkit system-wide")
            print("  3. Then reinstall gsplat: pip install --force-reinstall --no-cache-dir gsplat")
            return 1
    
    # Export step
    print(f"\n{'='*80}")
    print("EXPORTING 3DGS MODEL")
    print(f"{'='*80}\n")
    
    if not training_done:
        # Get the latest run after training
        runs = sorted([d for d in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, d))])
        if runs:
            latest_run = runs[-1]
            config_path = os.path.join(scene_root, latest_run, "config.yml")
        else:
            print("❌ Error: No training runs found")
            return 1
    else:
        config_path = os.path.join(scene_root, latest_run, "config.yml")
    
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found: {config_path}")
        return 1
    
    output_dir = os.path.join(scene_root, latest_run, "exports", "splat")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(output_dir, "splat.ply")):
        print(f"✅ Export already exists: {output_dir}/splat.ply")
        print("   Skipping export...")
    else:
        print(f"Exporting from: {config_path}")
        print(f"Output directory: {output_dir}\n")
        print("⚠️  Using custom export with lenient opacity threshold")
        print("   Default nerfstudio threshold (-5.5373) filters out 99%+ of Gaussians")
        print("   Using threshold -10.0 to keep many more Gaussians\n")
        
        # Set up CUDA environment for export (in case gsplat needs to compile)
        env = setup_cuda_env()
        env["NERFSTUDIO_DISABLE_TORCH_COMPILE"] = "1"
        
        # Use custom export script with lenient opacity threshold
        # The default nerfstudio threshold is -5.5373 (logit(1/255)), which is too strict
        # We use -10.0 (approximately logit(1/22026)) to keep many more Gaussians
        script_dir = os.path.dirname(os.path.abspath(__file__))
        custom_export_script = os.path.join(script_dir, "custom_export.py")
        
        export_command = [
            "python", custom_export_script,
            config_path,
            output_dir,
            # "-10.0"  # Lenient opacity threshold
        ]
        
        result = subprocess.run(export_command, env=env)
        if result.returncode != 0:
            print(f"❌ Export failed (exit code {result.returncode})")
            return 1
        
        print(f"✅ Successfully exported to {output_dir}/splat.ply")
    
    # Rotate PLY file
    from model.rotate_ply import rotate_ply_3dgs
    splat_path = os.path.join(output_dir, "splat.ply")
    rotated_path = os.path.join(output_dir, "splat_rotated.ply")
    
    if os.path.exists(splat_path):
        if os.path.exists(rotated_path):
            print(f"✅ Rotated PLY already exists: {rotated_path}")
        else:
            print(f"\nRotating PLY file...")
            try:
                rotate_ply_3dgs(splat_path, rotated_path)
                print(f"✅ Successfully rotated PLY: {rotated_path}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to rotate PLY: {e}")
    
    print(f"\n{'='*80}")
    print("✅ SCENE EXPANSION COMPLETED!")
    print(f"{'='*80}\n")
    print(f"Final output: {output_dir}/splat_rotated.ply")
    
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Continue scene expansion from training step")
    parser.add_argument("work_dir", help="Work directory (e.g., ./scenes/single_image_scenes_3.0_20260128_122421)")
    parser.add_argument("--scene-id", help="Scene ID (auto-detected from work_dir if not provided)")
    
    args = parser.parse_args()
    
    work_dir = os.path.abspath(args.work_dir)
    if not os.path.exists(work_dir):
        print(f"❌ Error: Work directory not found: {work_dir}")
        sys.exit(1)
    
    if args.scene_id:
        scene_id = args.scene_id
    else:
        # Extract scene ID from work_dir
        scene_id = os.path.basename(work_dir)
    
    sys.exit(continue_from_training(work_dir, scene_id))
