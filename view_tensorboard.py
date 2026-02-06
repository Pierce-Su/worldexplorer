#!/usr/bin/env python
"""
Helper script to launch TensorBoard for nerfstudio training logs.
This ensures TensorBoard is launched with the correct path and settings.
"""
import os
import sys
import subprocess
from pathlib import Path

def launch_tensorboard(log_dir, port=6006, host="0.0.0.0"):
    """
    Launch TensorBoard for the given log directory.
    
    Args:
        log_dir: Path to directory containing TensorBoard logs
        port: Port number for TensorBoard (default: 6006)
        host: Host address (default: 0.0.0.0 to allow remote access)
    """
    log_dir = Path(log_dir).resolve()
    
    if not log_dir.exists():
        print(f"❌ Error: Log directory not found: {log_dir}")
        return 1
    
    # Check if TensorBoard log files exist
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"⚠️  Warning: No TensorBoard event files found in {log_dir}")
        print(f"   Looking for files matching: events.out.tfevents.*")
        return 1
    
    print(f"✅ Found {len(event_files)} TensorBoard log file(s)")
    for ef in event_files[:3]:  # Show first 3 files
        size_mb = ef.stat().st_size / (1024 * 1024)
        print(f"   - {ef.name} ({size_mb:.1f} MB)")
    
    print(f"\n📁 Log directory: {log_dir}")
    print(f"🌐 Starting TensorBoard on http://{host}:{port}")
    
    # Verify logs can be read
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()
        tags = ea.Tags()
        if 'scalars' in tags:
            scalar_count = len(tags['scalars'])
            print(f"✅ Verified: {scalar_count} scalar metrics found")
            print(f"   Sample metrics: {', '.join(list(tags['scalars'])[:5])}")
        else:
            print("⚠️  Warning: No scalar metrics found in logs")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify logs: {e}")
    
    print(f"\n💡 Tips:")
    print(f"   - If running remotely, use SSH port forwarding:")
    print(f"     ssh -L {port}:localhost:{port} user@host")
    print(f"   - Then open: http://localhost:{port}")
    print(f"   - If dashboards are blank, try:")
    print(f"     * Clear browser cache (Ctrl+Shift+Delete)")
    print(f"     * Try incognito/private mode")
    print(f"     * Check browser console (F12) for errors")
    print(f"   - Press Ctrl+C to stop TensorBoard\n")
    
    # Launch TensorBoard
    try:
        cmd = [
            "tensorboard",
            "--logdir", str(log_dir),
            "--port", str(port),
            "--host", host,
            "--reload_interval", "5",  # Reload every 5 seconds
        ]
        
        # Use conda environment if available
        env = os.environ.copy()
        conda_env = os.environ.get("CONDA_PREFIX")
        if conda_env:
            # Ensure we're using the right Python
            python_path = os.path.join(conda_env, "bin", "python")
            if os.path.exists(python_path):
                cmd = ["python", "-m", "tensorboard.main"] + cmd[1:]
        
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n✅ TensorBoard stopped")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ TensorBoard failed to start (exit code {e.returncode})")
        print("\nTroubleshooting:")
        print("  1. Make sure TensorBoard is installed: pip install tensorboard")
        print("  2. Check if port is already in use: lsof -i :6006")
        print("  3. Try a different port: --port 6007")
        return 1
    except FileNotFoundError:
        print("\n❌ TensorBoard not found!")
        print("\nTo install TensorBoard:")
        print("  conda activate worldexplorer")
        print("  pip install tensorboard")
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch TensorBoard for nerfstudio training logs")
    parser.add_argument(
        "log_dir",
        nargs="?",
        help="Path to TensorBoard log directory (default: auto-detect latest training run)"
    )
    parser.add_argument("--port", type=int, default=6006, help="Port number (default: 6006)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument(
        "--scene-id",
        help="Scene ID to find logs for (e.g., single_image_scenes_3.0_20260128_122421)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect log directory if not provided
    if not args.log_dir:
        if args.scene_id:
            scene_id = args.scene_id
        else:
            # Try to find the most recent training run
            from model.scene_expansion import NERF_FOLDER
            nerf_output = Path(NERF_FOLDER)
            
            if not nerf_output.exists():
                print("❌ Error: Could not find nerfstudio output directory")
                print("   Please specify --log-dir or --scene-id")
                sys.exit(1)
            
            # Find all scene directories
            scenes = [d for d in nerf_output.iterdir() if d.is_dir()]
            if not scenes:
                print("❌ Error: No training runs found")
                sys.exit(1)
            
            # Get the most recent scene
            scene_id = max(scenes, key=lambda p: p.stat().st_mtime).name
            print(f"📁 Auto-detected scene: {scene_id}")
        
        # Find the most recent run in the scene
        scene_dir = Path(NERF_FOLDER) / scene_id / "splatfacto"
        if not scene_dir.exists():
            print(f"❌ Error: Scene directory not found: {scene_dir}")
            sys.exit(1)
        
        runs = [d for d in scene_dir.iterdir() if d.is_dir()]
        if not runs:
            print(f"❌ Error: No training runs found in {scene_dir}")
            sys.exit(1)
        
        # Get the most recent run
        latest_run = max(runs, key=lambda p: p.stat().st_mtime)
        args.log_dir = str(latest_run)
        print(f"📁 Auto-detected run: {latest_run.name}")
    
    sys.exit(launch_tensorboard(args.log_dir, args.port, args.host))
