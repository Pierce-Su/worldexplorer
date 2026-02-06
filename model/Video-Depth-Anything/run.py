# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import argparse
import numpy as np
import os
import torch
from pathlib import Path

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    parser.add_argument('--cpu', action='store_true', help='force CPU mode (useful when GPU memory is low)')

    args = parser.parse_args()

    if args.cpu:
        DEVICE = 'cpu'
    else:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # Get the script's directory and resolve checkpoint path relative to worldexplorer root
    script_dir = Path(__file__).parent.absolute()
    # Go up to worldexplorer root: script_dir is model/Video-Depth-Anything, so parent.parent is worldexplorer root
    worldexplorer_root = script_dir.parent.parent
    checkpoint_path = worldexplorer_root / "model" / "Video-Depth-Anything" / "checkpoints" / f"video_depth_anything_{args.encoder}.pth"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print(f"\nPlease download the checkpoint from:")
        print(f"  https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_{args.encoder}.pth")
        print(f"\nOr run from worldexplorer root:")
        print(f"  wget -O {checkpoint_path} 'https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_{args.encoder}.pth?download=true'")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    
    # Try to load on GPU, fallback to CPU if OOM
    try:
        video_depth_anything = video_depth_anything.to(DEVICE).eval()
    except torch.cuda.OutOfMemoryError as e:
        if DEVICE == 'cuda':
            print(f"CUDA out of memory when loading model on GPU. Falling back to CPU...")
            print(f"Original error: {e}")
            DEVICE = 'cpu'
            video_depth_anything = video_depth_anything.to(DEVICE).eval()
        else:
            raise

    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
    
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()

    


