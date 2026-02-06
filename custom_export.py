#!/usr/bin/env python
"""
Custom export script for Gaussian Splatting with configurable opacity threshold.
This patches nerfstudio's export to use a more lenient opacity threshold.
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict

# Set environment variables before importing nerfstudio
os.environ['NERFSTUDIO_DISABLE_TORCH_COMPILE'] = '1'

# Patch torch.load to allow loading weights
orig_torch_load = torch.load
torch.load = lambda *a, **k: orig_torch_load(*a, **{**k, 'weights_only': False})

# Import nerfstudio modules
from nerfstudio.scripts.exporter import ExportGaussianSplat
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.scripts.eval import eval_setup
from nerfstudio.models.splatfacto import SplatfactoModel


def export_with_lenient_opacity(config_path, output_dir, opacity_threshold=-10.0, ply_color_mode="sh_coeffs"):
    """
    Export Gaussian Splatting model with a lenient opacity threshold.
    
    Args:
        config_path: Path to nerfstudio config YAML file
        output_dir: Output directory for the PLY file
        opacity_threshold: Opacity threshold in logit space (default: -10.0, which is much more lenient than default -5.5373)
        ply_color_mode: Color mode - "sh_coeffs" or "rgb" (default: "sh_coeffs")
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pipeline using eval_setup (same as nerfstudio)
    _, pipeline, _, _ = eval_setup(config_path, test_mode="inference")
    
    assert isinstance(pipeline.model, SplatfactoModel)
    model: SplatfactoModel = pipeline.model
    
    output_filename = output_dir / "splat.ply"
    
    map_to_tensors = OrderedDict()
    
    with torch.no_grad():
        positions = model.means.cpu().numpy()
        count = positions.shape[0]
        n = count
        map_to_tensors["x"] = positions[:, 0]
        map_to_tensors["y"] = positions[:, 1]
        map_to_tensors["z"] = positions[:, 2]
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)
        
        if ply_color_mode == "rgb":
            colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
            colors = (colors * 255).astype(np.uint8)
            map_to_tensors["red"] = colors[:, 0]
            map_to_tensors["green"] = colors[:, 1]
            map_to_tensors["blue"] = colors[:, 2]
        elif ply_color_mode == "sh_coeffs":
            shs_0 = model.shs_0.contiguous().cpu().numpy()
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]
        
        if model.config.sh_degree > 0:
            if ply_color_mode == "rgb":
                CONSOLE.print(
                    "Warning: model has higher level of spherical harmonics, ignoring them and only export rgb."
                )
            elif ply_color_mode == "sh_coeffs":
                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
        
        map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()
        
        scales = model.scales.data.cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]
        
        quats = model.quats.data.cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]
    
    # Filter NaN/Inf values
    select = np.ones(n, dtype=bool)
    for k, t in map_to_tensors.items():
        n_before = np.sum(select)
        select = np.logical_and(select, np.isfinite(t).all(axis=-1))
        n_after = np.sum(select)
        if n_after < n_before:
            CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")
    
    # Calculate how many were filtered due to NaN/Inf (before opacity filtering)
    nan_count = n - np.sum(select)
    
    # Filter Gaussians with invalid scales
    # Scales in log space that are too negative (< -10) result in near-zero scales
    # which cause rendering corruption. Filter these out.
    scale_min_threshold = -10.0  # exp(-10) ≈ 0.000045, which is effectively invisible
    invalid_scales = (
        (map_to_tensors["scale_0"].squeeze(axis=-1) < scale_min_threshold) |
        (map_to_tensors["scale_1"].squeeze(axis=-1) < scale_min_threshold) |
        (map_to_tensors["scale_2"].squeeze(axis=-1) < scale_min_threshold)
    )
    invalid_scale_count = np.sum(invalid_scales)
    select[invalid_scales] = 0
    
    # Use lenient opacity threshold
    # Default nerfstudio uses -5.5373 (logit(1/255))
    # We use a much more lenient threshold to keep more Gaussians
    low_opacity_gaussians = (map_to_tensors["opacity"]).squeeze(axis=-1) < opacity_threshold
    lowopa_count = np.sum(low_opacity_gaussians)
    select[low_opacity_gaussians] = 0
    
    if np.sum(select) < n:
        CONSOLE.print(
            f"{nan_count} Gaussians have NaN/Inf, {invalid_scale_count} have invalid scales "
            f"(threshold={scale_min_threshold:.2f}), and {lowopa_count} have low opacity "
            f"(threshold={opacity_threshold:.2f}), exporting {np.sum(select)}/{n}"
        )
        for k, t in map_to_tensors.items():
            map_to_tensors[k] = map_to_tensors[k][select]
        count = np.sum(select)
    else:
        count = n
    
    # Write PLY file
    ExportGaussianSplat.write_ply(str(output_filename), count, map_to_tensors)
    CONSOLE.print(f"✅ Successfully exported to {output_filename}")
    return output_filename


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python custom_export.py <config_path> <output_dir> [opacity_threshold]")
        print("  opacity_threshold: Opacity threshold in logit space (default: -10.0)")
        print("                     Default nerfstudio uses -5.5373 (logit(1/255))")
        sys.exit(1)
    
    config_path = sys.argv[1]
    output_dir = sys.argv[2]
    opacity_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else -10.0
    
    export_with_lenient_opacity(config_path, output_dir, opacity_threshold)
