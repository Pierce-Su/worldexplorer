#!/usr/bin/env python
"""
Diagnostic script to inspect PLY files and identify corruption issues.
"""
import sys
import numpy as np
from pathlib import Path

try:
    from plyfile import PlyData
except ImportError:
    print("Error: plyfile not installed. Install with: pip install plyfile")
    sys.exit(1)


def inspect_ply(ply_path):
    """Inspect a PLY file and report any issues."""
    ply_path = Path(ply_path)
    
    if not ply_path.exists():
        print(f"❌ Error: File not found: {ply_path}")
        return False
    
    print(f"\n{'='*80}")
    print(f"Inspecting: {ply_path.name}")
    print(f"{'='*80}\n")
    
    try:
        ply = PlyData.read(str(ply_path))
    except Exception as e:
        print(f"❌ Error reading PLY file: {e}")
        return False
    
    if 'vertex' not in ply:
        print("❌ Error: No 'vertex' element found in PLY")
        return False
    
    vertex_data = ply['vertex'].data
    n_vertices = len(vertex_data)
    print(f"✅ Vertices: {n_vertices:,}")
    
    # Check properties
    props = vertex_data.dtype.names
    print(f"✅ Properties ({len(props)}): {', '.join(props[:10])}{'...' if len(props) > 10 else ''}")
    
    # Check for required properties
    required = ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    missing = [p for p in required if p not in props]
    if missing:
        print(f"⚠️  Missing required properties: {missing}")
    else:
        print("✅ All required properties present")
    
    # Check for NaN/Inf
    print("\n📊 Data Quality Checks:")
    issues = []
    
    for prop in ['x', 'y', 'z', 'opacity', 'scale_0']:
        if prop in props:
            data = vertex_data[prop]
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            if nan_count > 0:
                issues.append(f"{prop}: {nan_count} NaN values")
            if inf_count > 0:
                issues.append(f"{prop}: {inf_count} Inf values")
            
            if prop in ['x', 'y', 'z']:
                print(f"  {prop}: range [{data.min():.3f}, {data.max():.3f}], "
                      f"NaN={nan_count}, Inf={inf_count}")
            elif prop == 'opacity':
                print(f"  {prop}: range [{data.min():.6f}, {data.max():.6f}], "
                      f"NaN={nan_count}, Inf={inf_count}")
            else:
                print(f"  {prop}: range [{data.min():.6f}, {data.max():.6f}], "
                      f"NaN={nan_count}, Inf={inf_count}")
    
    if issues:
        print(f"\n⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ No NaN/Inf values detected")
    
    # Check data ranges for reasonableness
    print("\n📏 Reasonableness Checks:")
    warnings = []
    
    if 'x' in props:
        x_range = abs(vertex_data['x'].max() - vertex_data['x'].min())
        if x_range > 1000:
            warnings.append(f"Very large X range: {x_range:.2f}")
        elif x_range < 0.01:
            warnings.append(f"Very small X range: {x_range:.2f}")
    
    if 'opacity' in props:
        opacity_min = vertex_data['opacity'].min()
        opacity_max = vertex_data['opacity'].max()
        # Opacity should be in logit space, typically between -10 and 10
        if opacity_min < -20 or opacity_max > 20:
            warnings.append(f"Unusual opacity range: [{opacity_min:.2f}, {opacity_max:.2f}]")
    
    if 'scale_0' in props:
        scale_min = vertex_data['scale_0'].min()
        scale_max = vertex_data['scale_0'].max()
        if scale_min < 0:
            warnings.append(f"Negative scale values: min={scale_min:.6f}")
        if scale_max > 100:
            warnings.append(f"Very large scale values: max={scale_max:.6f}")
    
    if warnings:
        print("⚠️  Warnings:")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("✅ Data ranges look reasonable")
    
    # Sample data
    print("\n📋 Sample Data (first 3 vertices):")
    for i in range(min(3, n_vertices)):
        vi = vertex_data[i]
        print(f"  [{i}] pos=({vi['x']:.3f}, {vi['y']:.3f}, {vi['z']:.3f}), "
              f"opacity={vi['opacity']:.6f}, scale=({vi['scale_0']:.6f}, {vi['scale_1']:.6f}, {vi['scale_2']:.6f})")
    
    # Check spherical harmonics
    if 'f_dc_0' in props:
        print("\n🎨 Spherical Harmonics:")
        sh_dc = [vertex_data[f'f_dc_{i}'] for i in range(3) if f'f_dc_{i}' in props]
        if sh_dc:
            print(f"  DC coefficients: ranges [{sh_dc[0].min():.3f}, {sh_dc[0].max():.3f}]")
        sh_rest_count = sum(1 for i in range(100) if f'f_rest_{i}' in props)
        if sh_rest_count > 0:
            print(f"  Rest coefficients: {sh_rest_count} found")
    
    print(f"\n{'='*80}\n")
    
    return len(issues) == 0 and len(warnings) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_ply.py <ply_file> [ply_file2 ...]")
        sys.exit(1)
    
    all_good = True
    for ply_path in sys.argv[1:]:
        if not inspect_ply(ply_path):
            all_good = False
    
    sys.exit(0 if all_good else 1)
