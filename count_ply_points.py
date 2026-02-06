#!/usr/bin/env python3
"""
Script to count points in PLY files and provide basic statistics.
Works with both ASCII and binary PLY formats.
"""
import sys
import struct
from pathlib import Path


def count_ply_points(ply_path):
    """
    Count points in a PLY file by reading the header.
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        tuple: (point_count, is_binary, properties_list)
    """
    ply_path = Path(ply_path)
    
    if not ply_path.exists():
        print(f"❌ Error: File not found: {ply_path}")
        return None, None, None
    
    with open(ply_path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        # Check format
        header_text = ' '.join(header_lines).lower()
        is_binary = 'binary' in header_text
        
        # Find vertex count
        vertex_count = None
        properties = []
        for line in header_lines:
            if 'element vertex' in line.lower():
                vertex_count = int(line.split()[-1])
            elif 'property' in line.lower():
                # Extract property name (last word after 'property')
                prop_parts = line.split()
                if len(prop_parts) >= 2:
                    properties.append(prop_parts[-1])
        
        return vertex_count, is_binary, properties


def analyze_ply(ply_path):
    """Analyze a PLY file and print statistics."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {ply_path}")
    print(f"{'='*80}\n")
    
    point_count, is_binary, properties = count_ply_points(ply_path)
    
    if point_count is None:
        return False
    
    print(f"Format: {'Binary' if is_binary else 'ASCII'}")
    print(f"Number of points (vertices): {point_count:,}")
    
    if properties:
        print(f"\nProperties ({len(properties)}):")
        # Group properties for better readability
        coord_props = [p for p in properties if p in ['x', 'y', 'z']]
        color_props = [p for p in properties if p in ['red', 'green', 'blue', 'r', 'g', 'b']]
        gaussian_props = [p for p in properties if any(gp in p for gp in ['opacity', 'scale', 'rot', 'f_dc', 'f_rest'])]
        other_props = [p for p in properties if p not in coord_props + color_props + gaussian_props]
        
        if coord_props:
            print(f"  Coordinates: {', '.join(coord_props)}")
        if color_props:
            print(f"  Colors: {', '.join(color_props)}")
        if gaussian_props:
            print(f"  Gaussian parameters: {', '.join(gaussian_props[:10])}{'...' if len(gaussian_props) > 10 else ''}")
        if other_props:
            print(f"  Other: {', '.join(other_props[:10])}{'...' if len(other_props) > 10 else ''}")
    
    # Determine file type
    print(f"\nFile Type:")
    if any('opacity' in p.lower() or 'scale' in p.lower() or 'rot' in p.lower() for p in properties):
        print("  ✅ Gaussian Splatting PLY (contains Gaussian parameters)")
    else:
        print("  📍 Regular Point Cloud PLY (coordinates + colors)")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_ply_points.py <ply_file> [ply_file2 ...]")
        print("\nExample:")
        print("  python count_ply_points.py scene/vggt_pcl.ply")
        sys.exit(1)
    
    all_success = True
    for ply_path in sys.argv[1:]:
        if not analyze_ply(ply_path):
            all_success = False
    
    sys.exit(0 if all_success else 1)
