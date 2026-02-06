#!/usr/bin/env python3
"""
Diagnostic script to check alignment between point cloud and cameras for 3DGS.
This helps identify why Gaussian Splatting results don't match the videos.
"""
import json
import numpy as np
import struct
from pathlib import Path

def analyze_alignment(transforms_json, ply_file):
    """Analyze alignment between cameras and point cloud."""
    
    print("="*80)
    print("3DGS ALIGNMENT DIAGNOSTIC")
    print("="*80)
    
    # Load transforms.json
    with open(transforms_json, 'r') as f:
        data = json.load(f)
    
    # Extract camera positions
    camera_positions = []
    for frame in data['frames']:
        tm = np.array(frame['transform_matrix'])
        cam_pos = tm[:3, 3]  # Camera position (translation)
        camera_positions.append(cam_pos)
    
    camera_positions = np.array(camera_positions)
    
    # Read point cloud
    print("\n📊 Reading point cloud...")
    with open(ply_file, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        vertex_count = None
        for line in header_lines:
            if 'element vertex' in line.lower():
                vertex_count = int(line.split()[-1])
                break
        
        # Read all points
        points = []
        for i in range(vertex_count):
            x = struct.unpack('<d', f.read(8))[0]
            y = struct.unpack('<d', f.read(8))[0]
            z = struct.unpack('<d', f.read(8))[0]
            r = struct.unpack('<B', f.read(1))[0]
            g = struct.unpack('<B', f.read(1))[0]
            b = struct.unpack('<B', f.read(1))[0]
            points.append([x, y, z])
        
        points = np.array(points)
    
    print(f"✅ Loaded {len(points)} points and {len(camera_positions)} cameras")
    
    # Analyze cameras
    print("\n" + "="*80)
    print("CAMERA ANALYSIS")
    print("="*80)
    cam_center = camera_positions.mean(axis=0)
    cam_min = camera_positions.min(axis=0)
    cam_max = camera_positions.max(axis=0)
    cam_span = cam_max - cam_min
    cam_span_mag = np.linalg.norm(cam_span)
    
    print(f"Camera center: {cam_center}")
    print(f"Camera bbox: [{cam_min}, {cam_max}]")
    print(f"Camera span: {cam_span} (magnitude: {cam_span_mag:.3f})")
    
    # Analyze point cloud
    print("\n" + "="*80)
    print("POINT CLOUD ANALYSIS")
    print("="*80)
    pc_center = points.mean(axis=0)
    pc_min = points.min(axis=0)
    pc_max = points.max(axis=0)
    pc_span = pc_max - pc_min
    pc_span_mag = np.linalg.norm(pc_span)
    
    print(f"Point cloud center: {pc_center}")
    print(f"Point cloud bbox: [{pc_min}, {pc_max}]")
    print(f"Point cloud span: {pc_span} (magnitude: {pc_span_mag:.3f})")
    
    # Compare
    print("\n" + "="*80)
    print("ALIGNMENT ANALYSIS")
    print("="*80)
    
    # Center offset
    center_offset = np.linalg.norm(pc_center - cam_center)
    print(f"Center offset: {center_offset:.3f} units")
    
    # Scale ratio
    scale_ratio = pc_span_mag / cam_span_mag if cam_span_mag > 0 else float('inf')
    print(f"Scale ratio (pc/cam): {scale_ratio:.3f}x")
    
    # Check if cameras can see the point cloud
    print("\n" + "="*80)
    print("VISIBILITY CHECK")
    print("="*80)
    
    # For each camera, check how many points are in front
    points_in_view = []
    for i, cam_pos in enumerate(camera_positions[:10]):  # Check first 10 cameras
        # Get camera orientation (assuming looking down -Z)
        tm = np.array(data['frames'][i]['transform_matrix'])
        cam_forward = -tm[:3, 2]  # Negative Z axis
        
        # Vector from camera to points
        to_points = points - cam_pos
        distances = np.linalg.norm(to_points, axis=1)
        
        # Points in front of camera (dot product > 0)
        in_front = np.dot(to_points, cam_forward) > 0
        points_in_view.append(in_front.sum())
    
    avg_points_in_view = np.mean(points_in_view)
    print(f"Average points visible per camera: {avg_points_in_view:.0f} / {len(points)}")
    
    # Diagnose issues
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    issues = []
    
    if center_offset > 2.0:
        issues.append(f"❌ CRITICAL: Cameras and point cloud are {center_offset:.2f} units apart!")
        issues.append("   They should be roughly centered together.")
    
    if scale_ratio > 2.0 or scale_ratio < 0.5:
        issues.append(f"⚠️  WARNING: Scale mismatch - point cloud is {scale_ratio:.2f}x cameras")
        issues.append("   This may cause poor 3DGS results.")
    
    if avg_points_in_view < len(points) * 0.1:
        issues.append(f"❌ CRITICAL: Only {avg_points_in_view:.0f} points visible per camera!")
        issues.append("   Cameras can't see the scene properly.")
    
    if cam_span_mag < 0.5:
        issues.append(f"❌ CRITICAL: Cameras span only {cam_span_mag:.3f} units!")
        issues.append("   Cameras are too clustered - need more spread.")
    
    if pc_span_mag < 0.1:
        issues.append(f"❌ CRITICAL: Point cloud spans only {pc_span_mag:.3f} units!")
        issues.append("   Point cloud is too small.")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ No obvious alignment issues detected.")
        print("   If 3DGS still fails, check:")
        print("   1. Point cloud quality (density, coverage)")
        print("   2. Camera pose accuracy")
        print("   3. Training parameters")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if center_offset > 1.0:
        print(f"1. Translate point cloud by {-pc_center + cam_center} to align centers")
    
    if scale_ratio > 1.5:
        scale_factor = cam_span_mag / pc_span_mag
        print(f"2. Scale point cloud by {scale_factor:.3f}x to match camera scale")
    elif scale_ratio < 0.67:
        scale_factor = cam_span_mag / pc_span_mag
        print(f"2. Scale point cloud by {scale_factor:.3f}x to match camera scale")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python debug_3dgs_alignment.py <transforms.json> <vggt_pcl.ply>")
        sys.exit(1)
    
    transforms_json = sys.argv[1]
    ply_file = sys.argv[2]
    
    analyze_alignment(transforms_json, ply_file)
