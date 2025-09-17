#!/usr/bin/env python3
"""
Rotate a binary PLY (binary_little_endian 1.0) point cloud for 3DGS:
- Rotate positions (x, y, z)
- Update per-point quaternions (rot_0..3)
- Preserve ALL other attributes (scale, opacity, sh, labels, etc.)
"""

import numpy as np
from plyfile import PlyData, PlyElement


def quat_multiply(q1, q2):
    """Quaternion multiplication: q = q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def rot_matrix_to_quat(R):
    """Convert rotation matrix to quaternion (w, x, y, z)"""
    m = R
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def rotate_ply_3dgs(input_path: str, output_path: str):
    """
    Rotate a 3DGS binary PLY point cloud:
    - Rotate positions (x, y, z)
    - Rotate per-point quaternions (rot_0..3)
    - Keep all other attributes unchanged
    """

    # --- Step 1: Read binary PLY ---
    plydata = PlyData.read(input_path)
    vertex_data = plydata['vertex'].data  # structured array

    # --- Step 2: Build global rotation (90° X, 180° Y) ---
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(90)), -np.sin(np.radians(90))],
        [0, np.sin(np.radians(90)),  np.cos(np.radians(90))]
    ])
    Ry = np.array([
        [np.cos(np.radians(180)), 0, np.sin(np.radians(180))],
        [0, 1, 0],
        [-np.sin(np.radians(180)), 0, np.cos(np.radians(180))]
    ])
    R = Ry @ Rx
    q_global = rot_matrix_to_quat(R)

    # --- Step 3: Rotate positions ---
    xyz = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    rotated_xyz = xyz @ R.T
    vertex_data['x'], vertex_data['y'], vertex_data['z'] = rotated_xyz.T

    # --- Step 4: Rotate per-point quaternions ---
    if all(name in vertex_data.dtype.names for name in ['rot_0', 'rot_1', 'rot_2', 'rot_3']):
        quats = np.vstack([vertex_data['rot_0'],
                           vertex_data['rot_1'],
                           vertex_data['rot_2'],
                           vertex_data['rot_3']]).T
        rotated_quats = np.array([quat_multiply(q_global, q) for q in quats])
        vertex_data['rot_0'] = rotated_quats[:, 0]
        vertex_data['rot_1'] = rotated_quats[:, 1]
        vertex_data['rot_2'] = rotated_quats[:, 2]
        vertex_data['rot_3'] = rotated_quats[:, 3]

    # --- Step 5: Write back binary PLY ---
    PlyData([PlyElement.describe(vertex_data, 'vertex')],
            text=False).write(output_path)


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python rotate_ply_3dgs.py <input.ply> <output.ply>")
        sys.exit(1)
    rotate_ply_3dgs(sys.argv[1], sys.argv[2])
    print(f"Rotated 3DGS PLY saved to: {sys.argv[2]}")
