# 3DGS Debugging Report

## Summary
Gaussian Splatting results don't match generated videos. This report documents findings and potential fixes.

## Key Findings

### ✅ What's Working
1. **Point Cloud**: 131,478 points (reasonable count)
2. **PLY File**: Properly referenced in `transforms.json`
3. **Camera Count**: 2,445 cameras (good coverage)
4. **Basic Alignment**: Cameras and point cloud are roughly aligned (0.75 unit offset)

### ⚠️ Potential Issues Found

#### 1. Scale Mismatch (MODERATE)
- **Point cloud span**: ~4.2 units
- **Camera span**: ~3.0 units  
- **Ratio**: 1.4x (point cloud is 40% larger)
- **Impact**: May cause 3DGS to struggle with initialization

#### 2. Alignment Code Issue (POTENTIAL CRITICAL)
Looking at `pred_and_align_with_transforms.py`:
- Uses **only the first camera** (`i = 0`) for alignment transformation
- This could be problematic if first camera pose is inaccurate
- Scale factor calculated from camera bbox lengths might be wrong

```python
# Line 129: Uses only first camera!
i = 0
W2C_pred = pred_w2c[i]  # (4, 4)
C2W_gt = gt_c2w[i]  # (4, 4)
```

#### 3. Coordinate System Conversion
- Code converts OpenCV → OpenGL (flips Y and Z)
- But uses only first camera for transformation
- May not handle all cameras correctly

#### 4. Point Cloud Scaling Logic
The scaling uses:
```python
gt_to_pred_scale = gt_poses_bbox_length / pred_poses_bbox_length
```
- Scales point cloud to match camera pose scale
- But if VGGT predictions are wrong scale, this propagates error

## Diagnostic Results

### Camera Statistics
- **Center**: [0.032, -0.000, -0.016]
- **Span**: 2.96 units
- **Distance to PC center**: Mean 1.33 units (reasonable)

### Point Cloud Statistics  
- **Center**: [0.74, 0.23, -0.12]
- **Span**: 4.20 units
- **Bounding box**: [-1.14 to 1.71] in X, [-1.35 to 1.60] in Y, [-0.68 to 0.22] in Z

### Visibility
- Average points visible per camera: ~55,971 / 131,478 (42%)
- This is reasonable, but could be better

## Potential Root Causes

### Most Likely Issues (in order):

1. **Single Camera Alignment** ⚠️ CRITICAL
   - Using only first camera for transformation is fragile
   - If first camera pose is wrong, entire alignment fails
   - **Fix**: Use all cameras or robust alignment (RANSAC, Procrustes)

2. **Scale Factor Calculation** ⚠️ MODERATE
   - Scale based on camera bbox might not match point cloud scale
   - VGGT predictions might have different scale than GT
   - **Fix**: Use point cloud scale directly or better scale estimation

3. **Coordinate System Mismatch** ⚠️ MODERATE
   - OpenCV vs OpenGL conversion might be incomplete
   - Some cameras might not be properly converted
   - **Fix**: Verify all cameras are in correct coordinate system

4. **Point Cloud Quality** ⚠️ LOW
   - 131k points might not be dense enough
   - Downsampling (1%) might remove important details
   - **Fix**: Increase point count or use better sampling

## Recommended Fixes

### Fix 1: Use All Cameras for Alignment (HIGH PRIORITY)
Replace single-camera alignment with multi-camera alignment:

```python
# Instead of:
i = 0
T_pred_to_gt = C2W_gt[i] @ cv2gl @ W2C_pred[i]

# Use Procrustes or RANSAC alignment with all cameras
from scipy.spatial.transform import Rotation
# ... robust alignment code ...
```

### Fix 2: Verify Scale Factor
Add logging to check if scale factor is reasonable:

```python
print(f"GT camera bbox length: {gt_poses_bbox_length}")
print(f"Pred camera bbox length: {pred_poses_bbox_length}")
print(f"Scale factor: {gt_to_pred_scale}")
if gt_to_pred_scale > 2.0 or gt_to_pred_scale < 0.5:
    print("WARNING: Scale factor seems wrong!")
```

### Fix 3: Check Point Cloud Initialization
Verify nerfstudio actually uses the PLY file:

```python
# Add to training command or check logs
# Should see: "Loading initial point cloud from: ..."
```

### Fix 4: Increase Point Cloud Density
Reduce downsampling or use better sampling:

```python
# Current: 1% downsampling
pcd = pcd.random_down_sample(0.01)

# Try: 5% or use voxel downsampling
pcd = pcd.voxel_down_sample(voxel_size=0.01)
```

## Next Steps

1. **Check training logs** for initialization warnings
2. **Visualize** point cloud + cameras together (use Open3D or similar)
3. **Test** with fewer cameras first to isolate issue
4. **Compare** VGGT predictions with actual video frames
5. **Verify** coordinate system consistency throughout pipeline

## Files to Check

- `model/pred_and_align_with_transforms.py` - Alignment logic
- `model/svc_to_nerf_transform_sparse.py` - Transform merging
- Training logs in `nerfstudio_output/` - Check for warnings
- `transforms.json` - Verify all cameras have correct poses

## Tools Created

- `debug_3dgs_alignment.py` - Diagnostic script for alignment issues
- `count_ply_points.py` - Point cloud analysis tool
