# SEVA Scaffold Workflow: How It Works and the 007/000 Overlap Issue

## Trajectory: pan-in-place vs orbit

- **`pan-in-place`** (default): Camera **position is fixed**; only the viewing direction (yaw) rotates. Like standing in one spot and looking around. This avoids the camera path colliding with walls and producing broken views. Use this for indoor or constrained scenes.
- **`orbit`**: Camera **moves** along a circular path around the scene. Can give broken views when that path goes through walls or geometry. Use only when the scene is open (e.g. outdoor) or when you know the path is clear.

## How the SEVA workflow operates

When you use **scaffold method `single_image`** (the default in `run_batch_pipeline.py`), the scaffold is built in three steps, all automatic and **without any manual selection**:

### Step 1: Generate views (SEVA)

- **Entry**: `single_image_to_3d.generate_views_from_single_image()`
- **What it does**:
  - Takes your **single input image** (resized to 576×576).
  - Calls **SEVA** (`model/stable-virtual-camera/demo.py`) with task `img2trajvid_s-prob`.
  - SEVA uses a **procedural trajectory** (camera path) to generate many **synthetic views** from that one image.
- **Parameters you can set** (e.g. in batch pipeline or `single_image_to_3d.py`):
  - **`traj_prior`** (default `"orbit"`): Type of path — e.g. `orbit`, `spiral`, `pan-left`, `pan-right`, `move-forward`, etc.
  - **`num_targets`** (default `80`): Number of frames to generate along that path.
  - **`cfg`**, **`translation_scaling_factor`**: Model/scale options.
- **Output**: A sequence of frames in `work_dirs/backwards/img2trajvid_s-prob/<scene_name>/samples-rgb/` (and possibly `first-pass/samples-rgb/`). Frame order follows the trajectory (e.g. orbit = camera moving around the scene).

So SEVA produces a **video-like sequence** of views; it does **not** produce “8 cardinal views” by design. The next step turns that sequence into exactly 8 scaffold images.

### Step 2: Extract 8 key frames

- **Entry**: `single_image_to_3d.extract_key_frames_for_scaffold()`
- **What it does**:
  - Takes **all** frames from the SEVA output directory.
  - Picks **8 frames** by **even spacing** along the sequence:
    - `indices = np.linspace(0, len(frame_files) - 1, 8, dtype=int)`
  - Copies those 8 into the scaffold `images/` as `000.png` … `007.png`.
  - Then **overwrites `000.png`** with the **original input image** (so the first view is always your photo).
- **Important**: There is **no manual selection** here. The script does not show you alternatives or let you pick which frames become 001–007. It is fully automatic: “first frame = input, other 7 = evenly spaced from the SEVA sequence.”

### Step 3: Prepare final scaffold

- **Entry**: `single_image_to_3d.prepare_final_scaffold()`
- **What it does**: Ensures the final directory has exactly 8 images named `000.png`–`007.png` (copying/duplicating if there are fewer than 8).

So in total: **one image → SEVA trajectory → one fixed automatic downsampling to 8 views → your image forced as 000**. No manual curation step like in text-based scaffold generation.

---

## Why the 7th scaffold image (007) overlaps with the input (000)

This is a **geometry + sampling** issue.

### Trajectory is (almost) closed

- With **`traj_prior="orbit"`**, the camera path is a **horizontal arc** around the scene.
- In `seva/geometry.py`, `get_arc_horizontal_w2cs(..., endpoint=False)` is used with **degree=360**.
- So the trajectory runs from **0° to almost 360°**: the **last frame** is almost the same view as the **first frame** (one full loop).

So you have something like:

- Frame 0   ≈ 0°
- Frame 79  ≈ 360° (same as 0°)

### How we pick the 8 key frames

- We have ~80 frames (indices `0 .. 79`).
- We compute:  
  `indices = np.linspace(0, 79, 8) → [0, 11, 22, 33, 44, 55, 66, 79]`.
- So:
  - **000** is then replaced by the **input image** (≈ 0°).
  - **007** is taken from **frame 79** (≈ 360°) → **same viewing angle as 000**.

Result: **007 and 000 are the same view**, so you get “view inconsistency” and poor geometry (two scaffold views that are duplicates instead of spanning the panorama).

So: **SEVA does not by itself “misalign”**; the overlap comes from (1) using a **closed orbit** and (2) **evenly spacing 8 samples** over the full loop, so the last sample is the same as the first.

---

## Does SEVA allow manual selection?

**No.** In the current code:

- **Text-based scaffold** (WorldExplorer `scaffold_generation`): has **manual** mode — it generates many inpainting variants and **you** copy the chosen 001, 003, 005, 007 into `final/`.
- **SEVA-based scaffold** (`single_image_to_3d`): **no manual mode**. It always does “8 evenly spaced frames from the SEVA sequence + force input as 000.” You cannot currently:
  - Choose which of the 80 frames become 001–007, or
  - Replace any of them by hand before scene expansion.

So if you want manual control over the 8 views when using SEVA, that would require a **new option** (e.g. “save all SEVA frames + a script or UI to pick 8 and write 000–007,” or “manual keyframe indices” in config).

---

## Fix for the 007/000 overlap (recommended)

Avoid using the **end** of the orbit when picking the 8 key frames, so the last key frame is not the same view as the first.

A simple approach: **sample 8 frames only from the first 7/8 of the sequence** (or first 315° of the orbit), so the last key frame is around 315° instead of 360°:

- Instead of:  
  `indices = np.linspace(0, len(frame_files) - 1, 8, dtype=int)`  
  (which includes the last frame and causes 007 = 000).

- Use something like:  
  “8 evenly spaced indices over `[0, (N-1)*7/8]`”  
  so the 8th key frame is at ~87.5% of the sequence, not at 100%.

That keeps 000 and 007 distinct and improves view consistency. Optionally this can be made configurable (e.g. “use only first X% of trajectory for keyframes”) or default for `orbit` only.

---

## Summary

| Question | Answer |
|----------|--------|
| How does SEVA scaffold work? | One image → SEVA generates many views along a trajectory (e.g. orbit) → script picks 8 **evenly spaced** frames → input is forced as 000. All automatic. |
| Manual selection? | **No.** Only the text-based scaffold has a manual mode. SEVA path is fixed automatic downsampling. |
| Why does 007 overlap 000? | Orbit is a closed loop (0°–360°). Even spacing over the full loop makes the 8th frame the last frame ≈ 360° = same as 0°. |
| Fix? | Sample 8 key frames from the **first 7/8** of the sequence (or first ~315° of the orbit) so 007 is not the same view as 000. |

If you want, the next step is to implement the “first 7/8” sampling in `extract_key_frames_for_scaffold()` and optionally add a manual-selection path (e.g. export all SEVA frames + a small script to choose 8 indices and build the scaffold).
