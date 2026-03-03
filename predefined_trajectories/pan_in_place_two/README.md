# Pan-in-place two-view trajectory (0° + 180°)

Used by the **two_images** scaffold method in the same way Stage 2 uses `in/`, `left/`, `right/`, `up/`:

- **transforms.json**: 8 frames — frame 0 = 0°, frame 1 = 180° (inputs), frames 2–7 = 45°, 90°, 135°, 225°, 270°, 315° (target poses). All pan-in-place (rotation around Y only).
- **train_test_split_2.json**: `train_ids = [0, 1]`, `test_ids = [2, 3, 4, 5, 6, 7]`.

The pipeline copies the user's 0° and 180° images as `000.png` and `001.png`, and uses black placeholders for the target frames. SEVA **img2trajvid** then generates the six in-between views.

To change the trajectory (e.g. more target frames or different angles), edit `transforms.json` and `train_test_split_2.json` here. The two-image flow will use this folder when present (see `single_image_to_3d._build_two_images_reconfusion_folder`).
