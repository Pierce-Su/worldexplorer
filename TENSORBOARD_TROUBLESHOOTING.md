# TensorBoard Troubleshooting Guide

## Quick Start

Use the helper script to launch TensorBoard:

```bash
conda activate worldexplorer
python view_tensorboard.py --scene-id single_image_scenes_3.0_20260128_122421
```

Or specify the exact log directory:

```bash
python view_tensorboard.py ./nerfstudio_output/single_image_scenes_3.0_20260128_122421/splatfacto/2026-02-04_164440
```

## Common Issues and Solutions

### 1. "No dashboards are shown" / Blank TensorBoard

**Symptoms:** TensorBoard starts but shows empty dashboards or "No dashboards are currently active"

**Solutions:**

#### A. Point TensorBoard to the correct directory
TensorBoard needs to point to the **run directory** (the one containing `events.out.tfevents.*`), not the parent directory:

```bash
# ✅ CORRECT - points to the run directory
tensorboard --logdir ./nerfstudio_output/single_image_scenes_3.0_20260128_122421/splatfacto/2026-02-04_164440

# ❌ WRONG - points to parent directory
tensorboard --logdir ./nerfstudio_output/single_image_scenes_3.0_20260128_122421/splatfacto
```

#### B. Clear TensorBoard cache
TensorBoard caches data. Clear it:

```bash
# Clear TensorBoard cache
rm -rf ~/.tensorboard-info

# Or use --reload flag
tensorboard --logdir <path> --reload_interval 5
```

#### C. Check browser console
Open browser developer tools (F12) and check for JavaScript errors in the Console tab.

#### D. Try different browser
Sometimes browser extensions or cache can cause issues. Try:
- Incognito/Private mode
- Different browser (Chrome, Firefox, Edge)
- Clear browser cache

### 2. "TensorBoard not found" / Command not found

**Solution:**
```bash
conda activate worldexplorer
pip install tensorboard
```

### 3. Remote Server Access

If you're running TensorBoard on a remote server:

#### Option A: SSH Port Forwarding (Recommended)
```bash
# On your local machine
ssh -L 6006:localhost:6006 user@remote-server

# Then on remote server
tensorboard --logdir <path> --port 6006 --host localhost

# Access from local browser: http://localhost:6006
```

#### Option B: Use --host 0.0.0.0
```bash
# On remote server
tensorboard --logdir <path> --host 0.0.0.0 --port 6006

# Access via: http://remote-server-ip:6006
# (Make sure firewall allows port 6006)
```

### 4. Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find what's using the port
lsof -i :6006

# Kill the process or use a different port
tensorboard --logdir <path> --port 6007
```

### 5. Verify Logs Exist

Check if TensorBoard can read your logs:

```bash
python -c "
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('./nerfstudio_output/single_image_scenes_3.0_20260128_122421/splatfacto/2026-02-04_164440')
ea.Reload()
print('Available tags:', list(ea.Tags().keys()))
if 'scalars' in ea.Tags():
    print('Scalars:', list(ea.Tags()['scalars'])[:10])
"
```

### 6. Check Log File Size

Very small log files might indicate incomplete training:

```bash
ls -lh ./nerfstudio_output/single_image_scenes_3.0_20260128_122421/splatfacto/2026-02-04_164440/events.out.tfevents.*
```

Your log file should be several MB (yours is ~11MB, which is good).

## Available Metrics in Your Logs

Based on your training run, you should see these metrics:

- **Train Loss** - Overall training loss
- **Train Loss Dict/main_loss** - Main rendering loss
- **Train Loss Dict/mcmc_opacity_reg** - MCMC opacity regularization
- **Train Loss Dict/mcmc_scale_reg** - MCMC scale regularization
- **Train Loss Dict/scale_reg** - Scale regularization
- **Train Metrics Dict/psnr** - Peak Signal-to-Noise Ratio
- **Train Metrics Dict/gaussian_count** - Number of Gaussians
- **learning_rate/means** - Learning rate for Gaussian means
- **Train Iter (time)** - Training iteration time
- **ETA (time)** - Estimated time to completion

## Manual TensorBoard Launch

If the helper script doesn't work, launch manually:

```bash
conda activate worldexplorer
cd /mnt/facesim/pierce_E3DQA/locally_run_repos/worldexplorer
tensorboard \
    --logdir ./nerfstudio_output/single_image_scenes_3.0_20260128_122421/splatfacto/2026-02-04_164440 \
    --port 6006 \
    --host 0.0.0.0 \
    --reload_interval 5
```

Then open: `http://localhost:6006` (or `http://your-server-ip:6006` if remote)

## Still Not Working?

1. Check TensorBoard version: `tensorboard --version`
2. Try upgrading: `pip install --upgrade tensorboard`
3. Check Python version compatibility
4. Look at TensorBoard logs for errors
5. Try reading logs with Python directly (see section 5 above)
