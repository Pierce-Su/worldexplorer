import os
import json
import shutil
from tqdm.auto import tqdm



def select_images_for_vggt(scene_root, max_images=100):
    transforms_json = os.path.join(scene_root, "transforms.json")
    transforms_data = json.load(open(transforms_json, "r"))

    total_images = len(transforms_data["frames"])

    # sort per part
    input_frames = [x['file_path'] for x in transforms_data["frames"] if "input" in x['file_path']]

    trajs = {}
    for f in transforms_data["frames"]:
        file_path = f['file_path']
        if "input" in file_path:
            continue

        traj = file_path.split("/")[-2]
        if traj not in trajs:
            trajs[traj] = []
        trajs[traj].append(file_path)

    n_trajs = len(trajs.keys())
    allowed_images_per_trajper_traj = max_images // n_trajs

    for traj, images in trajs.items():
        images = sorted(images, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        n_images_in_traj = len(images)
        stride = n_images_in_traj // allowed_images_per_trajper_traj
        images = images[::stride]
        input_frames += images
    
    # copy input_frames to selected output folder
    output_folder = os.path.join(scene_root, "selected_for_vggt")
    os.makedirs(output_folder, exist_ok=True)
    for f in input_frames:
        parent_folder = os.path.dirname(f)
        parent_parent_folder = os.path.dirname(parent_folder)
        out_file_name = f"{os.path.basename(parent_parent_folder)}_{os.path.basename(parent_folder)}_{os.path.basename(f)}"
        shutil.copy(f, os.path.join(output_folder, out_file_name))