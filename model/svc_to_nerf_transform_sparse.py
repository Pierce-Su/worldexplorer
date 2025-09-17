import os
import json
import argparse

def merge_transforms(root_dir):
    """
    Merges all 'transforms.json' files found in the subfolders of 'root_dir'
    into a single 'transforms.json' file at the root level.
    Only includes frames whose images are not located in a folder named 'input'.
    """

    # Where we will write our final merged file:
    output_path = os.path.join(root_dir, "transforms.json")

    # Intrinsics (placed only once at the top of the new transforms.json):
    top_level_intrinsics = {
        "fl_x": 498,
        "fl_y": 498,
        "cx": 288,
        "cy": 288,
        "w":  576,
        "h":  576
    }

    all_frames = []

    # Find all subfolders (ignore files at the root level):
    subfolders = [
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f)) and f not in ('.', '..')
    ]

    added_input = 8
    for folder_name in subfolders:
        # The path to the transforms.json inside this subfolder:
        subfolder_transforms = os.path.join(root_dir, folder_name, "transforms.json")
        if not os.path.isfile(subfolder_transforms):
            # Skip if there's no transforms.json in this folder
            continue

        with open(subfolder_transforms, "r") as f:
            data = json.load(f)

        # Each subfolder's transforms.json has a "frames" list
        frames = data.get("frames", [])
        for frame in frames:
            # Remove the intrinsics from the frame (since they will be
            # specified at the top of the new transforms.json):
            frame.pop("fl_x", None)
            frame.pop("fl_y", None)
            frame.pop("cx", None)
            frame.pop("cy", None)
            frame.pop("w", None)
            frame.pop("h", None)

            # Fix the file_path to an absolute path:
            old_path = frame.get("file_path", "")
            abs_path = os.path.abspath(os.path.join(root_dir, folder_name, old_path))
            
            # Extract the directory part of the path (excluding the filename)
            dir_path = os.path.dirname(abs_path)
            # Check if the last directory is named "input"
            # make sure that we include the very first "input" folder
            if os.path.basename(dir_path) == "input":
                if added_input > 0:
                    added_input -= 1
                else:
                    # Skip this frame as it has "input" as the last folder
                    continue
                
            frame["file_path"] = abs_path

            # Append the extra row [0.0, 0.0, 0.0, 1.0] to the transform_matrix:
            # Make sure we don't accidentally add it twice.
            matrix = frame.get("transform_matrix", [])
            if len(matrix) == 3:
                matrix.append([0.0, 0.0, 0.0, 1.0])
            frame["transform_matrix"] = matrix

            # Add the updated frame to our master list of frames
            all_frames.append(frame)

    # Construct the final merged dictionary:
    merged_transforms = dict(top_level_intrinsics)
    # Optionally, set an orientation override or any other top-level keys:
    # merged_transforms["orientation_override"] = "none"
    merged_transforms["frames"] = all_frames

    # Write the merged transforms to the root folder
    with open(output_path, "w") as out_f:
        json.dump(merged_transforms, out_f, indent=4)

    print(f"Successfully created merged transforms file at: {output_path}")
    print(f"Included {len(all_frames)} frames (excluded frames with images in 'input' folders)")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Merge transforms.json files from subfolders into a single file.")
    parser.add_argument("--root", "-r", type=str, required=True, 
                        help="Path to the root directory containing subfolders with transforms.json files")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the merge_transforms function with the provided root directory
    merge_transforms(args.root)