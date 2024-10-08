import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import src.utils as utils
from evaluation import eval_utils
from evaluation.eval_utils import get_patches
from registration.registration_with_depth_matching import register


def load_depth_map(depth_path, base_number):
    """Load and resize the depth map based on the base number."""
    depth_image_name = f"{base_number + 1:06d}_rect.npy"  # Increment by 1 and format
    depth_map = np.load(depth_path.joinpath(depth_image_name))
    return cv2.resize(depth_map, (5328, 4608), interpolation=cv2.INTER_LINEAR)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate the registration method based on depth maps and Micasense images.'
    )
    parser.add_argument('micasense_path', type=str, help='Path to the directory containing Micasense images.')
    parser.add_argument('depth_path', type=str, help='Path to the directory containing depth maps.')
    parser.add_argument('name', type=str, help='Name for the result folder, used for organizing results.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory where results will be saved (default: output).')
    return parser.parse_args()


def main():
    timestamp1 = time.time()
    # Parse arguments
    args = parse_arguments()
    micasense_path = Path(args.micasense_path)
    depth_path = Path(args.depth_path)
    name = args.name
    output_dir = args.output_dir or f"../output/eval/{name}"

    eval_utils.create_output_directory(output_dir)

    # Load calibration and patch size data
    with open("../data/eval/images_to_eval.json", 'r') as json_file:
        data = json.load(json_file)

    with open("../data/eval/data.json", 'r') as json_file:
        data_patch_size = json.load(json_file)

    patch_size = data_patch_size["Patch size"]

    micasense_calib, P_L = eval_utils.load_calibration_data()

    metrics = {}
    for batch_name, images in data.items():
        base_number = int(images[0].split('_')[1])  # Assuming format "IMG_xxxx_x"
        depth_map_resized = load_depth_map(depth_path, base_number)
        thecapture, image_names = eval_utils.load_the_capture(images, micasense_calib, micasense_path)
        file_names = utils.extract_all_image_names(image_names)
        registered_bands = register(thecapture, depth_map_resized, micasense_calib, P_L, file_names)
        stack = registered_bands.get_stack(True)
        eval_utils.save_stack_to_disk(stack, output_dir, batch_name)
        band_names = [registered_bands.get_band_name(i) for i in range(stack.shape[2])]
        batch_patches = get_patches(stack, patch_size)
        eval_utils.store_metrics(metrics, batch_name, batch_patches, band_names)
    # Calculate statistics for metrics
    metrics = eval_utils.transform_metric(metrics)
    statistics = eval_utils.calculate_statistics(metrics)
    # Save metrics and statistics
    eval_utils.save_results_to_json(statistics, output_dir)
    eval_utils.save_results_to_pdf(statistics, output_dir)
    timestamp2 = time.time()
    print("This took %.2f seconds" % (timestamp2 - timestamp1))


if __name__ == "__main__":
    main()
