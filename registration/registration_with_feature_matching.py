# umwandlung des Jupyter Notebooks Alignment V2 in ein Python skript
import argparse
import os
import sys
from pathlib import Path

import numpy
import numpy as np
import skimage
from skimage.transform import ProjectiveTransform

import micasense.capture as capture
import src.utils as utils
from micasense.registered_micasense import RegisteredMicasense


def register(thecapture, version, save_warp_matrices, names, regenerate=True):
    cam_serial = thecapture.camera_serial
    warp_matrices_filename = '../output/warp/' + cam_serial + "_warp_matrices_SIFT.npy"
    if Path(warp_matrices_filename).is_file() and not regenerate:
        print("Found existing warp matrices for camera", cam_serial)
        load_warp_matrices = np.load(warp_matrices_filename, allow_pickle=True)
        loaded_warp_matrices = []
        for matrix in load_warp_matrices:
            transform = ProjectiveTransform(matrix=matrix.astype('float64'))
            loaded_warp_matrices.append(transform)
        print("Warp matrices successfully loaded.")

        warp_matrices_SIFT = loaded_warp_matrices
    else:
        print("No existing warp matrices found. Create them later.")
        warp_matrices_SIFT = False

    if thecapture.dls_present():
        img_type = 'reflectance'
        irradiance_list = thecapture.dls_irradiance() + [0]
    else:
        img_type = "radiance"
        irradiance_list = None

    if not warp_matrices_SIFT:
        print("Generating new warp matrices...")
        warp_matrices_SIFT = thecapture.SIFT_align_capture(min_matches=10)

    sharpened_stack, upsampled = thecapture.radiometric_pan_sharpened_aligned_capture(
        warp_matrices=warp_matrices_SIFT,
        irradiance_list=irradiance_list,
        img_type=img_type)

    if save_warp_matrices:
        working_wm = warp_matrices_SIFT
        temp_matrices = []
        for x in working_wm:
            if isinstance(x, numpy.ndarray):
                temp_matrices.append(x)
            if isinstance(x, skimage.transform._geometric.ProjectiveTransform):
                temp_matrices.append(x.params)

        os.makedirs("../output/warp", exist_ok=True)
        np.save(warp_matrices_filename, np.array(temp_matrices, dtype=object), allow_pickle=True)
        print("Saved to", Path(warp_matrices_filename).resolve())

    if version == "stack":
        registered_images = RegisteredMicasense(sharpened_stack, names)

    else:
        registered_images = RegisteredMicasense(upsampled, names)

    return registered_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Registration of the micasense images with feature matching.')
    parser.add_argument('image_path', type=str, help='Path to the directory containing images')
    parser.add_argument('save_warp_matrices', type=str, help='True or False for saving warp matrices')
    parser.add_argument('version', type=str, help='Stack or Upsampled')
    parser.add_argument('output', type=str, nargs='?', default="../output/feature_matching", help='Path to save images')
    parser.add_argument('image_number', type=str, nargs='?', default=None,
                        help='Image number with leading zeros (e.g., 0059)')

    args = parser.parse_args()

    image_path = Path(args.image_path)
    image_number = args.image_number
    version = args.version.lower()
    output = args.output
    save_warp_matrices = args.save_warp_matrices.lower() in ['true', '1', 't', 'y', 'yes']
    if image_number is None:
        image_number = "*"

    image_names = list(image_path.glob(f'IMG_{image_number}_*.tif'))
    image_names = sorted([x.as_posix() for x in image_names])

    if len(image_names) < 6:
        print(f"Warning: Expected at least 6 images, but found {len(image_names)}. Please check the image files.")
        sys.exit()

    for i in range(0, len(image_names), 6):
        batch = image_names[i:i + 6]
        if len(batch) < 6:
            print(f"Skipping incomplete batch: {batch}. Not enough images.")
            continue
        print(f"Processing batch: {batch}")

        thecapture = capture.Capture.from_filelist(batch)
        file_names = utils.extract_all_image_names(batch)
        registered_images = register(thecapture, version, save_warp_matrices,
                                     file_names)
        registered_images.save_images(output)
