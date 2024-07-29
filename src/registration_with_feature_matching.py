#umwandlung des Jupyter Notebooks Alignment V2 in ein Python skript
import os, glob
import sys
import argparse
import imageprocessing.micasense.capture as capture
from pathlib import Path
import utils as utils
import matplotlib.pyplot as plt
from skimage.transform import ProjectiveTransform
import numpy as np
import time

# Define the argument parser
parser = argparse.ArgumentParser(description='Process multispectral images.')

# Add positional arguments
parser.add_argument('image_path', type=str, help='Path to the directory containing images')
# Make image_number optional
parser.add_argument('image_number', type=str, nargs='?', default=None,
                    help='Image number with leading zeros (e.g., 0059)')
parser.add_argument('save_warp_matrices', type=str, help='True or False for saving warp matrices')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
image_path = Path(args.image_path)
image_number = args.image_number
save_warp_matrices = args.save_warp_matrices.lower() in ['true', '1', 't', 'y', 'yes']
if image_number is None:
    image_number = "*"
    print("test")
# Collect image names that match the pattern
image_names = list(image_path.glob(f'IMG_{image_number}_*.tif'))
image_names = [x.as_posix() for x in image_names]

# Check if we have enough pictures
if len(image_names) < 6:
    print(f"Warning: Expected at least 6 images, but found {len(image_names)}. Please check the image files.")
    sys.exit()  # Use sys.exit() to terminate the script if there are not enough images
batch = image_names[0:0 + 6]  # Take 6 images at a time
batch.sort()
thecapture = capture.Capture.from_filelist(batch)
cam_serial = thecapture.camera_serial
warp_matrices_filename = cam_serial + "_warp_matrices_SIFT.npy"
if Path('./' + warp_matrices_filename).is_file():
    print("Found existing warp matrices for camera", cam_serial)
    load_warp_matrices = np.load(warp_matrices_filename, allow_pickle=True)
    loaded_warp_matrices = []
    for matrix in load_warp_matrices:
        transform = ProjectiveTransform(matrix=matrix.astype('float64'))
        loaded_warp_matrices.append(transform)
    print("Warp matrices successfully loaded.")

    warp_matrices_SIFT = loaded_warp_matrices
else:
    print("No existing warp matrices found. Create them later in the notebook.")
    warp_matrices_SIFT = False
    warp_matrices = False
# Process images in batches of 6
for i in range(0, len(image_names), 6):
    batch = image_names[i:i + 6]  # Take 6 images at a time
    batch.sort()
    file_name = utils.extract_image_name(batch[0])
    thecapture = capture.Capture.from_filelist(batch)

    if thecapture.dls_present():
        img_type = 'reflectance'
        irradiance_list = thecapture.dls_irradiance() + [0]
        thecapture.plot_undistorted_reflectance(thecapture.dls_irradiance())
    else:
        img_type = "radiance"
        thecapture.plot_undistorted_radiance()
        irradiance_list = None

    utils.make_rgb_composite_from_original(thecapture, irradiance_list,"output/rgb_composite_original.png")
    st = time.time()
    if not warp_matrices_SIFT:
        print("Generating new warp matrices...")
        warp_matrices_SIFT = thecapture.SIFT_align_capture(min_matches=10)

    sharpened_stack, upsampled = thecapture.radiometric_pan_sharpened_aligned_capture(warp_matrices=warp_matrices_SIFT,
                                                                                      irradiance_list=irradiance_list,
                                                                                      img_type=img_type)

    # we can also use the Rig Relatives from the image metadata to do a quick, rudimentary alignment
    #     warp_matrices0=thecapture.get_warp_matrices(ref_index=5)
    #     sharpened_stack,upsampled = radiometric_pan_sharpen(thecapture,warp_matrices=warp_matrices0)

    print("Pansharpened shape:", sharpened_stack.shape)
    print("Upsampled shape:", upsampled.shape)
    # re-assign to im_aligned to match rest of code
    im_aligned = upsampled
    et = time.time()
    elapsed_time = et - st
    print('Alignment and pan-sharpening time:', int(elapsed_time), 'seconds')

    # set output name to unique capture ID, e.g. FWoNSvgDNBX63Xv378qs
    outputName = "output/" + file_name

    st = time.time()
    print("im_aligned")
    print(im_aligned)
    # in this example, we can export both a pan-sharpened stack and an upsampled stack
    # so you can compare them in GIS. In practice, you would typically only output the pansharpened stack
    thecapture.save_capture_as_stack(outputName + "-pansharpened.tif", sort_by_wavelength=False, pansharpen=True)
    thecapture.save_capture_as_stack(outputName + "-upsampled.tif", sort_by_wavelength=False, pansharpen=False)
    # Save each band to its own TIFF file
    #thecapture.save_each_band_as_tif(outputName+)

    et = time.time()
    elapsed_time = et - st
    print("Time to save stacks:", int(elapsed_time), "seconds.")
    #print(thecapture._Capture__aligned_radiometric_pan_sharpened_capture[0])
    # print(thecapture._Capture__aligned_radiometric_pan_sharpened_capture[0][0])
    thecapture = None
    break
