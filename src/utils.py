import cv2
import numpy
import skimage
from pathlib import Path
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import yaml
import imageprocessing.micasense.imageutils as imageutils
from skimage.transform import warp, matrix_transform, resize, FundamentalMatrixTransform, estimate_transform, \
    ProjectiveTransform


def extract_image_name(file_path):
    """
    Extracts the image name from a file path.

    Parameters:
        file_path (str): The file path from which to extract the image name.

    Returns:
        str: The extracted image name (e.g., 'IMG_42', 'IMG_00042', etc.) or None if not found.
    """
    # Extract the filename without extension
    file_name_with_extension = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]

    # Use regex to find the pattern "IMG_" followed by any number of digits
    match = re.search(r'(IMG_\d+)', file_name_without_extension)
    if match:
        return match.group(1)  # Return the matched pattern (e.g., 'IMG_42', 'IMG_00042', 'IMG_00000042')
    else:
        return None  # Return None if no match is found


def make_rgb_composite_from_original(thecapture, irradiance_list, output_png_path):
    """
    Creates an RGB composite image from a Capture object using undistorted reflectance images.

    Parameters:
        thecapture (Capture): A Capture object containing the image data.
        irradiance_list (list): List of irradiance values corresponding to EO images.
        output_png_path (str): Output file path for the saved RGB composite image.
    """
    # Compute undistorted reflectance images
    undistorted_images = thecapture.undistorted_reflectance(irradiance_list)

    # Retrieve band indices for RGB composite
    band_names_lower = thecapture.band_names_lower()
    rgb_band_indices = [band_names_lower.index('red'),
                        band_names_lower.index('green'),
                        band_names_lower.index('blue')]

    # Assuming all bands have the same dimensions, get the dimensions from the first band
    first_band_data = undistorted_images[rgb_band_indices[0]].data  # Assuming each image has a `data` attribute
    height, width = first_band_data.shape

    # Initialize an empty array for normalized images
    im_display = np.zeros((height, width, 3), dtype=np.float32)

    # Normalize the RGB bands for true color
    for i, band_index in enumerate(rgb_band_indices):
        band_data = np.array(undistorted_images[band_index].data)  # Convert memoryview to NumPy array
        im_min = np.percentile(band_data.flatten(), 0.5)  # Modify these percentiles to adjust contrast
        im_max = np.percentile(band_data.flatten(), 99.5)  # Good values for many images

        # Prevent division by zero
        im_display[:, :, i] = (band_data - im_min) / (im_max - im_min) if im_max > im_min else 0

    # Create RGB composite
    rgb = np.clip(im_display, 0, 1)  # Ensure values are clipped to [0, 1]

    # Display the RGB composite
    plt.figure(figsize=(16, 13))
    plt.imshow(rgb)
    plt.title("RGB Composite from Original Images")
    plt.axis('off')

    # Save the RGB composite as a PNG file
    plt.imsave(output_png_path, rgb, format='png')
    print(f"Saved RGB composite as {output_png_path}.")

    # Show the plot
    plt.show()


# Example usage
# Assuming `thecapture` is already created from a filelist and `irradiance_list` is available
# make_rgb_composite_from_original(thecapture, irradiance_list, output_png_path='output/rgb_composite_original.png"')

def read_basler_calib(file):
    with open(file, "r") as file:
        calib = yaml.safe_load(file)
        K = np.array(calib["cameraMatrix"])
        D = np.array(calib["distCoeffs"])
        P, R = None, None
        if "rotation" in calib:
            R = np.array(calib["rotation"])
        if "projectionMatrix" in calib:
            P = np.array(calib["projectionMatrix"])
        return K, D, P, R


def read_micasense_calib(file):
    with open(file, "r") as f:
        calib = yaml.safe_load(f)

    # Create a dictionary to hold the calibration data
    bands_data = {}

    # Iterate over each band in the calibration data
    for band_name, data in calib.items():
        K = np.array(data["cameraMatrix"])
        D = np.array(data["distCoeffs"])

        # Initialize the band data
        bands_data[band_name] = {
            "cameraMatrix": K,
            "distCoeffs": D
        }

        # Add rotation and translation if present
        if "rotation" in data:
            R = np.array(data["rotation"])
            bands_data[band_name]["rotation"] = R

        if "translation" in data:
            T = np.array(data["translation"])
            bands_data[band_name]["translation"] = T

    return bands_data


def get_band_data(bands_data, identifier):
    if isinstance(identifier, int):  # If index is provided
        if identifier < len(bands_data):
            band_name = list(bands_data.keys())[identifier]
            return bands_data[band_name]
        else:
            raise IndexError("Index out of range.")
    elif isinstance(identifier, str):  # If band name is provided
        if identifier in bands_data:
            return bands_data[identifier]
        else:
            raise KeyError(f"Band '{identifier}' not found.")
    else:
        raise TypeError("Identifier must be either an index (int) or a band name (str).")


def undistort(img, K, D, R=None, P=None):
    if P is None:
        P = K.copy()
    if R is None:
        R = np.eye(3, 3, dtype=np.float32)
    image_size = (img.shape[1], img.shape[0])
    map1, map2 = cv2.initUndistortRectifyMap(K, D, R, P, image_size, cv2.CV_16SC2)
    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return undistorted_img

def draw_lines(img1, img2, lines, pts1, pts2):
    """img1 - image on which we draw the epilines for the points in img2 lines - corresponding epilines"""
    r, c, _ = img1.shape
    img1 = img1.copy()
    img2 = img2.copy()
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        p1 = pt1[0].astype(np.int32)
        p2 = pt2[0].astype(np.int32)
        # print(p1, p2)
        img1 = cv2.circle(img1, p1, 5, color, -1)
        img2 = cv2.circle(img2, p2, 5, color, -1)
    return img1, img2

