import numpy
import skimage
from pathlib import Path
import os
import re
import numpy as np
import matplotlib.pyplot as plt
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
# make_rgb_composite_from_original(thecapture, irradiance_list, output_png_path='rgb_composite_original.png')