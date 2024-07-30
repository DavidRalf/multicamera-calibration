import argparse
from pathlib import Path
import imageprocessing.micasense.capture as capture
import cv2
import numpy as np
import utils as utils
import matplotlib.pyplot as plt

names = {"Blue",
         "Green",
         "Red",
         "NIR",
         "Red edge",
         "Panchro"}


def pixel_to_3d(x, y, depth, intrinsic):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    Z = depth[y, x]
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return X, Y, Z


def project_3d_to_2d(X, Y, Z, intrinsic, extrinsic):
    point_3d = np.array([X, Y, Z, 1]).reshape(4, 1)
    point_cam = extrinsic @ point_3d  # Transform to camera coordinates
    x = (point_cam[0] * intrinsic[0, 0] / point_cam[2]) + intrinsic[0, 2]
    y = (point_cam[1] * intrinsic[1, 1] / point_cam[2]) + intrinsic[1, 2]
    return int(x), int(y)


def register_image_with_depth(thecapture, depth_map, micasense_calib, basler_cameraMatrix):
    if thecapture.dls_present():
        img_type = 'reflectance'
        irradiance_list = thecapture.dls_irradiance() + [0]
        thecapture.plot_undistorted_reflectance(thecapture.dls_irradiance())
    else:
        img_type = "radiance"
        thecapture.plot_undistorted_radiance()
        irradiance_list = None
    height, width = depth_map.shape
    registered_band = np.zeros((height, width))
    images = thecapture.undistorted_reflectance(irradiance_list)
    registered_bands = []
    for i, image in enumerate(images):
        registered_band = np.zeros((height, width))
        if i == 3:
            break
        print(f"Register Image from Band {i + 1} ")
        band_data = utils.get_band_data(micasense_calib, i)
        rotation = band_data['rotation']
        # Convert the rotation list to a NumPy array
        R = np.array(rotation)
        translation = band_data['translation']
        # Convert the translation list to a NumPy array
        t = np.array(translation).reshape(3, 1)
        # Create the extrinsic matrix
        micasense_extrinsic = np.zeros((4, 4))
        micasense_extrinsic[:3, :3] = R  # Assign the rotation matrix
        micasense_extrinsic[:3, 3] = t.flatten()  # Assign the translation vector
        micasense_extrinsic[3, 3] = 1  # Homogeneous coordinate
        #undistorted_image= image.undistorted_reflectance(irradiance_list)
        print("help")
        # Extract camera matrix from band_data
        micasense_intrinsic = np.array(band_data['cameraMatrix'])
        for y in range(height):
            print(f"y {y} finish {range(height)}")
            for x in range(width):
                X, Y, Z = pixel_to_3d(x, y, depth_map, basler_cameraMatrix)
                new_x, new_y = project_3d_to_2d(X, Y, Z, micasense_intrinsic, micasense_extrinsic)

                if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                    registered_band[y, x] = image[new_y, new_x]
        registered_bands.append(registered_band)
    return registered_bands


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Registration of the micasense images with depth information.')

    # Add positional arguments
    parser.add_argument('depth_path', type=str, help='Path to the directory containing depth information.')
    parser.add_argument('micasense_path', type=str, help='Path to the directory containing Micasense images')
    # Make image_number optional
    parser.add_argument('image_number', type=str, nargs='?', default=None,
                        help='Image number based on the basler numbers with leading zeros (e.g., 000001)')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    depth_path = Path(args.depth_path)
    micasense_path = Path(args.micasense_path)
    image_number = args.image_number
    "{image_number}_rect.png"
    # Load the Basler image using the original image_number
    depth_map = np.load(depth_path.as_posix() + f"/{image_number}_rect.npy")
    print(depth_map)
    width = 5328
    height = 4608
    # Resize the depth map to the new dimensions
    depth_map_resized = cv2.resize(depth_map, (width, height))

    result = int(image_number) - 1
    num_leading_zeros = len(image_number) - len(image_number.lstrip('0'))
    formatted_result = f"{result:0{len(image_number)}d}"
    micasense_image_number = formatted_result[2:]
    image_names = list(micasense_path.glob(f'IMG_{micasense_image_number}_*.tif'))
    image_names = [x.as_posix() for x in image_names]
    thecapture = capture.Capture.from_filelist(image_names)
    micasense_calib = utils.read_micasense_calib("src/micasense_calib.yaml")
    cal_samson_1 = utils.read_basler_calib("/media/david/T71/multispektral/20240416_calib/SAMSON1/SAMSON1.yaml")
    K_L, D_L, _, _ = cal_samson_1
    for i, image in enumerate(thecapture.images):
        print(f"Processing Band {i + 1} and setting calibrated parameters from micasense_calib.yaml")
        # Get the calibration data for the current band
        band_data = utils.get_band_data(micasense_calib, i)
        # Extract camera matrix and distortion coefficients
        camera_matrix = band_data['cameraMatrix']
        dist_coeffs = band_data['distCoeffs']
        # Calculate parameters directly from the extracted data
        focal_length = camera_matrix[0][0] / image.focal_plane_resolution_px_per_mm[0]  # fx
        principal_point = (camera_matrix[0][2], camera_matrix[1][2])  # (cx, cy)
        distortion_parameters = dist_coeffs[0]
        # Assign parameters to the image
        #image.focal_length = focal_length
    # image.principal_point = principal_point
    # image.distortion_parameters = distortion_parameters
    registered_band = register_image_with_depth(thecapture, depth_map_resized, micasense_calib, K_L)
    print(len(registered_band))
    # Compute undistorted reflectance images
    band_names_lower = thecapture.band_names_lower()
    rgb_band_indices = [band_names_lower.index('red'),
                        band_names_lower.index('green'),
                        band_names_lower.index('blue')]

    # Assuming all bands have the same dimensions, get the dimensions from the first band
    first_band_data = registered_band[0]  # Assuming each image has a `data` attribute
    height, width = first_band_data.shape

    # Initialize an empty array for normalized images
    im_display = np.zeros((height, width, 3), dtype=np.float32)

    # Normalize the RGB bands for true color
    for i, registered_image in enumerate(registered_band):
        band_data = registered_image  # Convert memoryview to NumPy array
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
    output_png_path = "output/rgb_composite_depth.png"
    # Save the RGB composite as a PNG file
    plt.imsave(output_png_path, rgb, format='png')
    print(f"Saved RGB composite as {output_png_path}.")

    # Show the plot
    plt.show()
    thecapture._Capture__aligned_capture = registered_band
    thecapture._Capture__aligned_radiometric_pan_sharpened_capture = [None, registered_band]
    thecapture.save_capture_as_stack("test.tif", sort_by_wavelength=False, pansharpen=False)
    print(thecapture.band_names_lower())
