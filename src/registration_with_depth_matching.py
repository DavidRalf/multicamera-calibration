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
    point_cam = extrinsic @ point_3d
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
    images = thecapture.undistorted_reflectance(irradiance_list)
    registered_bands = []
    for i, image in enumerate(images):
        registered_band = np.zeros((height, width))
        if i == 3:
            break

        print(f"Register Image from Band {i + 1} ")
        band_data = utils.get_band_data(micasense_calib, i)
        rotation = band_data['rotation']
        R = np.array(rotation)
        translation = band_data['translation']
        t = np.array(translation).reshape(3, 1)

        micasense_extrinsic = np.zeros((4, 4))
        micasense_extrinsic[:3, :3] = R
        micasense_extrinsic[:3, 3] = t.flatten()
        micasense_extrinsic[3, 3] = 1

        print("help")
        micasense_intrinsic = np.array(band_data['cameraMatrix'])
        for y in range(height):
            print(f"y {y} finish {range(height)}")
            for x in range(width):
                print(f"x,y of depth: {x, y}")
                X, Y, Z = pixel_to_3d(x, y, depth_map, basler_cameraMatrix)
                print(f"x,y z : {X, Y, Z}")
                new_x, new_y = project_3d_to_2d(X, Y, Z, micasense_intrinsic, micasense_extrinsic)
                print(f" new_x, new_y : {new_x, new_y}")

                if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                    registered_band[y, x] = image[new_y, new_x]
        registered_bands.append(registered_band)
    return registered_bands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Registration of the micasense images with depth information.')
    parser.add_argument('depth_path', type=str, help='Path to the directory containing depth information.')
    parser.add_argument('micasense_path', type=str, help='Path to the directory containing Micasense images')
    parser.add_argument('image_number', type=str, nargs='?', default=None,
                        help='Image number based on the basler numbers with leading zeros (e.g., 000001)')

    args = parser.parse_args()

    depth_path = Path(args.depth_path)
    micasense_path = Path(args.micasense_path)
    image_number = args.image_number

    depth_map = np.load(depth_path.as_posix() + f"/{image_number}_rect.npy")
    print(depth_map)
    width = 5328
    height = 4608

    depth_map_resized = cv2.resize(depth_map, (width, height))

    result = int(image_number) - 1
    num_leading_zeros = len(image_number) - len(image_number.lstrip('0'))
    formatted_result = f"{result:0{len(image_number)}d}"
    micasense_image_number = formatted_result[2:]
    image_names = list(micasense_path.glob(f'IMG_{micasense_image_number}_*.tif'))
    image_names = [x.as_posix() for x in image_names]

    thecapture = capture.Capture.from_filelist(image_names)
    micasense_calib = utils.read_micasense_calib("src/micasense_calib.yaml")
    cal_samson_1 = utils.read_basler_calib("/media/david/T7/multispektral/20240416_calib/SAMSON1/SAMSON1.yaml")
    K_L, D_L, _, _ = cal_samson_1
    #for i, image in enumerate(thecapture.images):
    #print(f"Processing Band {i + 1} and setting calibrated parameters from micasense_calib.yaml")

    #band_data = utils.get_band_data(micasense_calib, i)
    #camera_matrix = band_data['cameraMatrix']
    #dist_coeffs = band_data['distCoeffs']

    #focal_length = camera_matrix[0][0] / image.focal_plane_resolution_px_per_mm[0]
    #principal_point = (camera_matrix[0][2], camera_matrix[1][2])
    #distortion_parameters = dist_coeffs[0]

    #image.focal_length = focal_length
    #image.principal_point = principal_point
    #image.distortion_parameters = distortion_parameters
    registered_band = register_image_with_depth(thecapture, depth_map_resized, micasense_calib, K_L)
    print(len(registered_band))

    band_names_lower = thecapture.band_names_lower()
    rgb_band_indices = [band_names_lower.index('red'),
                        band_names_lower.index('green'),
                        band_names_lower.index('blue')]

    first_band_data = registered_band[0]
    height, width = first_band_data.shape

    im_display = np.zeros((height, width, 3), dtype=np.float32)

    for i, registered_image in enumerate(registered_band):
        band_data = registered_image
        im_min = np.percentile(band_data.flatten(), 0.5)
        im_max = np.percentile(band_data.flatten(), 99.5)

        im_display[:, :, i] = (band_data - im_min) / (im_max - im_min) if im_max > im_min else 0

    rgb = np.clip(im_display, 0, 1)

    plt.figure(figsize=(16, 13))
    plt.imshow(rgb)
    plt.title("RGB Composite from Original Images")
    plt.axis('off')
    output_png_path = "output/rgb_composite_depth.png"

    plt.imsave(output_png_path, rgb, format='png')
    print(f"Saved RGB composite as {output_png_path}.")

    plt.show()
    registered_band = np.array(registered_band)
    registered_band = registered_band.transpose(2, 1, 0)
    print(registered_band.shape)
    thecapture._Capture__aligned_capture = registered_band
    thecapture._Capture__aligned_radiometric_pan_sharpened_capture = [None, registered_band]
    thecapture.save_capture_as_stack("test.tif", sort_by_wavelength=False, pansharpen=False)
    print(thecapture.band_names_lower())
