import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import micasense.capture as capture
import src.utils as utils
from micasense.registered_micasense import RegisteredMicasense


def pixel_to_3d(cam_positions, depth, intrinsic):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # = np.ones_like(depth)
    Z = depth
    x = cam_positions[..., 0]
    y = cam_positions[..., 1]
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)


def project_3d_to_2d(new_3d_position, intrinsic, extrinsic):
    point_3d = np.pad(new_3d_position, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)
    point_cam = (extrinsic[np.newaxis, np.newaxis] @ point_3d[..., np.newaxis])[..., 0]
    x = (point_cam[..., 0] * intrinsic[0, 0] / point_cam[..., 2]) + intrinsic[0, 2]
    y = (point_cam[..., 1] * intrinsic[1, 1] / point_cam[..., 2]) + intrinsic[1, 2]
    return np.round(x).astype(np.int32), np.round(y).astype(np.int32)


def register(thecapture, depth_map, micasense_calib, basler_cameraMatrix, image_names):
    if thecapture.dls_present():
        # reflectance
        irradiance_list = thecapture.dls_irradiance() + [0]
    else:
        # radiance
        irradiance_list = None

    height, width = depth_map.shape
    images = thecapture.undistorted_reflectance(irradiance_list)
    registered_bands = []

    for i, image in enumerate(images):
        registered_band = np.zeros((height, width))

        band_data = utils.get_band_data(micasense_calib, i)

        rotation = band_data['rotation']
        R = np.array(rotation)
        translation = band_data['translation']
        t = np.array(translation).reshape(3, 1)

        micasense_extrinsic = np.zeros((4, 4))
        micasense_extrinsic[:3, :3] = R
        micasense_extrinsic[:3, 3] = t.flatten()
        micasense_extrinsic[3, 3] = 1

        micasense_intrinsic = np.array(band_data['cameraMatrix'])

        cam_positions = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
        new_3d_position = pixel_to_3d(cam_positions, depth_map, basler_cameraMatrix)

        new_x, new_y = project_3d_to_2d(new_3d_position, micasense_intrinsic, micasense_extrinsic)

        within_image_x = np.logical_and(np.greater_equal(new_x, 0), np.less(new_x, image.shape[1]))
        within_image_y = np.logical_and(np.greater_equal(new_y, 0), np.less(new_y, image.shape[0]))
        within_image = np.logical_and(within_image_x, within_image_y)
        valid_new_x = new_x[within_image]
        valid_new_y = new_y[within_image]

        values = image[valid_new_y, valid_new_x]

        x_positions = cam_positions[..., 0][within_image]
        y_positions = cam_positions[..., 1][within_image]

        registered_band[y_positions, x_positions] = values
        registered_bands.append(registered_band)
    print("finished registration")
    return RegisteredMicasense(registered_bands, image_names)


def set_intrinsic(thecapture, micasense_calib):
    for i, image in enumerate(thecapture.images):
        band_data = utils.get_band_data(micasense_calib, i)

        camera_matrix = band_data['cameraMatrix']
        dist_coeffs = band_data['distCoeffs']

        focal_length = camera_matrix[0][0] / image.focal_plane_resolution_px_per_mm[0]
        principal_point = (camera_matrix[0][2] / image.focal_plane_resolution_px_per_mm[0],
                           camera_matrix[1][2] / image.focal_plane_resolution_px_per_mm[0])
        distortion_parameters = dist_coeffs[0]
        distortion_parameters = [distortion_parameters[0], distortion_parameters[1], distortion_parameters[4],
                                 distortion_parameters[2], distortion_parameters[3]]
        image.focal_length = focal_length
        image.principal_point = principal_point
        image.distortion_parameters = distortion_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Registration of the micasense images with depth information.')
    parser.add_argument('depth_path', type=str, help='Path to the directory containing depth information.')
    parser.add_argument('micasense_path', type=str, help='Path to the directory containing Micasense images')
    parser.add_argument('--output', type=str, nargs='?', default="../output/depth_matching", help='Path to save images')
    parser.add_argument('--image_number', type=str, nargs='?', default=None,
                        help='Image number based on the basler numbers with leading zeros (e.g., 000001)')
    parser.add_argument('--basler_size', type=tuple, nargs='?', default=(5328, 4608),
                        help='Original Basler image size (width, height)')

    args = parser.parse_args()

    depth_path = Path(args.depth_path)
    micasense_path = Path(args.micasense_path)
    image_number = args.image_number
    basler_size = args.basler_size
    output = args.output

    if image_number is None:
        micasense_image_number = "*"
    else:
        micasense_image_number = utils.get_micasense_number_from_basler_number(image_number)

    image_names = sorted(list(micasense_path.glob(f'IMG_{micasense_image_number}_*.tif')))
    image_names = [x.as_posix() for x in image_names]

    if len(image_names) < 6:
        print(f"Warning: Expected at least 6 images, but found {len(image_names)}. Please check the image files.")
        sys.exit()

    micasense_calib = utils.read_micasense_calib("../calib/micasense_calib.yaml")
    cal_samson_1 = utils.read_basler_calib("../calib/SAMSON1_SAMSON2_stereo.yaml")
    K_L, D_L, P_L, _ = cal_samson_1

    for i in range(0, len(image_names), 6):
        batch = image_names[i:i + 6]
        depth_map_number = int(utils.get_number_from_image_name(batch[0])) + 1
        file_to_depth_map = utils.find_depth_map_file(depth_path, depth_map_number)

        if file_to_depth_map is None:
            print(f"No depth map found for the micasense images {depth_map_number - 1}")
            continue

        depth_map = np.load(file_to_depth_map)
        depth_map_resized = cv2.resize(depth_map, basler_size, interpolation=cv2.INTER_LINEAR)

        thecapture = capture.Capture.from_filelist(batch)

        set_intrinsic(thecapture, micasense_calib)

        file_names = utils.extract_all_image_names(batch)

        registered_bands = register(thecapture, depth_map_resized, micasense_calib, P_L, file_names)
        registered_bands.save_images(output)

        registered_bands = None
        depth_map_resized = None
        thecapture = None
        print("finished a set")
    print("Registration complete")
