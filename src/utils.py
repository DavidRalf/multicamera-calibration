import os
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        normalized_image = image * 0
    return normalized_image


def get_number_from_image_name(image_path):
    match = re.search(r'IMG_(\d{4})_\d', image_path)
    if match:
        return match.group(1)
    else:
        raise ValueError("The image name does not contain a valid number pattern.")


def extract_image_name(file_path):
    file_name_with_extension = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]

    match = re.search(r'(IMG_\d+(_\d+)?)', file_name_without_extension)
    if match:
        return match.group(1)
    else:
        return None


def extract_all_image_names(file_paths):
    image_names = []
    for file_path in file_paths:
        name = extract_image_name(file_path)
        if name is not None:
            image_names.append(name)
    return image_names


def make_rgb_composite_from_original(thecapture, irradiance_list, output_png_path):
    undistorted_images = thecapture.undistorted_reflectance(irradiance_list)

    band_names_lower = thecapture.band_names_lower()
    rgb_band_indices = [band_names_lower.index('red'),
                        band_names_lower.index('green'),
                        band_names_lower.index('blue')]

    first_band_data = undistorted_images[rgb_band_indices[0]].data
    height, width = first_band_data.shape

    im_display = np.zeros((height, width, 3), dtype=np.float32)

    for i, band_index in enumerate(rgb_band_indices):
        band_data = np.array(undistorted_images[band_index].data)
        im_min = np.percentile(band_data.flatten(), 0.5)
        im_max = np.percentile(band_data.flatten(), 99.5)

        im_display[:, :, i] = (band_data - im_min) / (im_max - im_min) if im_max > im_min else 0

    rgb = np.clip(im_display, 0, 1)

    plt.figure(figsize=(16, 13))
    plt.imshow(rgb)
    plt.title("RGB Composite from Original Images")
    plt.axis('off')

    plt.imsave(output_png_path, rgb, format='png')
    print(f"Saved RGB composite as {output_png_path}.")

    plt.show()


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

    bands_data = {}

    for band_name, data in calib.items():
        K = np.array(data["cameraMatrix"])
        D = np.array(data["distCoeffs"])

        bands_data[band_name] = {
            "cameraMatrix": K,
            "distCoeffs": D
        }

        if "rotation" in data:
            R = np.array(data["rotation"])
            bands_data[band_name]["rotation"] = R

        if "translation" in data:
            T = np.array(data["translation"])
            bands_data[band_name]["translation"] = T

    return bands_data


def get_band_data(bands_data, identifier):
    if isinstance(identifier, int):
        if identifier < len(bands_data):
            band_name = list(bands_data.keys())[identifier]
            return bands_data[band_name]
        else:
            raise IndexError("Index out of range.")
    elif isinstance(identifier, str):
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
        img1 = cv2.circle(img1, p1, 5, color, -1)
        img2 = cv2.circle(img2, p2, 5, color, -1)
    return img1, img2


def str_to_bool(value):
    if value.lower() in ['true', '1', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'no']:
        return False
    else:
        raise ValueError(f"Boolean value expected, got '{value}'")


def write_calib(filename, K, D, R, P):
    with open(filename, "w") as file:
        yaml.safe_dump(
            {
                "cameraMatrix": K.tolist(),
                "distCoeffs": D.tolist(),
                "rotation": R.tolist(),
                "projectionMatrix": P.tolist(),
            },
            file,
            default_flow_style=None,
        )
    print("wrote:", filename)


def check_stereo_yaml_files(directory_path):
    path = Path(directory_path)
    required_files = ["SAMSON1_SAMSON2_stereo.yaml", "SAMSON2_SAMSON1_stereo.yaml"]
    missing_files = [f for f in required_files if not (path / f).is_file()]
    return not missing_files


def get_micasense_number_from_basler_number(image_number):
    return f"{int(image_number) - 1:0{len(image_number)}d}"[2:]


def validate_directory(path, name):
    if not path.is_dir():
        raise ValueError(f"The provided {name} path '{path}' is not a valid directory.")


def find_depth_map_file(directory, number):
    pattern = re.compile(r'(\d+)_rect\.npy$')
    for filename in os.listdir(directory):
        if pattern.match(filename):
            file_number = pattern.match(filename).group(1)
            if int(file_number) == number:
                return os.path.join(directory, filename)
    return None
