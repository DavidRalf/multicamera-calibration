import os

import cv2
import imageio
import numpy as np
from osgeo import gdal

import src.utils as utils
from micasense import imageutils


def crop(image_stack):
    non_black_mask = np.all(image_stack > 0, axis=-1)

    valid_coords = np.argwhere(non_black_mask)
    top_left = valid_coords.min(axis=0)
    bottom_right = valid_coords.max(axis=0) + 1

    cropped_image = image_stack[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    height, width, _ = cropped_image.shape

    row_threshold = int(0.9 * width)
    col_threshold = int(0.9 * height)

    non_zero_counts_rows = np.sum(np.all(cropped_image > 0, axis=-1), axis=1)
    non_zero_counts_cols = np.sum(np.all(cropped_image > 0, axis=-1), axis=0)

    top_left_row = np.argmax(non_zero_counts_rows >= row_threshold)
    top_left_col = np.argmax(non_zero_counts_cols >= col_threshold)
    top_left = (top_left_row, top_left_col)

    bottom_right_row = height - np.argmax(
        non_zero_counts_rows[::-1] >= row_threshold)
    bottom_right_col = width - np.argmax(
        non_zero_counts_cols[::-1] >= col_threshold)
    bottom_right = (bottom_right_row, bottom_right_col)

    if top_left is not None and bottom_right is not None:
        final_cropped_image = cropped_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        return final_cropped_image
    else:
        print("Error in cropping return original stack")
        return image_stack


class RegisteredMicasense:
    BAND_NAMES = {
        0: 'Blue',
        1: 'Green',
        2: 'Red',
        3: 'NIR',
        4: 'Red Edge',
        5: 'Panchro'
    }

    def __init__(self, images, file_names=None):
        self.rgb_composite_crop = None
        self.rgb_composite_enhanced_crop = None
        self.images = []
        self.load_images(images)
        self.file_names = file_names if file_names is not None else []

        self.rgb_composite = None
        self.rgb_composite_crop = None
        self.rgb_composite_enhanced = None
        self.rgb_composite_enhanced_crop = None

    def __repr__(self):
        return f'RegisteredMicasense with {len(self.images)} images.'

    def load_images(self, images):
        if isinstance(images, np.ndarray) and images.ndim == 3:
            self._load_from_3d_array(images)
        elif isinstance(images, list):
            self._load_from_list(images)
        else:
            raise ValueError("Input must be a list of images, a 3D numpy array, or a list of file paths to .tif files.")

    def _load_from_3d_array(self, array):
        if array.ndim != 3:
            raise ValueError("Input 3D array must have three dimensions (height, width, channels).")
        self.images = [array[:, :, i] for i in range(array.shape[2])]

    def _load_from_list(self, image_list):
        if all(isinstance(img, str) and img.endswith('.tif') for img in image_list):
            image_list = sorted(image_list)
            self.images = [self._load_image(img_path) for img_path in image_list]
            self.file_names = utils.extract_all_image_names(image_list)
        elif all(isinstance(img, np.ndarray) and img.ndim == 2 for img in image_list):
            self.images = image_list
        else:
            raise ValueError("List must contain either image paths (str) or 2D numpy arrays (ndarray).")

    def _load_image(self, path):

        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(f"Image at path '{path}' could not be loaded. Check the file path.")

        band = dataset.GetRasterBand(1)
        image_data = band.ReadAsArray().astype(np.float64)
        return image_data

    def get_image(self, index):

        if index < 0 or index >= len(self.images):
            raise IndexError("Index out of bounds.")
        return self.images[index]

    def get_band_name(self, index):
        if index not in self.BAND_NAMES:
            raise IndexError("Index out of bounds for band names.")
        return self.BAND_NAMES[index]

    def get_image_by_band_name(self, band_name):
        for index, name in self.BAND_NAMES.items():
            if name.lower() == band_name.lower():
                return self.get_image(index)
        raise ValueError(f"Band name '{band_name}' is not recognized.")

    def get_file_name_by_index(self, index):
        if index < 0 or index >= len(self.file_names):
            raise IndexError("Index out of bounds for file names.")
        return self.file_names[index]

    def get_file_name_by_band_name(self, band_name):
        for index, name in self.BAND_NAMES.items():
            if name.lower() == band_name.lower():
                return self.get_file_name_by_index(index)
        raise ValueError(f"Band name '{band_name}' is not recognized.")

    def save_images(self, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for index, image in enumerate(self.images):
            file_name = self.get_file_name_by_index(index) + ".tif"  # Append .tif
            file_path = os.path.join(output_directory, file_name)

            # Normalize the image for saving
            # image = utils.normalize_image(image)

            # Use GDAL to save the normalized image
            driver = gdal.GetDriverByName('GTiff')
            out_dataset = driver.Create(file_path, image.shape[1], image.shape[0], 1,
                                        gdal.GDT_Float64, options=['COMPRESS=DEFLATE'])

            out_band = out_dataset.GetRasterBand(1)
            out_band.WriteArray(image)
            out_band.FlushCache()
            out_dataset = None  # Close the dataset

    def get_rgb_indices(self):
        return [2, 1, 0]

    def make_rgb_composite(self, crop_image):
        rgb_band_indices = self.get_rgb_indices()

        band_red = self.get_image(rgb_band_indices[0])
        band_green = self.get_image(rgb_band_indices[1])
        band_blue = self.get_image(rgb_band_indices[2])

        rgb_stack = np.stack((band_red, band_green, band_blue), axis=-1)

        im_display = np.zeros(rgb_stack.shape, dtype=np.float32)
        non_zero_values = rgb_stack[rgb_stack > 0]
        im_min = np.percentile(non_zero_values.flatten(), 0.5)
        im_max = np.percentile(non_zero_values.flatten(), 99.5)

        for i, band_index in enumerate(rgb_band_indices):
            im_display[:, :, i] = imageutils.normalize(rgb_stack[:, :, i], im_min, im_max)
        if crop_image:
            self.rgb_composite_crop = crop(im_display)
        else:
            self.rgb_composite = im_display

    def make_rgb_enhancement(self, crop):
        if crop and self.rgb_composite_crop is None:
            self.make_rgb_composite(crop)
        elif not crop and self.rgb_composite is None:
            self.make_rgb_composite(crop)

        rgb_composite = self.rgb_composite_crop if crop else self.rgb_composite

        gaussian_rgb = cv2.GaussianBlur(rgb_composite, (9, 9), 10.0)
        gaussian_rgb[gaussian_rgb < 0] = 0
        gaussian_rgb[gaussian_rgb > 1] = 1
        unsharp_rgb = cv2.addWeighted(rgb_composite, 1.5, gaussian_rgb, -0.5, 0)
        unsharp_rgb[unsharp_rgb < 0] = 0
        unsharp_rgb[unsharp_rgb > 1] = 1
        # Apply a gamma correction to make the render appear closer to what our eyes would see
        gamma = 1.4
        gamma_corr_rgb = unsharp_rgb ** (1.0 / gamma)
        if crop:
            self.rgb_composite_enhanced_crop = gamma_corr_rgb
        else:
            self.rgb_composite_enhanced = gamma_corr_rgb

    def save_rgb_composite(self, crop=False, enhanced=False, output_directory="../output/comp"):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

            # Build the filename based on crop and enhancement flags
        file_name = os.path.join(
            output_directory,
            f"rgb_composite{'_enhanced' if enhanced else ''}{'_crop' if crop else ''}_{utils.get_number_from_image_name(self.file_names[0])}.png"
        )

        imageio.imwrite(file_name, (255 * self.get_rgb_composite(crop, enhanced)).astype('uint8'))

    def get_rgb_composite(self, crop=False, enhanced=False):
        if crop:
            return self._get_rgb_composite_crop() if not enhanced else self._get_rgb_composite_enhanced_crop()
        else:
            return self._get_rgb_composite() if not enhanced else self._get_rgb_composite_enhanced()

    def _get_rgb_composite(self):
        if self.rgb_composite is None:
            self.make_rgb_composite(False)
        return self.rgb_composite

    def _get_rgb_composite_crop(self):
        if self.rgb_composite_crop is None:
            self.make_rgb_composite(True)
        return self.rgb_composite_crop

    def _get_rgb_composite_enhanced(self):
        if self.rgb_composite_enhanced is None:
            self.make_rgb_enhancement(False)
        return self.rgb_composite_enhanced

    def _get_rgb_composite_enhanced_crop(self):
        if self.rgb_composite_enhanced_crop is None:
            self.make_rgb_enhancement(True)
        return self.rgb_composite_enhanced_crop
