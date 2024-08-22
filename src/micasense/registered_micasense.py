import os

import numpy as np
from osgeo import gdal

import src.utils as utils


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
        self.images = []
        self.load_images(images)
        self.file_names = file_names if file_names is not None else []

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
        """Saves each image to the specified directory with the corresponding file name."""
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
