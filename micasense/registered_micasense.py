import os

import cv2
import imageio
import numpy as np
from osgeo import gdal

import src.utils as utils
from micasense import imageutils


def crop(image_stack, eval=False):
    """
        Crop the input image stack to remove black borders and return the cropped image.

        Parameters:
            image_stack (np.ndarray): A 3D numpy array of images (height, width, channels).
            eval (bool): If True, return detailed cropping information.

        Returns:
            np.ndarray: The cropped image stack or the original stack if cropping fails.
            tuple: (original cropping coordinates, final cropping coordinates) if eval is True.
    """

    # Create a mask where all pixels are non-black
    non_black_mask = np.all(image_stack > 0, axis=-1)

    # Get the coordinates of valid (non-black) pixels
    valid_coords = np.argwhere(non_black_mask)

    # Determine the top-left and bottom-right corners of the bounding box
    top_left = valid_coords.min(axis=0)
    bottom_right = valid_coords.max(axis=0) + 1  # +1 to include the bottom/right edge

    # Crop the image stack using the bounding box
    cropped_image = image_stack[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    height, width, _ = cropped_image.shape

    # Define thresholds for rows and columns to determine final cropping
    row_threshold = int(0.9 * width)
    col_threshold = int(0.9 * height)

    # Count non-zero pixels along rows and columns
    non_zero_counts_rows = np.sum(np.all(cropped_image > 0, axis=-1), axis=1)
    non_zero_counts_cols = np.sum(np.all(cropped_image > 0, axis=-1), axis=0)

    # Determine the final cropping coordinates
    top_left_row = np.argmax(non_zero_counts_rows >= row_threshold)
    top_left_col = np.argmax(non_zero_counts_cols >= col_threshold)
    top_left_final = (top_left_row, top_left_col)

    bottom_right_row = height - np.argmax(non_zero_counts_rows[::-1] >= row_threshold)
    bottom_right_col = width - np.argmax(non_zero_counts_cols[::-1] >= col_threshold)
    bottom_right_final = (bottom_right_row, bottom_right_col)

    # Return the cropped image or the original stack based on the eval flag
    if eval:
        final_cropped_image = cropped_image[top_left_final[0]:bottom_right_final[0],
                              top_left_final[1]:bottom_right_final[1]]
        return final_cropped_image, ((top_left, bottom_right), (top_left_final, bottom_right_final))

    if top_left is not None and bottom_right is not None:
        final_cropped_image = cropped_image[top_left_final[0]:bottom_right_final[0],
                              top_left_final[1]:bottom_right_final[1]]
        return final_cropped_image
    else:
        print("Error in cropping, returning original stack")
        return image_stack


class RegisteredMicasense:
    """
       A class to handle Registered Micasense RedEdge-P images.

       Attributes:
           BAND_NAMES (dict): A dictionary mapping band indices to names.
           BAND_ALIASES (dict): A dictionary mapping common names to canonical band names.
    """

    BAND_NAMES = {
        0: 'Blue',
        1: 'Green',
        2: 'Red',
        3: 'NIR',
        4: 'Red Edge',
        5: 'Panchro'
    }

    BAND_ALIASES = {
        'b': 'Blue', 'blue': 'Blue',
        'g': 'Green', 'green': 'Green',
        'r': 'Red', 'red': 'Red',
        'nir': 'NIR',
        're': 'Red Edge', 'red edge': 'Red Edge',
        'p': 'Panchro', 'pan': 'Panchro', 'panchro': 'Panchro'
    }

    def __init__(self, images, file_names=None):
        """
            Initialize the RegisteredMicasense class.

            Parameters:
                images (np.ndarray or list): A 3D numpy array or list of image paths.
                file_names (list): Optional list of filenames corresponding to the images.
        """
        self.rgb_composite_crop = None
        self.rgb_composite_enhanced_crop = None
        self.images = []
        self._load_images(images)
        self.file_names = file_names if file_names is not None else []

        # Initialize composite images
        self.rgb_composite = None
        self.rgb_composite_crop = None
        self.rgb_composite_enhanced = None
        self.rgb_composite_enhanced_crop = None

    def __repr__(self):
        """Return a string representation of the RegisteredMicasense object."""
        return f'RegisteredMicasense with {len(self.images)} images.'

    def _load_images(self, images):
        """
            Load images into the RegisteredMicasense object.

            Parameters:
                images (np.ndarray or list): Images to be loaded.
        """
        if isinstance(images, np.ndarray) and images.ndim == 3:
            self._load_from_3d_array(images)
        elif isinstance(images, list):
            self._load_from_list(images)
        else:
            raise ValueError("Input must be a list of images, a 3D numpy array, or a list of file paths to .tif files.")

    def _load_from_3d_array(self, array):
        """Load images from a 3D numpy array."""
        if array.ndim != 3:
            raise ValueError("Input 3D array must have three dimensions (height, width, channels).")
        self.images = [array[:, :, i] for i in range(array.shape[2])]

    def _load_from_list(self, image_list):
        """
            Load images from a list of file paths or 2D arrays.

            Parameters:
                image_list (list): List of image file paths or 2D numpy arrays.
        """
        if all(isinstance(img, str) and img.endswith('.tif') for img in image_list):
            image_list = sorted(image_list)
            self.images = [self._load_image(img_path) for img_path in image_list]
            self.file_names = utils.extract_all_image_names(image_list)
        elif all(isinstance(img, np.ndarray) and img.ndim == 2 for img in image_list):
            self.images = image_list
        else:
            raise ValueError("List must contain either image paths (str) or 2D numpy arrays (ndarray).")

    def _load_image(self, path):
        """
            Load a single image using GDAL.

            Parameters:
                path (str): Path to the image file.

            Returns:
                np.ndarray: Loaded image as a 2D numpy array.
        """
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(f"Image at path '{path}' could not be loaded. Check the file path.")

        band = dataset.GetRasterBand(1)
        image_data = band.ReadAsArray().astype(np.float64)
        return image_data

    def get_image(self, index):
        """
            Retrieve an image by index or band name.

            Parameters:
                index (int or str): Index of the image or band name.

            Returns:
                np.ndarray: The requested image.
        """
        if isinstance(index, str):
            return self._get_image_by_band_name(index)

        return self._get_image_by_index(index)

    def get_stack(self, cropping=False, images=None):
        """
            Retrieve a stack of images.

            Parameters:
                cropping (bool): If True, crop the stack.
                images (list): List of indices or band name for the images to include in the stack.

            Returns:
                np.ndarray: The image stack (cropped or not).
        """
        if images is None:
            images = list(range(len(self.images)))
        stack = [self.get_image(img) for img in images]
        image_stack = np.stack(stack, axis=-1)

        return crop(image_stack) if cropping else image_stack

    def get_band_index(self, band_name):
        """
            Get the index of a band by its name.

            Parameters:
                band_name (str): The name of the band.

            Returns:
                int: The index of the band.

            Raises:
                ValueError: If the band name is not recognized.
        """
        band_name = band_name.lower()
        canonical_name = self.BAND_ALIASES.get(band_name, band_name)
        for index, name in self.BAND_NAMES.items():
            if name == canonical_name:
                return index
        raise ValueError(f"Band name '{band_name}' is not recognized.")

    def get_band_name(self, index):
        """
            Get the name of a band by its index.

            Parameters:
                index (int): The index of the band.

            Returns:
                str: The name of the band.

            Raises:
                IndexError: If the index is out of bounds.
        """
        if index not in self.BAND_NAMES:
            raise IndexError("Index out of bounds for band names.")
        return self.BAND_NAMES[index]

    def _get_image_by_index(self, index):
        """
            Get an image by its index.

            Parameters:
                index (int): The index of the image.

            Returns:
                np.ndarray: The requested image.

            Raises:
                IndexError: If the index is out of bounds.
        """
        if not (0 <= index < len(self.images)):
            raise IndexError("Index out of bounds.")
        return self.images[index]

    def _get_image_by_band_name(self, band_name):
        """
            Get an image by its band name.

            Parameters:
                band_name (str): The name of the band.

            Returns:
                np.ndarray: The requested image.
        """
        band_name = band_name.lower()

        canonical_name = self.BAND_ALIASES.get(band_name, band_name)

        for index, name in self.BAND_NAMES.items():
            if name == canonical_name:
                return self.get_image(index)

        raise ValueError(f"Band name '{band_name}' is not recognized.")

    def get_file_name_by_index(self, index):
        """
            Retrieve the file name corresponding to the given index.

            Parameters:
                index (int): The index of the file name to retrieve.

            Returns:
                str: The file name corresponding to the specified index.

            Raises:
                IndexError: If the index is out of bounds for the file names list.
          """
        if index < 0 or index >= len(self.file_names):
            raise IndexError("Index out of bounds for file names.")
        return self.file_names[index]

    def get_file_name_by_band_name(self, band_name):
        """
            Retrieve the file name corresponding to a specified band name.

            Parameters:
                band_name (str): The name of the band for which to retrieve the file name.

            Returns:
                str: The file name corresponding to the specified band name.

            Raises:
                ValueError: If the band name is not recognized.
        """
        for index, name in self.BAND_NAMES.items():
            if name.lower() == band_name.lower():
                return self.get_file_name_by_index(index)
        raise ValueError(f"Band name '{band_name}' is not recognized.")

    def save_images(self, output_directory):
        """
            Save the loaded images to the specified output directory in TIFF format.

            Parameters:
                output_directory (str): The directory where the images will be saved.

            Raises:
                Exception: If there is an error in saving the images.
        """
        # Check if the output directory exists; if not, create it
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Iterate over the list of images and their corresponding indices
        for index, image in enumerate(self.images):
            # Construct the file name for the image by getting the corresponding file name
            # and appending the ".tif" extension
            file_name = self.get_file_name_by_index(index) + ".tif"

            # Create the full file path by joining the output directory and the file name
            file_path = os.path.join(output_directory, file_name)

            # Uncomment the following line if image normalization is needed before saving
            # image = utils.normalize_image(image)

            # Use GDAL to create a new TIFF dataset for saving the image
            driver = gdal.GetDriverByName('GTiff')
            out_dataset = driver.Create(file_path, image.shape[1], image.shape[0], 1,
                                        gdal.GDT_Float64, options=['COMPRESS=DEFLATE'])

            # Get the first raster band of the output dataset for writing the image data
            out_band = out_dataset.GetRasterBand(1)

            # Write the image array to the raster band
            out_band.WriteArray(image)

            # Flush the cache to ensure all data is written to disk
            out_band.FlushCache()

            # Close the dataset to free up resources
            out_dataset = None  # Close the dataset

    def _get_rgb_indices(self):
        """
            Retrieve the indices for the RGB bands in the image stack.

            Returns:
                list: A list containing the indices of the Red, Green, and Blue bands.
        """
        return [2, 1, 0]  # Return the indices for the Red, Green, and Blue bands respectively

    def _make_rgb_composite(self, crop_image):
        """
            Create an RGB composite image from the red, green, and blue bands.

            Parameters:
                crop_image (bool): If True, the resulting composite image will be cropped to remove any black borders.
        """
        # Retrieve the indices for the RGB bands
        rgb_band_indices = self._get_rgb_indices()

        # Get the individual RGB band images using their indices
        band_red = self.get_image(rgb_band_indices[0])  # Red band
        band_green = self.get_image(rgb_band_indices[1])  # Green band
        band_blue = self.get_image(rgb_band_indices[2])  # Blue band

        # Stack the individual bands into a single 3D array
        rgb_stack = np.stack((band_red, band_green, band_blue), axis=-1)

        # Initialize an array to store the normalized image display
        im_display = np.zeros(rgb_stack.shape, dtype=np.float32)

        # Extract non-zero values from the RGB stack for normalization
        non_zero_values = rgb_stack[rgb_stack > 0]

        # Calculate the minimum and maximum pixel values for normalization
        im_min = np.percentile(non_zero_values.flatten(), 0.5)  # 0.5 percentile
        im_max = np.percentile(non_zero_values.flatten(), 99.5)  # 99.5 percentile

        # Normalize each band in the RGB stack using the calculated min and max values
        for i, band_index in enumerate(rgb_band_indices):
            im_display[:, :, i] = imageutils.normalize(rgb_stack[:, :, i], im_min, im_max)

        # If cropping is requested, crop the composite image to remove black borders
        if crop_image:
            self.rgb_composite_crop = crop(im_display)
        else:
            self.rgb_composite = im_display  # Store the composite image without cropping

    def _make_rgb_enhancement(self, crop):
        """
            Enhance the RGB composite image using Gaussian blur and unsharp masking.

            Parameters:
                crop (bool): If True, enhance the cropped composite image;
                            if False, enhance the full composite image.
        """
        # Check if the cropped composite image needs to be created
        if crop and self.rgb_composite_crop is None:
            self._make_rgb_composite(crop)
        # Check if the full composite image needs to be created
        elif not crop and self.rgb_composite is None:
            self._make_rgb_composite(crop)

        # Select the appropriate RGB composite based on the crop parameter
        rgb_composite = self.rgb_composite_crop if crop else self.rgb_composite

        # Apply Gaussian blur to the composite image
        gaussian_rgb = cv2.GaussianBlur(rgb_composite, (9, 9), 10.0)

        # Clip values to ensure they are within the range [0, 1]
        gaussian_rgb[gaussian_rgb < 0] = 0
        gaussian_rgb[gaussian_rgb > 1] = 1

        # Perform unsharp masking to enhance edges
        unsharp_rgb = cv2.addWeighted(rgb_composite, 1.5, gaussian_rgb, -0.5, 0)

        # Clip values again to ensure they are within the range [0, 1]
        unsharp_rgb[unsharp_rgb < 0] = 0
        unsharp_rgb[unsharp_rgb > 1] = 1

        # Apply gamma correction to adjust the brightness and contrast
        gamma = 1.4
        gamma_corr_rgb = unsharp_rgb ** (1.0 / gamma)

        # Store the enhanced composite image in the appropriate attribute
        if crop:
            self.rgb_composite_enhanced_crop = gamma_corr_rgb  # Enhanced cropped composite
        else:
            self.rgb_composite_enhanced = gamma_corr_rgb  # Enhanced full composite

    def save_rgb_normalized(self, crop=False, enhanced=False, output_directory="../output/comp"):
        """
            Save the normalized RGB image as a PNG file.

            Parameters:
                crop (bool): If True, save the cropped version of the RGB image;
                            if False, save the full version.
                enhanced (bool): If True, save the enhanced version of the RGB image;
                                if False, save the standard version.
                output_directory (str): The directory where the image will be saved.
                                        Default is "../output/comp".
        """
        # Create the output directory if it does not exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Construct the file name based on the parameters and the first file name in the list
        file_name = os.path.join(
            output_directory,
            f"rgb_true_color{'_enhanced' if enhanced else ''}{'_crop' if crop else ''}_{utils.get_number_from_image_name(self.file_names[0])}.png"
        )

        # Save the normalized RGB image as a PNG file
        imageio.imwrite(file_name, (255 * self.get_rgb_normalized(crop, enhanced)).astype('uint8'))

    def get_rgb_normalized(self, crop=False, enhanced=False):
        """
            Retrieve the normalized RGB composite image.

            Parameters:
                crop (bool): If True, retrieve the cropped version of the RGB image;
                             if False, retrieve the full version.
                enhanced (bool): If True, retrieve the enhanced version of the RGB image;
                                 if False, retrieve the standard version.

            Returns:
                np.ndarray: The normalized RGB composite image.
        """
        if crop:
            return self._get_rgb_composite_crop() if not enhanced else self._get_rgb_composite_enhanced_crop()
        else:
            return self._get_rgb_composite() if not enhanced else self._get_rgb_composite_enhanced()

    def _get_rgb_composite(self):
        """
            Retrieve the standard RGB composite image.

            If the RGB composite image has not been created yet, it calls the
            method to create it.

            Returns:
                np.ndarray: The standard RGB composite image.
        """
        if self.rgb_composite is None:
            self._make_rgb_composite(False)
        return self.rgb_composite

    def _get_rgb_composite_crop(self):
        """
            Retrieve the cropped RGB composite image.

            If the cropped RGB composite image has not been created yet, it calls
            the method to create it.

            Returns:
                np.ndarray: The cropped RGB composite image.
        """
        if self.rgb_composite_crop is None:
            self._make_rgb_composite(True)
        return self.rgb_composite_crop

    def _get_rgb_composite_enhanced(self):
        """
            Retrieve the enhanced RGB composite image.

            If the enhanced RGB composite image has not been created yet, it calls
            the method to create it.

            Returns:
                np.ndarray: The enhanced RGB composite image.
        """
        if self.rgb_composite_enhanced is None:
            self._make_rgb_enhancement(False)
        return self.rgb_composite_enhanced

    def _get_rgb_composite_enhanced_crop(self):
        """
            Retrieve the cropped enhanced RGB composite image.

            If the cropped enhanced RGB composite image has not been created yet,
            it calls the method to create it.

            Returns:
                np.ndarray: The cropped enhanced RGB composite image.
        """
        if self.rgb_composite_enhanced_crop is None:
            self._make_rgb_enhancement(True)
        return self.rgb_composite_enhanced_crop

    def get_ndvi(self, cropping=False):
        """
            Calculate the Normalized Difference Vegetation Index (NDVI) from the image stack.

            NDVI is calculated using the Near-Infrared (NIR) and Red bands, which helps
            in analyzing vegetation health.

            Args:
                cropping (bool): If True, the NDVI will be calculated on the cropped image.
                                 If False, it will be calculated on the full image.

            Returns:
                np.ndarray: The calculated NDVI values as a 2D array.
        """
        stack = self.get_stack(cropping)
        nir_band = self.get_band_index("nir")
        red_band = self.get_band_index("red")
        ndvi = (stack[:, :, nir_band] - stack[:, :, red_band]) / (
                stack[:, :, nir_band] + stack[:, :, red_band])
        return ndvi

    def get_ndre(self, cropping=False):
        """
            Calculate the Normalized Difference Red Edge Index (NDRE) from the image stack.

            NDRE is particularly useful for assessing plant health and chlorophyll content,
            especially in densely vegetated areas.

            Args:
                cropping (bool): If True, the NDRE will be calculated on the cropped image.
                                If False, it will be calculated on the full image.

            Returns:
                np.ndarray: The calculated NDRE values as a 2D array.
        """
        stack = self.get_stack(cropping)
        nir_band = self.get_band_index("nir")
        rededge_band = self.get_band_index("red edge")

        ndre = (stack[:, :, nir_band] - stack[:, :, rededge_band]) / (
                stack[:, :, nir_band] + stack[:, :, rededge_band])
        return ndre

    def get_cir_normalized(self, cropping=False):
        """
            Generate a normalized Color Infrared (CIR) image.

            The CIR image is created using the NIR, Red, and Green bands.
            Normalization is applied to ensure the values are within the range [0, 1].

            Args:
                cropping (bool): If True, the CIR image will be generated from the cropped image.
                                 If False, it will be generated from the full image.

            Returns:
                np.ndarray: The normalized CIR image as a 3D array.
        """
        stack = self.get_stack(cropping)
        nir_band = self.get_band_index("nir")
        red_band = self.get_band_index("red")
        green_band = self.get_band_index("green")
        cir = np.zeros((stack.shape[0], stack.shape[1], 3), dtype=np.float32)
        cir[:, :, 0] = imageutils.normalize(stack[:, :, nir_band])
        cir[:, :, 1] = imageutils.normalize(stack[:, :, red_band])
        cir[:, :, 2] = imageutils.normalize(stack[:, :, green_band])
        return cir
