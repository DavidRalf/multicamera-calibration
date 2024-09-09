import argparse
from pathlib import Path
from micasense.registered_micasense import RegisteredMicasense
from src import utils
import matplotlib.pyplot as plt


def display_images(rgb, rgb_crop, nir, nir_crop, ndvi, ndvi_crop, ndre, ndre_crop):
    """Function to display the images in a grid format."""
    fig, axs = plt.subplots(4, 2, figsize=(12, 16))

    # Display RGB images
    axs[0, 0].imshow(rgb)
    axs[0, 0].set_title('RGB Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(rgb_crop)
    axs[0, 1].set_title('Cropped RGB Image')
    axs[0, 1].axis('off')

    # Display CIR images
    axs[1, 0].imshow(nir)
    axs[1, 0].set_title('Color Infrared (CIR) Image')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(nir_crop)
    axs[1, 1].set_title('Cropped CIR Image')
    axs[1, 1].axis('off')

    # Display NDVI images
    axs[2, 0].imshow(ndvi, cmap='hot')
    axs[2, 0].set_title('NDVI Image')
    axs[2, 0].axis('off')

    axs[2, 1].imshow(ndvi_crop, cmap='hot')
    axs[2, 1].set_title('Cropped NDVI Image')
    axs[2, 1].axis('off')

    # Display NDRE images
    axs[3, 0].imshow(ndre, cmap='hot')
    axs[3, 0].set_title('NDRE Image')
    axs[3, 0].axis('off')

    axs[3, 1].imshow(ndre_crop, cmap='hot')
    axs[3, 1].set_title('Cropped NDRE Image')
    axs[3, 1].axis('off')

    plt.tight_layout()
    plt.show()


def main(image_number, path_to_registered_images):
    # Ensure the path to registered images is a Path object
    path_to_registered_images = Path(path_to_registered_images)

    # Load image names matching the given image number
    image_names = sorted(path_to_registered_images.glob(f'IMG_{image_number}_*.tif'))
    image_names = [x.as_posix() for x in image_names]

    # Extract file names
    file_names = utils.extract_all_image_names(image_names)

    # Create a RegisteredMicasense object
    registered_images = RegisteredMicasense(image_names, file_names)

    # Get normalized RGB images
    rgb = registered_images.get_rgb_normalized()
    rgb_crop = registered_images.get_rgb_normalized(crop=True)

    # Get Color Infrared (CIR) normalized images
    cir = registered_images.get_cir_normalized()
    cir_crop = registered_images.get_cir_normalized(cropping=True)

    # Calculate NDRE
    ndre = registered_images.get_ndre()
    ndre_crop = registered_images.get_ndre(cropping=True)

    # Calculate NDVI
    ndvi = registered_images.get_ndvi()
    ndvi_crop = registered_images.get_ndvi(cropping=True)

    # Display the images
    display_images(rgb, rgb_crop, cir, cir_crop, ndvi, ndvi_crop, ndre, ndre_crop)


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process and visualize Registered Micasense RedEdge-P images for RGB, NDVI, NDRE, and CIR.')
    parser.add_argument('image_number', type=str, help='Image number to process.')
    parser.add_argument('path_to_registered_images', type=str,
                        help='Path to the directory containing registered TIFF images.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.image_number, args.path_to_registered_images)