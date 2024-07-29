from osgeo import gdal
import numpy as np
from PIL import Image

# Open the GeoTIFF file
tif_file = "../output/IMG_0042-pansharpened.tif"
dataset = gdal.Open(tif_file)

# Check if the dataset was opened successfully
if dataset is None:
    print("Failed to open the TIFF file.")
    exit()

# Get the number of bands
num_bands = dataset.RasterCount
print(f"Number of bands: {num_bands}")

# Create a mapping of band names to their indices
band_names = {}
for i in range(1, num_bands + 1):
    band = dataset.GetRasterBand(i)
    description = band.GetDescription() or f'Band {i}'
    band_names[description] = i  # Store the index with the description
    print(f"Band {i}: {description}")

# Determine the indices for Blue, Green, and Red bands
blue_index = band_names.get('Blue', None)
green_index = band_names.get('Green', None)
red_index = band_names.get('Red', None)

# Check if the required bands are available
if blue_index is None or green_index is None or red_index is None:
    print("Error: Required bands (Blue, Green, Red) are not available.")
    exit()

# Read the RGB bands using the identified indices
band_blue = dataset.GetRasterBand(blue_index).ReadAsArray()  # Blue
band_green = dataset.GetRasterBand(green_index).ReadAsArray()  # Green
band_red = dataset.GetRasterBand(red_index).ReadAsArray()  # Red

# Normalize the bands to [0, 255] for visualization
band_blue = (band_blue / np.max(band_blue) * 255).astype(np.uint8) if np.max(band_blue) > 0 else band_blue
band_green = (band_green / np.max(band_green) * 255).astype(np.uint8) if np.max(band_green) > 0 else band_green
band_red = (band_red / np.max(band_red) * 255).astype(np.uint8) if np.max(band_red) > 0 else band_red

# Create a true color composite
true_color = np.stack((band_red, band_green, band_blue), axis=-1)  # RGB order

# Save the true color composite using PIL
output_png = 'true_color_composite.png'
Image.fromarray(true_color).save(output_png, format='PNG', quality=100)

print(f"True color composite saved as {output_png}")

# Close the dataset
dataset = None
