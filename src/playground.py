import cv2
import numpy as np
from matplotlib import pyplot as plt

from registration.registration_with_depth_matching import register
from registration.registration_with_depth_matching import set_intrinsic
#from registration.registration_with_feature_matching import register
import src.utils as utils
from pathlib import Path
from micasense import capture

depth_map_number="000051"
micasense_image_number = utils.get_micasense_number_from_basler_number(depth_map_number)

micasense_path="/media/david/T71/multispektral/20240416_esteburg/0006SET/000"
micasense_path=Path(micasense_path)


depth_path="/media/david/T71/SAMSON1_depth_old/SAMSON1_depth"
depth_path=Path(depth_path)


image_names = sorted(list(micasense_path.glob(f'IMG_{micasense_image_number}_*.tif')))
image_names = [x.as_posix() for x in image_names]

micasense_calib = utils.read_micasense_calib("../calib/micasense_calib.yaml")
cal_samson_1 = utils.read_basler_calib("../calib/SAMSON1_SAMSON2_stereo.yaml")
K_L, D_L, P_L, _ = cal_samson_1

file_to_depth_map = utils.find_depth_map_file(depth_path, int(depth_map_number))
depth_map = np.load(file_to_depth_map)
depth_map_resized = cv2.resize(depth_map, (5328, 4608), interpolation=cv2.INTER_LINEAR)

thecapture = capture.Capture.from_filelist(image_names)

set_intrinsic(thecapture, micasense_calib)

file_names = utils.extract_all_image_names(image_names)

registered_bands = register(thecapture, depth_map_resized, micasense_calib, P_L, file_names)
#registered_bands = register(thecapture, image_names,"stack",True,file_names)

composite = registered_bands.get_rgb_normalized()
#registered_bands.save_rgb_composite()
#registered_bands.save_rgb_composite_enhanced()

# Draw the bounding box on the image
bgr_image = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
# Display the image with the bounding box
cv2.namedWindow('Image composite basic', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite basic', bgr_image)
cv2.resizeWindow('Image composite basic', 800, 600)  # Set the window size (adjust as needed)





# Draw the bounding box on the image
composite_crop = registered_bands.get_rgb_normalized(crop=True)
bgr_image = cv2.cvtColor(composite_crop, cv2.COLOR_RGB2BGR)
# Display the image with the bounding box
cv2.namedWindow('Image composite crop', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite crop', bgr_image)
cv2.resizeWindow('Image composite crop', 800, 600)  # Set the window size (adjust as needed)






# Draw the bounding box on the image
composite_enhanced = registered_bands.get_rgb_normalized(enhanced=True)
bgr_image = cv2.cvtColor(composite_enhanced, cv2.COLOR_RGB2BGR)
# Display the image with the bounding box
cv2.namedWindow('Image composite enhanced', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite enhanced', bgr_image)
cv2.resizeWindow('Image composite enhanced', 800, 600)  # Set the window size (adjust as needed)




# Draw the bounding box on the image
composite_enhanced_crop = registered_bands.get_rgb_normalized(enhanced=True, crop=True)
bgr_image = cv2.cvtColor(composite_enhanced_crop, cv2.COLOR_RGB2BGR)
# Display the image with the bounding box
cv2.namedWindow('Image composite enhanced_crop', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite enhanced_crop', bgr_image)
cv2.resizeWindow('Image composite enhanced_crop', 800, 600)  # Set the window size (adjust as needed)





# Draw the bounding box on the image
image_stack = registered_bands.get_stack()
#bgr_image = cv2.cvtColor(composite_enhanced_crop, cv2.COLOR_RGB2BGR)
# Display the image with the bounding box
cv2.namedWindow('Image composite image_stack', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite image_stack', image_stack[...,0:3])
cv2.resizeWindow('Image composite image_stack', 800, 600)  # Set the window size (adjust as needed)



# Draw the bounding box on the image
image_stack_crop = registered_bands.get_stack(True)
#bgr_image = cv2.cvtColor(composite_enhanced_crop, cv2.COLOR_RGB2BGR)
# Display the image with the bounding box
cv2.namedWindow('Image composite image_stack_crop', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite image_stack_crop', image_stack_crop[...,0:3])
cv2.resizeWindow('Image composite image_stack_crop', 800, 600)  # Set the window size (adjust as needed)

nir_band=3
red_band=2
ndvi=registered_bands.get_ndvi(True)
ndre=registered_bands.get_ndre(True)
cir=registered_bands.get_cir_normalized(True)
figsize=(16,13)
fig, (ax3,ax4) = plt.subplots(1, 2, figsize=figsize)
ax3.set_title("NDVI")
ax3.imshow(ndvi)
ax4.set_title("Color Infrared (CIR) Composite")
ax4.imshow(cir)
fig, (ax5,ax6) = plt.subplots(1, 2, figsize=figsize)
ax5.imshow(ndre)
ax5.set_title("ndre")
plt.show()


cv2.waitKey(0)






