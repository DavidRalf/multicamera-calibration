import cv2
import numpy as np

from registration.registration_with_depth_matching import register, set_intrinsic
from registration.registration_with_feature_matching import register as register_feature
import utils as utils
from pathlib import Path
from micasense import capture

depth_map_number="000043"
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
#registered_bands = register_feature(thecapture, image_names,"stack",True,file_names)

composite = registered_bands.get_rgb_composite()
#registered_bands.save_rgb_composite()
#registered_bands.save_rgb_composite_enhanced()

# Draw the bounding box on the image
bgr_image = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
# Display the image with the bounding box
cv2.namedWindow('Image composite basic', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite basic', bgr_image)
cv2.resizeWindow('Image composite basic', 800, 600)  # Set the window size (adjust as needed)





# Draw the bounding box on the image
composite_crop = registered_bands.get_rgb_composite(crop=True)
bgr_image = cv2.cvtColor(composite_crop, cv2.COLOR_RGB2BGR)
# Display the image with the bounding box
cv2.namedWindow('Image composite crop', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite crop', bgr_image)
cv2.resizeWindow('Image composite crop', 800, 600)  # Set the window size (adjust as needed)






# Draw the bounding box on the image
composite_enhanced = registered_bands.get_rgb_composite(enhanced=True)
bgr_image = cv2.cvtColor(composite_enhanced, cv2.COLOR_RGB2BGR)
# Display the image with the bounding box
cv2.namedWindow('Image composite enhanced', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite enhanced', bgr_image)
cv2.resizeWindow('Image composite enhanced', 800, 600)  # Set the window size (adjust as needed)




# Draw the bounding box on the image
composite_enhanced_crop = registered_bands.get_rgb_composite(enhanced=True,crop=True)
bgr_image = cv2.cvtColor(composite_enhanced_crop, cv2.COLOR_RGB2BGR)
# Display the image with the bounding box
cv2.namedWindow('Image composite enhanced_crop', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite enhanced_crop', bgr_image)
cv2.resizeWindow('Image composite enhanced_crop', 800, 600)  # Set the window size (adjust as needed)




cv2.waitKey(0)
