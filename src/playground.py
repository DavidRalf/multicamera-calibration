import cv2
import numpy as np

from registration.registration_with_depth_matching import register, set_intrinsic
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

composite = registered_bands.get_rgb_composite()
registered_bands.save_rgb_composite()
registered_bands.save_rgb_composite_enhanced()
