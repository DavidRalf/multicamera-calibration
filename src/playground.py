import itertools

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.util import view_as_windows

from evaluation import eval_utils
from registration.registration_with_depth_matching import register
from registration.registration_with_depth_matching import set_intrinsic
#from registration.registration_with_feature_matching import register
import src.utils as utils
from pathlib import Path
from micasense import capture
from micasense.registered_micasense import crop
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
# Funktion zum Einlesen der YOLOv8 Segmentierungsdaten
def load_yolov8_segments(file_path):
    segments = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            class_index = int(values[0])
            # Die Koordinaten sind normalisiert
            coordinates = np.array(values[1:], dtype=float).reshape(-1, 2)
            segments.append((class_index, coordinates))
    return segments


def show_segment(image, name):
    # Convert image to 8-bit if it's not already
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert single-channel image to 3-channel for colored drawing
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Load segment data
    segment_data = load_yolov8_segments(f'../data/eval/segments/{name}_png.txt')

    # Draw each segment on the image
    for class_index, coordinates in segment_data:
        # Convert normalized coordinates to absolute pixel values
        coordinates[:, 0] *= image.shape[1]  # width
        coordinates[:, 1] *= image.shape[0]  # height
        coordinates = coordinates.astype(np.int32)

        # Draw the polygon on the color image
        cv2.polylines(image_color, [coordinates], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the image with segments
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis
    plt.show()

def transform_segments(image, name, depth_map_resized, micasense_calib, P_L,index):
    P_L = P_L[:, :3]  # Nimm die ersten drei Spalten
    segment_data = load_yolov8_segments(f'../data/eval/segments/{name}_png.txt')
    transformed_segments = []
    h1, w1 = depth_map_resized.shape
    band_data = utils.get_band_data(micasense_calib, index)
    rotation = band_data['rotation']
    R = np.array(rotation)
    translation = band_data['translation']
    t = np.array(translation).reshape(3, 1)
    h2, w2 = image.shape

    micasense_extrinsic = np.zeros((4, 4))
    micasense_extrinsic[:3, :3] = R
    micasense_extrinsic[:3, 3] = t.flatten()
    micasense_extrinsic[3, 3] = 1

    micasense_intrinsic = np.array(band_data['cameraMatrix'])
    u1, v1 = np.meshgrid(np.arange(w1), np.arange(h1))
    homog_coords = np.stack([u1.flatten(), v1.flatten(), np.ones_like(u1.flatten())], axis=0)
    # Berechnung der 3D-Koordinaten für alle Punkte in Bild 1
    K1_inv = np.linalg.inv(P_L)
    X1 = depth_map_resized.flatten() * np.dot(K1_inv, homog_coords)
    # Transformation in Kamera 2 Koordinaten
    X2 = np.dot(R, X1) + t

    # Projektion in Bild 2 Koordinaten
    x2 = np.dot(micasense_intrinsic, X2)
    # Normiere die homogenen Koordinaten (Floating Point Koordinaten)
    u2_float = x2[0] / x2[2]
    v2_float = x2[1] / x2[2]
    # Apply the within_image logic instead of clipping
    u2_float =  np.round(u2_float).astype(np.int32)
    v2_float =   np.round(v2_float).astype(np.int32)
    within_image_x = np.logical_and(np.greater_equal(u2_float, 0), np.less(u2_float, w2))
    within_image_y = np.logical_and(np.greater_equal(v2_float, 0), np.less(v2_float, h2))
    within_image = np.logical_and(within_image_x, within_image_y)

    valid_u2 = u2_float[within_image].astype(np.int32)
    valid_v2 = v2_float[within_image].astype(np.int32)

    for class_index, coordinates in segment_data:
        # Convert normalized coordinates to absolute pixel values
        coordinates[:, 0] *= image.shape[1]  # width
        coordinates[:, 1] *= image.shape[0]  # height
        coordinates = np.round(coordinates).astype(np.int32)
        mask = np.zeros((h2, w2), dtype=np.uint8)

        cv2.fillPoly(mask, [coordinates], 1)
        points = np.vstack((valid_u2, valid_v2)).T
        is_in_segment = mask[points[:, 1], points[:, 0]] == 1   # Combine u2 and v2 into a single array

        u1_segment = u1.flatten()[within_image][is_in_segment]
        v1_segment = v1.flatten()[within_image][is_in_segment]
        transformed_segments.append((class_index, u1_segment, v1_segment))



    return transformed_segments


def show_transformed_segments(image, transformed_segments, crop_offsets=False):
    """
    Displays the transformed segments on the transformed image.

    :param image: The transformed image on which the segments will be displayed.
    :param transformed_segments: List of transformed segments [(class_index, u1_segment, v1_segment), ...].
    :param crop_offsets: Tuple of (crop_offset_top, crop_offset_left, crop_offset_bottom, crop_offset_right).
    """
    # Unpack the crop offsets
    #crop_offset_top, crop_offset_left, crop_offset_bottom, crop_offset_right = crop_offsets

    # Convert the image to an 8-bit image if it is not already
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert a single channel image to a 3-channel image for colored drawing
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Colors for the segments (can be adjusted)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    # Loop over the transformed segments
    for class_index, u1_segment, v1_segment in transformed_segments:
        # Select a color based on the class index
        color = colors[class_index % len(colors)]

        # Create the coordinates for the polygon
        coordinates = np.stack((u1_segment, v1_segment), axis=-1).astype(np.int32)

        # Debug: Print original coordinates before adjustment
        #print("Original Coordinates:", coordinates)

        # Adjust coordinates based on cropping
        #coordinates[:, 0] -= crop_offset_left  # Adjust x coordinates
        #coordinates[:, 1] -= crop_offset_top   # Adjust y coordinates

        # Debug: Print adjusted coordinates
        #print("Adjusted Coordinates:", coordinates)

        # Ensure the coordinates are within the image bounds
        #coordinates = np.clip(coordinates,
        #                      (0, 0),
         #                     (image_color.shape[1] - 1 - crop_offset_right, image_color.shape[0] - 1 - crop_offset_bottom))

        # Draw the polygon on the colored image
        cv2.polylines(image_color, [coordinates], isClosed=True, color=color, thickness=2)

    # Show the image with the transformed segments
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes
    plt.show()


def calculate_iou(transformed_segments, original_transformed_segments, image_shape):
    ious = []

    for i in range(len(transformed_segments)):
        # Extract the class index and segment coordinates
        class_index, u1_segment, v1_segment = transformed_segments[i]
        orig_class_index, orig_u1_segment, orig_v1_segment = original_transformed_segments[i]

        # Ensure that the class indexes match
        assert class_index == orig_class_index, "Class indexes do not match!"

        # Create binary masks for both segments
        transformed_mask = np.zeros(image_shape, dtype=np.uint8)
        transformed_mask[v1_segment, u1_segment] = 1

        original_mask = np.zeros(image_shape, dtype=np.uint8)
        original_mask[orig_v1_segment, orig_u1_segment] = 1

        # Calculate intersection and union
        intersection = np.logical_and(transformed_mask, original_mask).sum()
        union = np.logical_or(transformed_mask, original_mask).sum()

        # Calculate IoU
        if union == 0:
            iou = 0  # Avoid division by zero
        else:
            iou = intersection / union

        ious.append((class_index, iou))

    return ious


depth_map_number="000043"
micasense_image_number = utils.get_micasense_number_from_basler_number(depth_map_number)

micasense_path="/media/david/T71/multispektral/20240416_esteburg/0006SET/000"
micasense_path=Path(micasense_path)


depth_path="/media/david/T71/SAMSON1_depth_new/SAMSON1_depth"
depth_path=Path(depth_path)


image_names = sorted(list(micasense_path.glob(f'IMG_{micasense_image_number}_*.tif')))
image_names = [x.as_posix() for x in image_names]
file_names = utils.extract_all_image_names(image_names)
micasense_calib = utils.read_micasense_calib("../calib/micasense_calib.yaml")
cal_samson_1 = utils.read_basler_calib("../calib/SAMSON1_SAMSON2_stereo.yaml")
K_L, D_L, P_L, _ = cal_samson_1

file_to_depth_map = utils.find_depth_map_file(depth_path, int(depth_map_number))
depth_map = np.load(file_to_depth_map)
depth_map_resized = cv2.resize(depth_map, (5328, 4608), interpolation=cv2.INTER_LINEAR)



thecapture = capture.Capture.from_filelist(image_names)
set_intrinsic(thecapture, micasense_calib)
irradiance_list = thecapture.dls_irradiance() + [0]

images = thecapture.undistorted_reflectance(irradiance_list)
# Segmentierungsdaten aus Bild 2 laden
#segment_data = np.loadtxt('../data/eval/segments/IMG_0042_1_png.txt')
#for index,image in enumerate(images):
#    show_segment(image,file_names[index])

images_segments=[]
for index, image in enumerate(images):
    images_segments.append(transform_segments(image, file_names[index],depth_map_resized,micasense_calib,P_L,index))

registered_bands = register(thecapture, depth_map_resized, micasense_calib, P_L, file_names)
rgb_stack= registered_bands.get_stack(False,[2,1,0])

#iou= calculate_iou(images_segments[0],images_segments[1],registered_bands.images[0].shape)
##show_transformed_segments(stack[:,:,0], images_segments[0])
#batch_pairs={}
#band_names = [registered_bands.get_band_name(i) for i in range(len(registered_bands.images))]
#for i, j in itertools.combinations(range(len(registered_bands.images)), 2):
#    image1 = registered_bands.images[i]
#    image2 = registered_bands.images[j]
#    iou = calculate_iou(images_segments[i], images_segments[j], registered_bands.images[0].shape)
#    batch_pairs[(band_names[i], band_names[j])] = iou
#mean_values={"overall":[]}
#for pair, values in batch_pairs.items():
#    if pair not in mean_values:
#        mean_values[pair] = {}
#        iou_values=[]
#        for segment in values:
#
#            iou_values.append(segment[1])
#    mean_values[pair] = np.mean(iou_values)
#    mean_values["overall"].extend(iou_values)
#
#np.mean(mean_values["overall"])
#
stack= registered_bands.get_stack(False)
cropped_stack, crop_offset =crop(stack,True)

basler_image_bgr = cv2.imread("/media/david/T71/multispektral/20240416_esteburg/SAMSON1/000043_rect.png")

gray_basler = cv2.cvtColor(basler_image_bgr, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Image composite basler_image_bgr', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite basler_image_bgr', basler_image_bgr)
cv2.resizeWindow('Image composite basler_image_bgr', 800, 600)  # Set the window size (adjust as needed)

cv2.namedWindow('Image composite gray_basler', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite gray_basler', gray_basler)
cv2.resizeWindow('Image composite gray_basler', 800, 600)  # Set the window size (adjust as needed)
top_left,bottom_right=crop_offset[0]
top_left_final,bottom_right_final=crop_offset[1]

# Step 1: Crop the image twice
cropped_image = gray_basler[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
cropped_image = cropped_image[top_left_final[0]:bottom_right_final[0], top_left_final[1]:bottom_right_final[1]]

# Step 2: Convert the cropped image to grayscale
#cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# Step 3: Calculate Mutual Information (MI)
mi_values = mutual_info_score(None, None, contingency=np.histogram2d(cropped_image.ravel(), cropped_stack[:,:,0].ravel(), bins=256)[0])

# Step 4: Calculate Structural Similarity Index (SSIM)
ssim_value = ssim(cropped_image, cropped_stack[:,:,0], data_range=cropped_image.max() - cropped_image.min())

# Step 5: Calculate Normalized Cross-Correlation (NCC)
# Assumes patches1 and patches2 are already extracted from the cropped_image_gray and cropped_stack respectively
mean1 = cropped_image.mean()
mean2 = cropped_stack[:,:,0].mean()

# Normalize the patches
norm_patch1 = cropped_image - mean1
norm_patch2 = cropped_stack[:,:,0] - mean2

# Calculate NCC
numerator = np.sum(norm_patch1 * norm_patch2)
denominator = np.sqrt(np.sum(norm_patch1 ** 2) * np.sum(norm_patch2 ** 2))

ncc_value = numerator / denominator if denominator != 0 else 0

# Handle NaNs if any (though the above should prevent NaNs)
ncc_value = 0 if np.isnan(ncc_value) else ncc_value

cv2.namedWindow('Image composite cropped_image', cv2.WINDOW_NORMAL)
cv2.imshow('Image composite cropped_image', cropped_image)
cv2.resizeWindow('Image composite cropped_image', 800, 600)  # Set the window size (adjust as needed)
cv2.waitKey(0)

#show_transformed_segments(stack[:,:,0], images_segments[0])
#show_transformed_segments(stack[:,:,1], images_segments[1])


#show_transformed_segments(cropped_stack[:,:,0], images_segments[0],crop_offset)

blue_image= stack[:,:,0]
green_image= stack[:,:,1]

patches_blue= eval_utils.create_patches(blue_image,(100,100))
patches_green= eval_utils.create_patches(green_image,(100,100))

ncc_value = eval_utils.calculate_ncc_patches(patches_blue, patches_green)
ssim_value = eval_utils.calculate_ssim_patches(patches_blue, patches_green)
mi_value = eval_utils.calculate_mi_patches(patches_blue, patches_green)


blue_image = np.expand_dims(blue_image, axis=0)
green_image = np.expand_dims(green_image, axis=0)
ncc_value2 = eval_utils.calculate_ncc_patches(blue_image, green_image)
ssim_value2 = eval_utils.calculate_ssim_patches(blue_image, green_image)
mi_value2 = eval_utils.calculate_mi_patches(blue_image, green_image)



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
rgb_stack = registered_bands.get_stack(True,[2,1,0])
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






