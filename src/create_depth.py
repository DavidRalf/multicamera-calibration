import argparse
import os

import cv2
import numpy as np
import yaml


# Function to load YAML file with camera parameters
def load_camera_params(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data


# Function to compute depth map
def compute_depth_map(left_image_path, right_image_path, yaml_camera1, yaml_camera2, output_depth_path):
    # Load rectified stereo images
    img_left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure images are loaded
    if img_left is None or img_right is None:
        print("Error: Unable to load images.")
        return

    # Load camera parameters from YAML files
    camera1_params = load_camera_params(yaml_camera1)
    camera2_params = load_camera_params(yaml_camera2)

    # Extract focal length from Camera 1's projection matrix
    projection_matrix_camera1 = camera1_params['projectionMatrix']
    f_camera1 = projection_matrix_camera1[0][0]  # Focal length fx for Camera 1

    # Extract baseline from Camera 2's projection matrix
    projection_matrix_camera2 = camera2_params['projectionMatrix']
    baseline = -projection_matrix_camera2[0][3]  # Extract baseline

    # Create StereoSGBM object
    stereo = cv2.StereoSGBM_create(numDisparities=256, blockSize=7,
                                   P1=8 * 3 * 7 ** 2,
                                   P2=32 * 3 * 7 ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32)

    disparity = stereo.compute(img_left, img_right).astype(np.float32)

    # Normalize disparity map for visualization
    disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_vis = np.uint8(disparity_vis)

    # Save the disparity map as an image
    left_image_name = os.path.splitext(os.path.basename(left_image_path))[0]  # Get the left image name
    disparity_image_path = os.path.join(output_depth_path, f'{left_image_name}_disparity.png')
    cv2.imwrite(disparity_image_path, disparity_vis)
    print(f"Disparity map saved as '{disparity_image_path}'")

    # Avoid division by zero in disparity map
    disparity[disparity <= 0] = 0.1  # Set small positive disparity where it's 0 or negative

    # Compute depth map (Z = f * B / d)
    depth_map = (f_camera1 * baseline) / disparity

    # Use percentile clipping to remove extreme outliers
    upper_limit = np.percentile(depth_map, 60)

    # Clip depth map to remove extreme outliers
    depth_map_clipped = np.clip(depth_map, None, upper_limit)
    # Normalize disparity map for visualization
    depth_map_color = cv2.normalize(depth_map_clipped, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_color = np.uint8(depth_map_color)

    # Colorize the depth map using a different color map for better depth visualization
    depth_map_color = cv2.applyColorMap(depth_map_color, cv2.COLORMAP_PLASMA)
    depth_map_color = cv2.bitwise_not(depth_map_color)  # Invert colors if needed

    # Save the colorized depth map as an image
    depth_image_path = os.path.join(output_depth_path, f'{left_image_name}_depth.png')
    cv2.imwrite(depth_image_path, depth_map_color)
    print(f"Colorized depth map saved as '{depth_image_path}'")

    # Save the depth map as a NumPy array with the custom naming convention
    npy_file_name = f"{left_image_name}.npy"  # Custom naming convention
    npy_file_path = os.path.join(output_depth_path, npy_file_name)
    np.save(npy_file_path, depth_map)
    print(f"Depth map saved as '{npy_file_path}'")

    # Show the colorized depth map
    cv2.namedWindow('Colorized Depth Map', cv2.WINDOW_NORMAL)
    cv2.imshow('Colorized Depth Map', depth_map_color)
    cv2.resizeWindow('Colorized Depth Map', 800, 600)  # Set the window size
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Close the window after a key press


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute depth map from stereo images.')
    parser.add_argument('left_image', type=str, help='Path to the left rectified image.')
    parser.add_argument('right_image', type=str, help='Path to the right rectified image.')
    parser.add_argument('yaml_camera1', type=str, help='Path to the YAML file for Camera 1.')
    parser.add_argument('yaml_camera2', type=str, help='Path to the YAML file for Camera 2.')
    parser.add_argument('output_depth_path', type=str, help='Path to save the depth map.')

    args = parser.parse_args()

    # Call the function to compute depth map
    compute_depth_map(args.left_image, args.right_image, args.yaml_camera1, args.yaml_camera2, args.output_depth_path)
