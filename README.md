# Multicamera Calibration

## Overview

This project was developed as part of the Hauptprojekt at HAW (University of Applied Sciences). The main goal was to calibrate a camera system so that sensor data could be combined and used efficiently. The camera system is used to monitor trees that are close to the camera, which results in images having objects at different depths. This depth difference leads to parallax, making feature matching unsuitable for accurate registration as previously mentioned.

A major focus of this project was on the registration of Micasense bands due to physical offsets that were not corrected mechanically. As the manufacturer states:

*"The images (and lenses) of RedEdge are not registered or aligned mechanically. This is because the level of precision to be able to do this mechanically and get good alignment results is quite high (and unrealistic). In addition, shock, vibration, or handling of the camera can easily shift the alignment slightly, enough to be very noticeable in the output image."*

Because of this, the registration must happen during post-processing. Micasense offers a feature-based method, but notes that this approach fails when parallax causes a 3D depth field, which occurs when objects are very close or at significantly different distances, such as a nearby tree against a distant background. In such cases, photogrammetry techniques are needed to find a 3D transformation between images.

This project implements a method that registers the bands using depth matching (3D transformation). Both approaches – feature matching and depth matching – were implemented and evaluated.

## Project Objectives

Key objectives of this project include:

- Registering Micasense images (bands) with physical offsets.
- Implementing two registration methods: feature matching and depth matching.
- Calculating intrinsic and extrinsic parameters between cameras (calibration).
- Developing scripts to evaluate and compare the results of both registration methods.

## Project Structure

The project is organized into the following directories:

- **`calib/`**: Contains camera parameters from the calibration process.
- **`data/`**: Directory for some data used in this project.
- **`evaluation/`**: Scripts and tools for evaluating registration accuracy and results.
- **`micasense/`**: Code and utilities for processing Micasense images, including the developed `RegisteredMicasense` class to handle registered images.
- **`output/`**: Directory for saving processed images and results from the scripts.
- **`registration/`**: Functions and methods for image registration using feature matching and depth matching.
- **`src/`**: Source code for scripts such as calibration (intrinsic and extrinsic), depth creation using stereo, and more.
- **`showcase.ipynb`**: Jupyter Notebook showcasing how to use the code for image processing and registration, and how to work with the `RegisteredMicasense` class.

## Installation

To get started, clone the repository and install the necessary dependencies. Ensure you have Python and `pip` or `conda` installed on your system.

```bash
git clone <repository_url>
cd multicamera-calibration
pip install -r requirements.txt
# or
conda env create --file=micasense.yaml
```

## Calibration

If calibration was performed outside this code, add the YAML files to the `calib/` folder. The structure and naming of the YAML files must match exactly.

Alternatively, the script `src/calibrate.py` can be used to calibrate the sensors. Depending on the parameters, only the intrinsic parameters for each band and the extrinsic parameters between SAMSON1 and each band will be calculated, or additionally, the intrinsics for SAMSON1 and SAMSON2 and the extrinsics between the two sensors can be calculated as well.

The calibration uses ChArUco boards, stereo calibration, and the `cv2.calibrateCamera` function.

### Usage

- **micasense_path**: Path to the Micasense calibration images.
- **basler1_path**: Path to the Basler1 (SAMSON1) calibration images.
- **basler2_path**: Path to the Basler2 (SAMSON2) calibration images.
- **image_number**: Image number for extrinsic calibration based on Basler numbers (e.g., 000002).
- **calculate_basler_new**: Recalculate Basler calibration (true/false, default: false).
- **calculate_micasense_new**: Recalculate Micasense calibration (true/false, default: true).

```bash
cd src
python3 calibrate.py micasense_path basler1_path basler2_path image_number calculate_basler_new calculate_micasense_new
```

The generated YAML files will be saved in the `calib/` folder.

## Registration

For depth matching, a custom script was developed, and the feature matching method provided by Micasense was converted into a script as well. Both scripts allow the registration of a single capture (6 bands) or multiple captures.

The registered images are saved for possible future analysis.

### Feature Matching

The bands are registered to a reference band.

- **image_path**: Path to the directory containing Micasense images.
- **save_warp_matrices**: True or False, depending on whether to save the latest warp matrices.
- **version**: 'Stack' or 'Upsampled'.
- **output**: Path to save the registered images (default: `output/feature_matching`).
- **image_number**: Image number with leading zeros (e.g., 0059) to register, or `None` to register all images in the directory.

```bash
cd registration
python3 registration_with_feature_matching.py image_path save_warp_matrices version output image_number
```

Each registered band is saved as a separate `.tif` file in the output path.

### Depth Matching (3D-Transformation)

All Micasense bands are registered to Basler1 (SAMSON1).

- **depth_path**: Path to the directory containing depth data.
- **micasense_path**: Path to the directory containing Micasense images.
- **output**: Path to save the registered images (default: `output/depth_matching`).
- **image_number**: Image number based on Basler numbers with leading zeros (e.g., 000001), default is `None` to register all images in the folder.
- **basler_size**: Original Basler image size (width, height), default: `(5328, 4608)`.

```bash
cd registration
python3 registration_with_depth_matching.py depth_path micasense_path --output output --image_number image_number --basler_size basler_size
```

Each registered band is saved as a separate `.tif` file in the output path.

## RegisteredMicasense Class

To understand how to use the implemented `RegisteredMicasense` class, as well as how to utilize the code for custom scripts—such as further analyzing the registered images—please refer to the `showcase.ipynb` Jupyter Notebook. It provides examples and guidance on working with the class and the registration process.

## create_depth.py

This script was developed to generate depth maps using stereo methods (specifically, `StereoSGBM_create`) to evaluate the registration process with two depth estimation techniques. One is the standard stereo depth calculation, and the other is an AI model in development by Juri Zach. This comparison helps assess how well the AI model performs against the conventional method and highlights the importance of AI in depth estimation.

- **left_image**: Path to the left rectified image.
- **right_image**: Path to the right rectified image.
- **yaml_camera1**: Path to the YAML file for Camera 1.
- **yaml_camera2**: Path to the YAML file for Camera 2.
- **output_depth_path**: Path to save the generated depth map.
```bash
cd src
python3 create_depth.py left_image right_image yaml_camera1 yaml_camera2 output_depth_path
```

## Evaluation

Three scripts were developed to evaluate the registration process. These include:

1. **`eval_depth.py`**: Evaluates the results of the depth matching.
2. **`eval_feature.py`**: Evaluates the results of the feature matching.
3. **`eval_compare.py`**: Compares either two registration methods or two depth estimation methods (e.g., AI vs. conventional methods).

The evaluation is performed using three metrics: NCC (Normalized Cross-Correlation), SSIM (Structural Similarity Index), and MI (Mutual Information). Additionally, an objective evaluation is done by displaying the registered images in RGB format. Each script generates a PDF report summarizing the results.

### eval_depth.py

- **micasense_path**: Path to the directory containing Micasense images that are listed in `data/eval/images_to_eval.json`.
- **depth_path**: Path to the directory containing depth maps corresponding to the images listed in `data/eval/images_to_eval.json`.
- **name**: Name for the result folder, used for organizing results.
- **output_dir**: Directory where results will be saved (default: `output`).

```bash
cd evaluation
python3 eval_depth.py micasense_path depth_path name --output_dir output_dir
```

### eval_feature.py

- **micasense_path**: Path to the directory containing Micasense images that are listed in `data/eval/images_to_eval.json`.
- **name**: Name for the result folder, used for organizing results.
- **output_dir**: Directory where results will be saved (default: `output`).

```bash
cd evaluation
python3 eval_feature.py micasense_path name --output_dir output_dir
```

### eval_compare.py

- **method1**: Path to the evaluation results from Method 1.
- **method2**: Path to the evaluation results from Method 2.
- **output_dir**: Output directory for the PDF report (default: `output/eval/compare`).

```bash
cd evaluation
python3 eval_compare.py method1 method2 output_dir
```

The PDF report will display the results from Method 1 on the left and Method 2 on the right for easy comparison.

## viz_registered_image.py

This script visualizes a registered image, displaying several formats:

- RGB image
- Cropped RGB image
- CIR (Color Infrared) image
- Cropped CIR image
- NDVI (Normalized Difference Vegetation Index)
- Cropped NDVI
- NDRE (Normalized Difference Red Edge)
- Cropped NDRE

### Parameters

- **image_number**: Image number to process.
- **path_to_registered_images**: Path to the directory containing registered TIFF images.

```bash
cd src
python3 viz_registered_image.py image_number path_to_registered_images
```
