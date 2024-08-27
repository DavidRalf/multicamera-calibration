import itertools
import json
import os

import cv2
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mutual_info_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
import src.utils as utils
from micasense import capture
from registration.registration_with_depth_matching import set_intrinsic


def load_the_capture(images,micasense_calib,micasense_path):
    image_names = sorted(images)
    image_names = [micasense_path.joinpath(image_name).as_posix() for image_name in image_names]

    # Capture Micasense images and set intrinsic parameters
    thecapture = capture.Capture.from_filelist(image_names)
    set_intrinsic(thecapture, micasense_calib)
    return thecapture,image_names


def create_output_directory(output_dir):
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_calibration_data():
    """Load calibration data for Micasense and SAMSON cameras."""
    micasense_calib = utils.read_micasense_calib("../calib/micasense_calib.yaml")
    cal_samson_1 = utils.read_basler_calib("../calib/SAMSON1_SAMSON2_stereo.yaml")
    K_L, D_L, P_L, _ = cal_samson_1
    return micasense_calib, P_L


def calculate_ncc_patches(R, F):
    """Calculate NCC for each pair of corresponding patches using vectorization."""
    # Ensure inputs are numpy arrays
    R = np.array(R)
    F = np.array(F)

    # Calculate means
    R_mean = np.mean(R, axis=(1, 2), keepdims=True)
    F_mean = np.mean(F, axis=(1, 2), keepdims=True)

    # Center the patches
    R_diff = R - R_mean
    F_diff = F - F_mean

    # Calculate numerator and denominator for NCC
    numerator = np.sum(R_diff * F_diff, axis=(1, 2))
    denominator = np.sqrt(np.sum(R_diff ** 2, axis=(1, 2)) * np.sum(F_diff ** 2, axis=(1, 2)))

    # Calculate NCC values
    ncc_values = numerator / denominator
    ncc_values[denominator == 0] = 0  # Handle division by zero

    return ncc_values


def calculate_ssim_patches(R, F):
    """Calculate SSIM for each pair of corresponding patches using vectorization."""
    # Ensure inputs are numpy arrays
    R = np.array(R)
    F = np.array(F)

    # Calculate means
    mu_R = np.mean(R, axis=(1, 2))
    mu_F = np.mean(F, axis=(1, 2))

    # Calculate variances
    sigma_R_sq = np.var(R, axis=(1, 2))
    sigma_F_sq = np.var(F, axis=(1, 2))

    # Calculate covariance
    covariance = np.mean((R - mu_R[:, np.newaxis, np.newaxis]) * (F - mu_F[:, np.newaxis, np.newaxis]), axis=(1, 2))

    # SSIM constants
    C1 = 1e-6
    C2 = 1e-6

    # Calculate SSIM
    numerator = (2 * mu_R * mu_F + C1) * (2 * covariance + C2)
    denominator = (mu_R ** 2 + mu_F ** 2 + C1) * (sigma_R_sq + sigma_F_sq + C2)

    # Calculate SSIM values
    ssim_values = numerator / denominator
    ssim_values[denominator == 0] = 0  # Handle division by zero

    return ssim_values



def calculate_mi_patches(patches_R, patches_F, bins=64, n_jobs=-1):
    """
    Calculate the Mutual Information (MI) for all pairs of corresponding patches
    using mutual_info_score from sklearn with parallel processing.

    Parameters:
    - patches_R: list or numpy array of shape (num_patches, patch_height, patch_width)
    - patches_F: list or numpy array of shape (num_patches, patch_height, patch_width)
    - bins: number of bins to use for discretization
    - n_jobs: number of parallel jobs to run (-1 uses all available cores)

    Returns:
    - mi_values: numpy array of MI values for each pair of corresponding patches
    """

    # Convert lists to numpy arrays if they are not already
    patches_R = np.array(patches_R)
    patches_F = np.array(patches_F)

    # Flatten patches to shape (num_patches, num_pixels)
    num_patches = patches_R.shape[0]
    patches_R_flat = patches_R.reshape(num_patches, -1)
    patches_F_flat = patches_F.reshape(num_patches, -1)

    # Discretize the data
    patches_R_discretized = np.floor(patches_R_flat * bins).astype(int)
    patches_F_discretized = np.floor(patches_F_flat * bins).astype(int)

    # Parallel processing of mutual_info_score
    mi_values = Parallel(n_jobs=n_jobs)(delayed(mutual_info_score)(
        patches_R_discretized[i], patches_F_discretized[i]
    ) for i in range(num_patches))

    return mi_values



def get_ncc_ssim_for_batch(stack, band_names, patch_size, ncc_stats, ssim_stats, mi_stats):
    """Calculate NCC, SSIM, and MI for each pair of bands in a stack."""
    for i, j in itertools.combinations(range(stack.shape[2]), 2):
        image1 = stack[:, :, i]
        image2 = stack[:, :, j]

        patches_R = get_patches(patch_size, image1)
        patches_F = get_patches(patch_size, image2)

        ncc_values = calculate_ncc_patches(patches_R, patches_F)
        ssim_values = calculate_ssim_patches(patches_R, patches_F)
        mi_values = calculate_mi_patches(patches_R, patches_F)

        ncc_stats[(band_names[i], band_names[j])] = ncc_values  # Store NCC values directly
        ssim_stats[(band_names[i], band_names[j])] = ssim_values  # Store SSIM values directly
        mi_stats[(band_names[i], band_names[j])] = mi_values  # Store MI values directly

    return ncc_stats, ssim_stats, mi_stats


def get_patches(patch_size, image):
    """Extract patches from an image based on the specified patch size."""
    h, w = image.shape
    patch_height, patch_width = patch_size

    num_patches_h = (h - patch_height) // patch_height + 1
    num_patches_w = (w - patch_width) // patch_width + 1

    return [image[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
            for i in range(num_patches_h) for j in range(num_patches_w)]


def calculate_mean_per_batch(ncc_stats, ssim_stats, mi_stats):
    """Calculate mean NCC, SSIM, and MI for each batch."""
    mean_ncc_per_pair = {pair: np.mean(ncc_values) for pair, ncc_values in ncc_stats.items()}
    mean_ssim_per_pair = {pair: np.mean(ssim_values) for pair, ssim_values in ssim_stats.items()}
    mean_mi_per_pair = {pair: np.mean(mi_values) for pair, mi_values in mi_stats.items()}

    overall_mean_ncc = np.mean(list(mean_ncc_per_pair.values()))
    overall_mean_ssim = np.mean(list(mean_ssim_per_pair.values()))
    overall_mean_mi = np.mean(list(mean_mi_per_pair.values()))

    return overall_mean_ncc, overall_mean_ssim, overall_mean_mi, mean_ncc_per_pair, mean_ssim_per_pair, mean_mi_per_pair


def calculate_statistics(metrics):
    """Calculate mean, median, and standard deviation for NCC, SSIM, and MI."""
    results = {metric: {"overall": {}, "pairs": {}} for metric in ['ncc', 'ssim', 'mi']}
    overall_values = {metric: [] for metric in results.keys()}
    all_pair_values = {metric: {} for metric in results.keys()}  # To store all pair values

    for batch_name, pairs in metrics.items():
        for pair, values in pairs.items():
            for metric in results.keys():
                pair_values = values.get(metric, [])

                if pair_values:
                    # Calculate pair statistics for current batch
                    pair_stats = calculate_pair_statistics(pair_values)
                    results[metric].setdefault(batch_name, {})[pair] = pair_stats

                    # Collect for batch and overall statistics
                    overall_values[metric].extend(pair_values)

                    # Collect pair values across all batches
                    if pair not in all_pair_values[metric]:
                        all_pair_values[metric][pair] = []
                    all_pair_values[metric][pair].extend(pair_values)

        # After processing all pairs, calculate batch statistics for each metric
        for metric in results.keys():
            batch_values = [v for p, v in pairs.items() if metric in v for v in v[metric]]
            if batch_values:
                results[metric][batch_name]['batch_stats'] = calculate_batch_statistics(batch_values)

    # Calculate overall statistics
    for metric in results.keys():
        results[metric]["overall"] = calculate_overall_statistics(overall_values[metric])

        # Calculate overall statistics for each pair across all batches
        for pair, pair_values in all_pair_values[metric].items():
            if pair_values:
                results[metric]["pairs"][pair] = calculate_overall_statistics(pair_values)

    return results


def calculate_pair_statistics(pair_values):
    """Calculate mean, median, and standard deviation for a pair of values."""
    return {
        "mean": np.mean(pair_values),
        "median": np.median(pair_values),
        "std": np.std(pair_values),
    }


def calculate_batch_statistics(batch_values):
    """Calculate batch-level statistics."""
    return {
        "batch_mean": np.mean(batch_values),
        "batch_median": np.median(batch_values),
        "batch_std": np.std(batch_values),
    }


def calculate_overall_statistics(overall_values):
    """Calculate overall statistics."""
    if not overall_values:  # Handle empty list case
        return {"mean": None, "median": None, "std": None}

    return {
        "mean": np.mean(overall_values),
        "median": np.median(overall_values),
        "std": np.std(overall_values),
    }


def save_stack_to_disk(stack, output_dir, filename="stack.npy"):
    """Save the stack to disk as a NumPy file."""
    file_path = os.path.join(output_dir, filename)
    np.save(file_path, stack)
    print(f"Stack saved to {file_path}")


def save_results_to_pdf(statistics, output_dir):
    # Create the PDF document
    pdf_file = os.path.join(output_dir, "evaluation_report.pdf")
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()

    # Define custom styles
    custom_heading_style = ParagraphStyle(name='CustomHeading', parent=styles['Heading2'], fontSize=14,
                                          textColor=colors.darkblue)
    custom_normal_style = ParagraphStyle(name='CustomNormal', parent=styles['Normal'], fontSize=12,
                                         textColor=colors.black, spaceAfter=2)

    story = []

    # Title Page
    story.append(Paragraph("SAMSON Evaluation Report", styles['Title']))
    story.append(Spacer(1, 24))  # Increased spacing for title
    story.append(Paragraph("Generated Report", custom_normal_style))
    story.append(Spacer(1, 48))  # More space before the next section

    # Overall Stats for each metric
    story.append(Paragraph("Overall Statistics", custom_heading_style))
    for metric, metric_stats in statistics.items():
        overall = metric_stats.get("overall", {})
        overall_data = [
            [Paragraph("Statistic", custom_normal_style), Paragraph("Value", custom_normal_style)],
            [Paragraph("Mean", custom_normal_style), Paragraph(str(overall.get("mean", "N/A")), custom_normal_style)],
            [Paragraph("Median", custom_normal_style),
             Paragraph(str(overall.get("median", "N/A")), custom_normal_style)],
            [Paragraph("Standard Deviation", custom_normal_style),
             Paragraph(str(overall.get("std", "N/A")), custom_normal_style)],
        ]
        overall_table = Table(overall_data, colWidths=[150, 150])
        overall_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(overall_table)
        story.append(Spacer(1, 24))  # Increased space after overall stats

    # Add a page for pair statistics across all batches
    story.append(PageBreak())  # Create a new page
    story.append(Paragraph("Pair Statistics Across All Batches", custom_heading_style))

    for metric, metric_stats in statistics.items():
        story.append(Paragraph(f"Metric: {metric.upper()}", custom_normal_style))
        pair_data = [[Paragraph("Pair", custom_normal_style), Paragraph("Mean", custom_normal_style),
                      Paragraph("Median", custom_normal_style), Paragraph("Standard Deviation", custom_normal_style)]]

        # Directly use the pairs statistics provided in the input
        pairs_stats = metric_stats.get("pairs", {})
        for pair, pair_stats in pairs_stats.items():
            pair=str(pair)
            pair_data.append([
                Paragraph(pair, custom_normal_style),
                Paragraph(str(pair_stats.get("mean", "N/A")), custom_normal_style),
                Paragraph(str(pair_stats.get("median", "N/A")), custom_normal_style),
                Paragraph(str(pair_stats.get("std", "N/A")), custom_normal_style)
            ])

        pair_table = Table(pair_data, colWidths=[200, 100, 100, 100])
        pair_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(pair_table)
        story.append(Spacer(1, 48))  # More space before the next metric

    # Iterate over metrics to create sections for each batch
    for batch_name in set(
            batch for metric in statistics.values() for batch in metric.keys() if batch not in ["overall", "pairs"]):
        for metric, metric_stats in statistics.items():
            batch_stats = metric_stats.get(batch_name)

            if not batch_stats:
                continue  # Skip if there are no stats for this batch in this metric

            story.append(PageBreak())  # Create a new page for each batch
            story.append(Paragraph(f"Batch: {batch_name}", custom_heading_style))

            # Load and display the stack image for the current batch
            stack_file = os.path.join(output_dir, f"{batch_name}.npy")
            if not os.path.exists(stack_file):
                print(f"Warning: {stack_file} not found. Skipping {batch_name}.")
                continue  # Skip if the stack file doesn't exist
            # Load the stack image (6 channels)
            stack = np.load(stack_file)

            # Display the RGB image and add it to the PDF
            if not display_rgb_image(stack, story, styles, output_dir, batch_name):
                print(f"Error displaying image for batch {batch_name}. Skipping.")

            # Add metric statistics
            story.append(Paragraph(f"Statistics for Metric: {metric.upper()}", custom_heading_style))
            batch_data = [
                [Paragraph("Statistic", custom_normal_style), Paragraph("Value", custom_normal_style)],
                [Paragraph("Batch Mean", custom_normal_style),
                 Paragraph(str(batch_stats["batch_stats"].get("batch_mean", "N/A")), custom_normal_style)],
                [Paragraph("Batch Median", custom_normal_style),
                 Paragraph(str(batch_stats["batch_stats"].get("batch_median", "N/A")), custom_normal_style)],
                [Paragraph("Batch Standard Deviation", custom_normal_style),
                 Paragraph(str(batch_stats["batch_stats"].get("batch_std", "N/A")), custom_normal_style)],
            ]
            batch_table = Table(batch_data, colWidths=[150, 150])
            batch_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(batch_table)
            story.append(Spacer(1, 24))  # Increased space after batch stats

            # Add stats for each pair in the batch
            story.append(Paragraph("Pair Statistics", custom_heading_style))
            pair_data = [[Paragraph("Pair", custom_normal_style), Paragraph("Mean", custom_normal_style),
                          Paragraph("Median", custom_normal_style),
                          Paragraph("Standard Deviation", custom_normal_style)]]
            for pair, pair_stats in batch_stats.items():
                if pair == "batch_stats":
                    continue
                pair_str = str(pair) if not isinstance(pair, str) else pair
                pair_data.append([Paragraph(pair_str, custom_normal_style),
                                  Paragraph(str(pair_stats.get("mean", "N/A")), custom_normal_style),
                                  Paragraph(str(pair_stats.get("median", "N/A")), custom_normal_style),
                                  Paragraph(str(pair_stats.get("std", "N/A")), custom_normal_style)])
            pair_table = Table(pair_data, colWidths=[200, 100, 100, 100])
            pair_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(pair_table)
            story.append(Spacer(1, 48))  # More space before the next batch

    # Build the PDF
    doc.build(story)


def display_rgb_image(stack, story, styles, output_dir, batch_name):
    """Generate an RGB image from a 6-channel stack, save it, and add it to the PDF story."""
    # Ensure stack has 6 channels
    if stack.shape[2] != 6:
        print(f"Error: Expected stack with 6 channels, but got {stack.shape[2]}.")
        return False

    # Extract Blue, Green, Red channels from the stack
    blue_channel = stack[:, :, 0]
    green_channel = stack[:, :, 1]
    red_channel = stack[:, :, 2]

    # Normalize channels to the range [0, 1]
    def normalize(channel):
        min_val, max_val = np.min(channel), np.max(channel)
        return (channel - min_val) / (max_val - min_val) if max_val > min_val else channel

    blue_normalized = normalize(blue_channel)
    green_normalized = normalize(green_channel)
    red_normalized = normalize(red_channel)

    # Create an RGB image
    rgb_image = np.zeros((*blue_channel.shape, 3))
    rgb_image[..., 0] = red_normalized  # Red channel
    rgb_image[..., 1] = green_normalized  # Green channel
    rgb_image[..., 2] = blue_normalized  # Blue channel

    # Save the RGB image
    rgb_image_path = os.path.join(output_dir, f"{batch_name}_rgb_image.png")
    plt.imsave(rgb_image_path, rgb_image)

    # Add the image to the PDF
    image = Image(rgb_image_path, width=400, height=300)  # Set size as needed
    story.append(image)
    story.append(Spacer(1, 12))  # Space after image

    return True


def convert_tuple_keys_to_strings(d):
    """Recursively convert tuple keys in a dictionary to strings."""
    if isinstance(d, dict):
        return {str(k): convert_tuple_keys_to_strings(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_tuple_keys_to_strings(item) for item in d]
    else:
        return d


def save_results_to_json(metrics, statistics, output_dir):
    """Save metrics and statistics to a JSON file."""

    # Convert any tuple keys in metrics and statistics to strings
    #metrics = convert_tuple_keys_to_strings(metrics)
    statistics = convert_tuple_keys_to_strings(statistics)

    # Save metrics to JSON
    #with open(os.path.join(output_dir, "metrics.json"), 'w') as json_file:
    #    json.dump(metrics, json_file, indent=4)

    # Structure the statistics for saving to JSON
    structured_statistics = {
        metric: {
            "overall": statistics[metric]["overall"],
            "pairs": {pair: {
                "mean": statistics[metric]["pairs"][pair]["mean"],
                "median": statistics[metric]["pairs"][pair]["median"],
                "std": statistics[metric]["pairs"][pair]["std"]
            } for pair in statistics[metric]["pairs"]},
            **{batch_name: {
                **statistics[metric][batch_name],
                "batch_stats": statistics[metric][batch_name].get("batch_stats", {})
            } for batch_name in statistics[metric] if batch_name != "overall" and batch_name != "pairs"}
        }
        for metric in statistics
    }

    # Save statistics to JSON
    with open(os.path.join(output_dir, "statistics.json"), 'w') as json_file:
        json.dump(structured_statistics, json_file, indent=4)


def store_metrics(metrics,batch_name,stack,band_names,patch_size):
    ncc_stats, ssim_stats, mi_stats = get_ncc_ssim_for_batch(stack, band_names, patch_size, {}, {}, {})

    # Store metrics in structured format
    for pair in ncc_stats.keys():  # Assuming ncc_stats has the same keys as channel pairs
        if batch_name not in metrics:
            metrics[batch_name] = {}
        if pair not in metrics[batch_name]:
            metrics[batch_name][pair] = {"ncc": [], "ssim": [], "mi": []}

        metrics[batch_name][pair]["ncc"].extend(ncc_stats[pair])  # Assuming these are lists of values
        metrics[batch_name][pair]["ssim"].extend(ssim_stats[pair])  # Assuming these are lists of values
        metrics[batch_name][pair]["mi"].extend(mi_stats[pair])  # Assuming these are lists of values
