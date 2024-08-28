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
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
import src.utils as utils
from micasense import capture
from registration.registration_with_depth_matching import set_intrinsic
from skimage.metrics import structural_similarity as ssim


def load_the_capture(images, micasense_calib, micasense_path):
    image_names = sorted(images)
    image_names = [micasense_path.joinpath(image_name).as_posix() for image_name in image_names]

    # Capture Micasense images and set intrinsic parameters
    thecapture = capture.Capture.from_filelist(image_names)
    set_intrinsic(thecapture, micasense_calib)
    return thecapture, image_names


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


def calculate_ncc_patches(image1, image2):
    assert image1.shape == image2.shape, "Input images must have the same dimensions"

    patches1 = image1
    patches2 = image2
    # Normalize patches
    mean1 = patches1.mean(axis=(1, 2), keepdims=True)
    mean2 = patches2.mean(axis=(1, 2), keepdims=True)
    norm_patch1 = patches1 - mean1
    norm_patch2 = patches2 - mean2

    # Calculate NCC for each patch
    numerator = np.sum(norm_patch1 * norm_patch2, axis=(1, 2))  # Sum over height and width of the patches
    denominator = np.sqrt(np.sum(norm_patch1 ** 2, axis=(1, 2)) * np.sum(norm_patch2 ** 2, axis=(1, 2)))

    ncc_values = numerator / denominator
    ncc_values[np.isnan(ncc_values)] = 0  # Handle NaN values

    return ncc_values


def calculate_ssim_patches(image1, image2):
    assert image1.shape == image2.shape, "Input images must have the same dimensions"

    patches1 = image1
    patches2 = image2

    # Calculate SSIM for each patch
    ssim_values = np.array([
        ssim(p1, p2, data_range=p1.max() - p1.min()) for p1, p2 in zip(patches1, patches2)
    ])

    # Return the SSIM values in a reshaped array
    #num_patches = int(np.sqrt(len(ssim_values)))  # Assuming square root to get dimensions
    return ssim_values


def calculate_mi_patches(image1, image2, bins=256):
    assert image1.shape == image2.shape, "Input images must have the same dimensions"

    patches1 = image1
    patches2 = image2

    # Calculate MI for each patch
    mi_values = np.array([
        mutual_info_score(None, None, contingency=np.histogram2d(p1.ravel(), p2.ravel(), bins=bins)[0])
        for p1, p2 in zip(patches1, patches2)
    ])

    # Return the MI values in a reshaped array
    #num_patches = int(np.sqrt(len(mi_values)))  # Assuming square root to get dimensions
    return mi_values


def get_ncc_ssim_for_batch(batch_patches, band_names, ncc_stats, ssim_stats, mi_stats):
    """Calculate NCC, SSIM, and MI for each pair of bands in a stack."""
    for i, j in itertools.combinations(range(len(batch_patches)), 2):
        image1 = batch_patches[i]
        image2 = batch_patches[j]

        patches_R = image1
        patches_F = image2

        ncc_values = calculate_ncc_patches(patches_R, patches_F)
        ssim_values = calculate_ssim_patches(patches_R, patches_F)
        mi_values = calculate_mi_patches(patches_R, patches_F)

        ncc_stats[(band_names[i], band_names[j])] = ncc_values  # Store NCC values directly
        ssim_stats[(band_names[i], band_names[j])] = ssim_values  # Store SSIM values directly
        mi_stats[(band_names[i], band_names[j])] = mi_values  # Store MI values directly

    return ncc_stats, ssim_stats, mi_stats


def get_patches(stack, patch_size):
    batch_patches = []
    for i in range(stack.shape[2]):
        image1 = stack[:, :, i]
        batch_patches.append(create_patches(image1, patch_size))
    return batch_patches


def create_patches(image, patch_size=(64, 64)):
    h, w = image.shape
    ph, pw = patch_size

    # Truncate the image to ensure dimensions are divisible by patch size
    h_trunc = h - (h % ph)
    w_trunc = w - (w % pw)
    image_truncated = image[:h_trunc, :w_trunc]

    # Calculate number of patches
    num_patches_h = h_trunc // ph
    num_patches_w = w_trunc // pw

    # Create patches using slicing
    patches = image_truncated.reshape(num_patches_h, ph, num_patches_w, pw).swapaxes(1, 2).reshape(-1, ph, pw)

    return patches


def transform_metric(raw_data):
    results = {"overall": {},
               "pairs": {},
               "batches": {}
               }

    for batch_name, pairs in raw_data.items():
        if batch_name not in results["batches"]:
            results["batches"][batch_name] = pairs
        for pair, metrics in list(pairs.items()):
            if pair not in results["pairs"]:
                results["pairs"][pair] = {}
            for metric, values in metrics.items():
                if metric not in results["pairs"][pair]:
                    results["pairs"][pair][metric] = []
                if metric not in results["overall"]:
                    results["overall"][metric] = []
                if metric not in results["batches"][batch_name]:
                    results["batches"][batch_name][metric] = []

                results["batches"][batch_name][metric].extend(values)
                results["overall"][metric].extend(values)
                results["pairs"][pair][metric].extend(values)

    return results


def calculate_statistics(raw_data):
    statistics = {"overall": {},
                  "pairs": {},
                  "batches": {}
                  }

    #overall across all batches
    for metric, values in raw_data["overall"].items():
        statistics["overall"][metric] = calculate_stats(raw_data["overall"][metric])

    # pair across all batches
    for pair, metrics in raw_data["pairs"].items():
        statistics["pairs"][pair] = {}
        for metric, values in metrics.items():
            statistics["pairs"][pair][metric] = calculate_stats(values)

    # for each pair in a batch and the batch across all pairs
    for batch_name, pairs in raw_data["batches"].items():
        statistics["batches"][batch_name] = {}
        for pair, metrics in pairs.items():
            if pair in statistics["overall"]:
                statistics["batches"][batch_name][pair] = calculate_stats(metrics)
                continue
            statistics["batches"][batch_name][pair] = {}
            for metric, values in metrics.items():
                statistics["batches"][batch_name][pair][metric] = calculate_stats(values)

    return statistics


def calculate_stats(values):
    """Calculate mean, median, and standard deviation for a pair of values."""
    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
    }


def save_stack_to_disk(stack, output_dir, filename="stack.npy"):
    """Save the stack to disk as a NumPy file."""
    file_path = os.path.join(output_dir, filename)
    np.save(file_path, stack)
    print(f"Stack saved to {file_path}")


def abbreviate_pair(pair):
    """Abbreviate the pair name for better fitting."""
    return f"{pair[1]}-{pair[3]}"  # Example: ('Blue', 'Green') -> 'B-G'


def save_results_to_pdf(statistics, output_dir):
    # Create the PDF document
    pdf_file = os.path.join(output_dir, "evaluation_report.pdf")
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()

    # Define custom styles
    custom_heading_style = ParagraphStyle(name='CustomHeading', parent=styles['Heading2'], fontSize=14,
                                          textColor=colors.darkblue, alignment=1)  # Center alignment
    custom_normal_style = ParagraphStyle(name='CustomNormal', parent=styles['Normal'], fontSize=10,
                                         textColor=colors.black, spaceAfter=2, alignment=1)  # Center alignment
    custom_normal_style2 = ParagraphStyle(name='CustomNormal', parent=styles['Normal'], fontSize=5,
                                          textColor=colors.black, spaceAfter=2, alignment=1)  # Center alignment

    story = []

    # Title Page
    story.append(Paragraph("SAMSON", styles['Title']))
    story.append(Spacer(1, 12))  # Spacing between title and subtitle
    story.append(Paragraph("Micasense Registration Evaluation Report", styles['Title']))
    story.append(Spacer(1, 48))  # More space before generated report
    story.append(Paragraph("Generated Report", custom_normal_style))
    story.append(Spacer(1, 48))  # More space before the next section

    # Overall Stats Table for all metrics
    story.append(Paragraph("Overall Statistics", custom_heading_style))

    # Prepare data for the overall statistics table
    overall_data = [
        [Paragraph("Metric", custom_normal_style),
         Paragraph("Mean", custom_normal_style),
         Paragraph("Median", custom_normal_style),
         Paragraph("Std Dev", custom_normal_style)]  # Shortened label
    ]

    # Iterate through the statistics to fill in the overall data
    for metric, metric_stats in statistics["overall"].items():
        overall_data.append([
            Paragraph(metric.upper(), custom_normal_style),
            Paragraph(f"{metric_stats.get('mean', 'N/A'):.4f}", custom_normal_style),
            Paragraph(f"{metric_stats.get('median', 'N/A'):.4f}", custom_normal_style),
            Paragraph(f"{metric_stats.get('std', 'N/A'):.4f}", custom_normal_style)
        ])

    # Create the overall statistics table
    overall_table = Table(overall_data, colWidths=[80, 50, 50, 50])  # Adjusted column widths
    overall_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    # Add the overall statistics table to the story
    story.append(overall_table)
    story.append(Spacer(1, 12))  # Space after overall stats

    # Add a page for pair statistics matrix
    story.append(PageBreak())  # Create a new page
    story.append(Paragraph("Pair Statistics Matrix", custom_heading_style))

    # Prepare data for the pairs matrix
    metrics = ['ncc', 'ssim', 'mi']  # List of metrics to display in the matrix
    pairs_list = list(statistics["pairs"].keys())  # List of pairs

    # Prepare data for the pairs matrix
    pairs_matrix_data = [
        [Paragraph("Pairs", custom_normal_style)] + [Paragraph(metric.upper(), custom_normal_style) for metric in
                                                     metrics]]

    # Fill in the matrix with mean, median, std values for each pair and each metric
    for pair in pairs_list:
        row = [Paragraph(str(pair), custom_normal_style)]  # Pair name at the start of the row
        for metric in metrics:
            metric_stats = statistics["pairs"][pair].get(metric, {})
            mean = round(metric_stats.get("mean", 0), 4)
            median = round(metric_stats.get("median", 0), 4)
            std = round(metric_stats.get("std", 0), 4)
            row.append(Paragraph(f"M: {mean:.4f}\nMed: {median:.4f}\nStd: {std:.4f}",
                                 custom_normal_style))  # Multi-line display
        pairs_matrix_data.append(row)

    # Calculate optimal column widths for pairs matrix
    pairs_matrix_col_widths = [80] + [50] * len(metrics)  # Adjusted widths for better fit

    # Create the pairs statistics matrix
    pairs_matrix = Table(pairs_matrix_data, colWidths=pairs_matrix_col_widths)
    pairs_matrix.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    # Add the pairs statistics matrix to the story
    story.append(pairs_matrix)
    story.append(Spacer(1, 12))  # Space after pairs matrix

    # Add a new page for batch statistics
    story.append(PageBreak())  # Create a new page for batch statistics

    # Process and add batch statistics tables
    for batch_name, batch_data in statistics["batches"].items():
        story.append(Paragraph(f"Statistics for {batch_name}", custom_heading_style))
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
        # Prepare data for the batch statistics table
        batch_stats_data = [
            [Paragraph("Metric", custom_normal_style),
             Paragraph("Mean", custom_normal_style),
             Paragraph("Median", custom_normal_style),
             Paragraph("Std Dev", custom_normal_style)]  # Shortened label
        ]

        # Iterate through metrics for the batch
        for metric in ['ncc', 'ssim', 'mi']:
            stats = batch_data[metric]
            batch_stats_data.append([
                Paragraph(metric.upper(), custom_normal_style),
                Paragraph(f"{stats['mean']:.4f}", custom_normal_style),
                Paragraph(f"{stats['median']:.4f}", custom_normal_style),
                Paragraph(f"{stats['std']:.4f}", custom_normal_style)
            ])

        # Create the batch statistics table
        batch_stats_table = Table(batch_stats_data, colWidths=[80, 50, 50, 50])
        batch_stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Add the batch statistics table to the story
        story.append(Paragraph(f"Overall Statistics for {batch_name} across all Pairs", custom_heading_style))
        story.append(batch_stats_table)
        story.append(Spacer(1, 12))  # Space after batch statistics
        story.append(PageBreak())  # Create a new page for batch statistics

        story.append(Paragraph(f"Pair Statistics for {batch_name}", custom_heading_style))
        # Prepare data for the pairs matrix
        pairs_matrix_data = [
            [Paragraph("Pairs", custom_normal_style)] + [Paragraph(metric.upper(), custom_normal_style) for metric in
                                                          metrics]]
        pairs_list = list(batch_data.keys())  # List of pairs
        # Fill in the matrix with mean, median, std values for each pair and each metric
        for pair in pairs_list:
            if pair in statistics["overall"]:
                continue
            row = [Paragraph(str(pair), custom_normal_style)]  # Pair name at the start of the row
            for metric in statistics["overall"].keys():
                metric_stats = batch_data[pair].get(metric)
                mean = round(metric_stats.get("mean", 0), 4)
                median = round(metric_stats.get("median", 0), 4)
                std = round(metric_stats.get("std", 0), 4)
                row.append(Paragraph(f"M: {mean:.4f}\nMed: {median:.4f}\nStd: {std:.4f}",
                                     custom_normal_style))  # Multi-line display
            pairs_matrix_data.append(row)

        # Calculate optimal column widths for pairs matrix
        pairs_matrix_col_widths = [20] + [30] * len(metrics)  # Adjusted widths for better fit

        # Create the pairs statistics matrix
        pairs_matrix = Table(pairs_matrix_data, colWidths=[50, 50, 50])
        pairs_matrix.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Add the pairs statistics matrix to the story
        story.append(pairs_matrix)
        story.append(Spacer(1, 12))  # Space after pairs matrix

        story.append(PageBreak())  # Create a new page for batch statistics
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

    gaussian_rgb = cv2.GaussianBlur(rgb_image, (9, 9), 10.0)
    gaussian_rgb[gaussian_rgb < 0] = 0
    gaussian_rgb[gaussian_rgb > 1] = 1
    unsharp_rgb = cv2.addWeighted(rgb_image, 1.5, gaussian_rgb, -0.5, 0)
    unsharp_rgb[unsharp_rgb < 0] = 0
    unsharp_rgb[unsharp_rgb > 1] = 1
    # Apply a gamma correction to make the render appear closer to what our eyes would see
    gamma = 1.4
    gamma_corr_rgb = unsharp_rgb ** (1.0 / gamma)
    # Save the RGB image
    rgb_image_path = os.path.join(output_dir, f"{batch_name}_rgb_image.png")
    plt.imsave(rgb_image_path, gamma_corr_rgb)

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


def save_results_to_json(statistics, output_dir):
    """Save metrics and statistics to a JSON file."""

    # Convert any tuple keys in metrics and statistics to strings
    #metrics = convert_tuple_keys_to_strings(metrics)
    statistics = convert_tuple_keys_to_strings(statistics)

    # Save statistics to JSON
    with open(os.path.join(output_dir, "statistics.json"), 'w') as json_file:
        json.dump(statistics, json_file, indent=4)


def store_metrics(metrics, batch_name, batch_patches, band_names):
    ncc_stats, ssim_stats, mi_stats = get_ncc_ssim_for_batch(batch_patches, band_names, {}, {}, {})

    # Store metrics in structured format
    for pair in ncc_stats.keys():  # Assuming ncc_stats has the same keys as channel pairs
        if batch_name not in metrics:
            metrics[batch_name] = {}
        if pair not in metrics[batch_name]:
            metrics[batch_name][pair] = {"ncc": [], "ssim": [], "mi": []}

        metrics[batch_name][pair]["ncc"].extend(ncc_stats[pair])  # Assuming these are lists of values
        metrics[batch_name][pair]["ssim"].extend(ssim_stats[pair])  # Assuming these are lists of values
        metrics[batch_name][pair]["mi"].extend(mi_stats[pair])  # Assuming these are lists of values
