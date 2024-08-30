import itertools
import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score

import src.utils as utils
from micasense import capture
from registration.registration_with_depth_matching import set_intrinsic


def load_the_capture(images, micasense_calib, micasense_path):
    image_names = sorted(images)
    image_names = [micasense_path.joinpath(image_name).as_posix() for image_name in image_names]

    thecapture = capture.Capture.from_filelist(image_names)
    set_intrinsic(thecapture, micasense_calib)
    return thecapture, image_names


def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_calibration_data():
    micasense_calib = utils.read_micasense_calib("../calib/micasense_calib.yaml")
    cal_samson_1 = utils.read_basler_calib("../calib/SAMSON1_SAMSON2_stereo.yaml")
    K_L, D_L, P_L, _ = cal_samson_1
    return micasense_calib, P_L


def calculate_ncc_patches(image1, image2):
    assert image1.shape == image2.shape, "Input images must have the same dimensions"

    patches1 = image1
    patches2 = image2

    mean1 = patches1.mean(axis=(1, 2), keepdims=True)
    mean2 = patches2.mean(axis=(1, 2), keepdims=True)
    norm_patch1 = patches1 - mean1
    norm_patch2 = patches2 - mean2

    numerator = np.sum(norm_patch1 * norm_patch2, axis=(1, 2))
    denominator = np.sqrt(np.sum(norm_patch1 ** 2, axis=(1, 2)) * np.sum(norm_patch2 ** 2, axis=(1, 2)))

    ncc_values = numerator / denominator
    ncc_values[np.isnan(ncc_values)] = 0

    return ncc_values


def calculate_ssim_patches(image1, image2):
    assert image1.shape == image2.shape, "Input images must have the same dimensions"

    patches1 = image1
    patches2 = image2

    ssim_values = np.array([
        ssim(p1, p2,data_range=1.0) for p1, p2 in zip(patches1, patches2)
    ])

    return ssim_values


def calculate_mi_patches(image1, image2, bins=256):
    assert image1.shape == image2.shape, "Input images must have the same dimensions"

    patches1 = image1
    patches2 = image2

    mi_values = np.array([
        mutual_info_score(None, None, contingency=np.histogram2d(p1.ravel(), p2.ravel(), bins=bins)[0])
        for p1, p2 in zip(patches1, patches2)
    ])

    return mi_values


def get_stats_for_batch(batch_patches, band_names, ncc_stats, ssim_stats, mi_stats):
    for i, j in itertools.combinations(range(len(batch_patches)), 2):
        image1 = batch_patches[i]
        image2 = batch_patches[j]

        patches_R = image1
        patches_F = image2

        ncc_values = calculate_ncc_patches(patches_R, patches_F)
        ssim_values = calculate_ssim_patches(patches_R, patches_F)
        mi_values = calculate_mi_patches(patches_R, patches_F)
        ncc_stats[(band_names[i], band_names[j])] = ncc_values
        ssim_stats[(band_names[i], band_names[j])] = ssim_values
        mi_stats[(band_names[i], band_names[j])] = mi_values

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

    h_trunc = h - (h % ph)
    w_trunc = w - (w % pw)
    image_truncated = image[:h_trunc, :w_trunc]

    num_patches_h = h_trunc // ph
    num_patches_w = w_trunc // pw

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

    for metric, values in raw_data["overall"].items():
        statistics["overall"][metric] = calculate_stats(raw_data["overall"][metric])

    for pair, metrics in raw_data["pairs"].items():
        statistics["pairs"][pair] = {}
        for metric, values in metrics.items():
            statistics["pairs"][pair][metric] = calculate_stats(values)

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
    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
    }


def save_stack_to_disk(stack, output_dir, filename="stack.npy"):
    file_path = os.path.join(output_dir, filename)
    np.save(file_path, stack)
    print(f"Stack saved to {file_path}")



def save_results_to_pdf(statistics, output_dir):
    # Create the PDF document
    pdf_file = os.path.join(output_dir, "evaluation_report.pdf")
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()

    custom_heading_style = ParagraphStyle(name='CustomHeading', parent=styles['Heading2'], fontSize=14,
                                          textColor=colors.darkblue, alignment=1)
    custom_normal_style = ParagraphStyle(name='CustomNormal', parent=styles['Normal'], fontSize=10,
                                         textColor=colors.black, spaceAfter=2, alignment=1)

    story = []


    story.append(Paragraph("SAMSON", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Micasense Registration Evaluation Report", styles['Title']))
    story.append(Spacer(1, 48))
    story.append(Paragraph("Generated Report", custom_normal_style))
    story.append(Spacer(1, 48))


    story.append(Paragraph("Overall Statistics", custom_heading_style))


    overall_data = [
        [Paragraph("Metric", custom_normal_style),
         Paragraph("Mean", custom_normal_style),
         Paragraph("Median", custom_normal_style),
         Paragraph("Std Dev", custom_normal_style)]
    ]

    for metric, metric_stats in statistics["overall"].items():
        overall_data.append([
            Paragraph(metric.upper(), custom_normal_style),
            Paragraph(f"{metric_stats.get('mean', 'N/A'):.4f}", custom_normal_style),
            Paragraph(f"{metric_stats.get('median', 'N/A'):.4f}", custom_normal_style),
            Paragraph(f"{metric_stats.get('std', 'N/A'):.4f}", custom_normal_style)
        ])

    overall_table = Table(overall_data, colWidths=[80, 50, 50, 50])
    overall_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    story.append(overall_table)
    story.append(Spacer(1, 12))

    story.append(PageBreak())
    story.append(Paragraph("Pair Statistics Matrix", custom_heading_style))

    metrics = ['ncc', 'ssim', 'mi']
    pairs_list = list(statistics["pairs"].keys())

    pairs_matrix_data = [
        [Paragraph("Pairs", custom_normal_style)] + [Paragraph(metric.upper(), custom_normal_style) for metric in
                                                     metrics]]

    for pair in pairs_list:
        row = [Paragraph(str(pair), custom_normal_style)]
        for metric in metrics:
            metric_stats = statistics["pairs"][pair].get(metric, {})
            mean = round(metric_stats.get("mean", 0), 4)
            median = round(metric_stats.get("median", 0), 4)
            std = round(metric_stats.get("std", 0), 4)
            row.append(Paragraph(f"M: {mean:.4f}\nMed: {median:.4f}\nStd: {std:.4f}",
                                 custom_normal_style))
        pairs_matrix_data.append(row)

    pairs_matrix_col_widths = [80] + [50] * len(metrics)


    pairs_matrix = Table(pairs_matrix_data, colWidths=pairs_matrix_col_widths)
    pairs_matrix.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    story.append(pairs_matrix)
    story.append(Spacer(1, 12))

    story.append(PageBreak())


    for batch_name, batch_data in statistics["batches"].items():
        story.append(Paragraph(f"Statistics for {batch_name}", custom_heading_style))

        stack_file = os.path.join(output_dir, f"{batch_name}.npy")
        if not os.path.exists(stack_file):
            print(f"Warning: {stack_file} not found. Skipping {batch_name}.")
            continue

        stack = np.load(stack_file)


        if not display_rgb_image(stack, story, styles, output_dir, batch_name):
            print(f"Error displaying image for batch {batch_name}. Skipping.")

        batch_stats_data = [
            [Paragraph("Metric", custom_normal_style),
             Paragraph("Mean", custom_normal_style),
             Paragraph("Median", custom_normal_style),
             Paragraph("Std Dev", custom_normal_style)]
        ]


        for metric in ['ncc', 'ssim', 'mi']:
            stats = batch_data[metric]
            batch_stats_data.append([
                Paragraph(metric.upper(), custom_normal_style),
                Paragraph(f"{stats['mean']:.4f}", custom_normal_style),
                Paragraph(f"{stats['median']:.4f}", custom_normal_style),
                Paragraph(f"{stats['std']:.4f}", custom_normal_style)
            ])


        batch_stats_table = Table(batch_stats_data, colWidths=[80, 50, 50, 50])
        batch_stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))


        story.append(Paragraph(f"Overall Statistics for {batch_name} across all Pairs", custom_heading_style))
        story.append(batch_stats_table)
        story.append(Spacer(1, 12))
        story.append(PageBreak())

        story.append(Paragraph(f"Pair Statistics for {batch_name} across all Patches", custom_heading_style))

        pairs_matrix_data = [
            [Paragraph("Pairs", custom_normal_style)] + [Paragraph(metric.upper(), custom_normal_style) for metric in
                                                         metrics]]
        pairs_list = list(batch_data.keys())

        for pair in pairs_list:
            if pair in statistics["overall"]:
                continue
            row = [Paragraph(str(pair), custom_normal_style)]
            for metric in statistics["overall"].keys():
                metric_stats = batch_data[pair].get(metric)
                mean = round(metric_stats.get("mean", 0), 4)
                median = round(metric_stats.get("median", 0), 4)
                std = round(metric_stats.get("std", 0), 4)
                row.append(Paragraph(f"M: {mean:.4f}\nMed: {median:.4f}\nStd: {std:.4f}",
                                     custom_normal_style))
            pairs_matrix_data.append(row)


        pairs_matrix = Table(pairs_matrix_data, colWidths=[50, 50, 50])
        pairs_matrix.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        story.append(pairs_matrix)
        story.append(Spacer(1, 12))

        story.append(PageBreak())

    doc.build(story)


def display_rgb_image(stack, story, styles, output_dir, batch_name):
    if stack.shape[2] != 6:
        print(f"Error: Expected stack with 6 channels, but got {stack.shape[2]}.")
        return False

    blue_channel = stack[:, :, 0]
    green_channel = stack[:, :, 1]
    red_channel = stack[:, :, 2]

    def normalize(channel):
        min_val, max_val = np.min(channel), np.max(channel)
        return (channel - min_val) / (max_val - min_val) if max_val > min_val else channel

    blue_normalized = normalize(blue_channel)
    green_normalized = normalize(green_channel)
    red_normalized = normalize(red_channel)

    rgb_image = np.zeros((*blue_channel.shape, 3))
    rgb_image[..., 0] = red_normalized
    rgb_image[..., 1] = green_normalized
    rgb_image[..., 2] = blue_normalized

    gaussian_rgb = cv2.GaussianBlur(rgb_image, (9, 9), 10.0)
    gaussian_rgb[gaussian_rgb < 0] = 0
    gaussian_rgb[gaussian_rgb > 1] = 1
    unsharp_rgb = cv2.addWeighted(rgb_image, 1.5, gaussian_rgb, -0.5, 0)
    unsharp_rgb[unsharp_rgb < 0] = 0
    unsharp_rgb[unsharp_rgb > 1] = 1

    gamma = 1.4
    gamma_corr_rgb = unsharp_rgb ** (1.0 / gamma)

    rgb_image_path = os.path.join(output_dir, f"{batch_name}_rgb_image.png")
    plt.imsave(rgb_image_path, gamma_corr_rgb)

    # Add the image to the PDF
    image = Image(rgb_image_path, width=400, height=300)
    story.append(image)
    story.append(Spacer(1, 12))

    return True


def convert_tuple_keys_to_strings(d):
    if isinstance(d, dict):
        return {str(k): convert_tuple_keys_to_strings(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_tuple_keys_to_strings(item) for item in d]
    else:
        return d


def save_results_to_json(statistics, output_dir):
    statistics = convert_tuple_keys_to_strings(statistics)

    with open(os.path.join(output_dir, "statistics.json"), 'w') as json_file:
        json.dump(statistics, json_file, indent=4)


def store_metrics(metrics, batch_name, batch_patches, band_names):
    ncc_stats, ssim_stats, mi_stats = get_stats_for_batch(batch_patches, band_names, {}, {}, {})


    for pair in ncc_stats.keys():
        if batch_name not in metrics:
            metrics[batch_name] = {}
        if pair not in metrics[batch_name]:
            metrics[batch_name][pair] = {"ncc": [], "ssim": [], "mi": []}

        metrics[batch_name][pair]["ncc"].extend(ncc_stats[pair])
        metrics[batch_name][pair]["ssim"].extend(ssim_stats[pair])
        metrics[batch_name][pair]["mi"].extend(mi_stats[pair])
