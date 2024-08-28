import argparse
import os
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,Image
from evaluation import eval_utils


def save_results_to_pdf(statistics_method1, statistics_method2,path_method_1,path_method_2, output_dir):
    # Create the PDF document
    pdf_file = os.path.join(output_dir, "comparison_report.pdf")
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()

    # Define custom styles
    custom_heading_style = ParagraphStyle(name='CustomHeading', parent=styles['Heading2'], fontSize=12,
                                          textColor=colors.darkblue, alignment=1)  # Center alignment
    custom_normal_style = ParagraphStyle(name='CustomNormal', parent=styles['Normal'], fontSize=8,
                                         textColor=colors.black, spaceAfter=2, alignment=1)  # Center alignment

    story = []

    # Title Page
    story.append(Paragraph("SAMSON", styles['Title']))
    story.append(Spacer(1, 12))  # Spacing between title and subtitle
    story.append(Paragraph("Micasense Registration Comparison Report", styles['Title']))
    story.append(Spacer(1, 48))  # More space before generated report
    story.append(Paragraph("Generated Report", custom_normal_style))
    story.append(Spacer(1, 48))  # More space before the next section

    # Overall Stats Table for both methods side by side
    story.append(Paragraph("Overall Statistics", custom_heading_style))

    # Prepare data for the overall statistics tables
    overall_data_method1 = [
        [Paragraph("Metric", custom_normal_style),
         Paragraph("Mean", custom_normal_style),
         Paragraph("Median", custom_normal_style),
         Paragraph("Std Dev", custom_normal_style)]
    ]

    overall_data_method2 = [
        [Paragraph("Metric", custom_normal_style),
         Paragraph("Mean", custom_normal_style),
         Paragraph("Median", custom_normal_style),
         Paragraph("Std Dev", custom_normal_style)]
    ]

    # Iterate through the statistics to fill in the overall data
    for metric in statistics_method1["overall"]:
        method1_stats = statistics_method1["overall"][metric]
        method2_stats = statistics_method2["overall"].get(metric, {})
        overall_data_method1.append([
            Paragraph(metric.upper(), custom_normal_style),
            Paragraph(f"{method1_stats.get('mean', 'N/A'):.4f}", custom_normal_style),
            Paragraph(f"{method1_stats.get('median', 'N/A'):.4f}", custom_normal_style),
            Paragraph(f"{method1_stats.get('std', 'N/A'):.4f}", custom_normal_style),
        ])
        overall_data_method2.append([
            Paragraph(metric.upper(), custom_normal_style),
            Paragraph(f"{method2_stats.get('mean', 'N/A'):.4f}", custom_normal_style),
            Paragraph(f"{method2_stats.get('median', 'N/A'):.4f}", custom_normal_style),
            Paragraph(f"{method2_stats.get('std', 'N/A'):.4f}", custom_normal_style),
        ])

    # Create the overall statistics tables
    overall_table_method1 = Table(overall_data_method1, colWidths=[80, 50, 50, 50])
    overall_table_method2 = Table(overall_data_method2, colWidths=[80, 50, 50, 50])

    # Style for both tables
    for table in [overall_table_method1, overall_table_method2]:
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),  # Reduce font size
        ]))

    # Create a new table to hold both overall statistics tables side by side
    combined_overall_table = Table([[overall_table_method1, overall_table_method2]], colWidths=[300, 300])
    combined_overall_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    # Add the combined table to the story
    story.append(combined_overall_table)
    story.append(Spacer(1, 12))  # Space after first table

    # Add a page for pair statistics matrix
    story.append(PageBreak())
    story.append(Paragraph("Pair Statistics Matrix", custom_heading_style))

    # Prepare data for the pairs matrix for both methods
    metrics = ['ncc', 'ssim', 'mi']
    pairs_list = list(statistics_method1["pairs"].keys())

    # Define a limit for the number of rows per page
    rows_per_page = 5  # Reduce the number of rows displayed per page

    # Loop through pairs in chunks to handle pagination
    for i in range(0, len(pairs_list), rows_per_page):
        # Prepare data for this chunk of pairs
        pairs_matrix_data_method1 = [
            [Paragraph("Pairs", custom_normal_style)] + [Paragraph(metric.upper(), custom_normal_style) for metric in metrics]
        ]
        pairs_matrix_data_method2 = [
            [Paragraph("Pairs", custom_normal_style)] + [Paragraph(metric.upper(), custom_normal_style) for metric in metrics]
        ]

        # Fill in the matrix for the current chunk
        for pair in pairs_list[i:i + rows_per_page]:
            row_method1 = [Paragraph(str(pair), custom_normal_style)]
            row_method2 = [Paragraph(str(pair), custom_normal_style)]
            for metric in metrics:
                method1_stats = statistics_method1["pairs"][pair].get(metric, {})
                method2_stats = statistics_method2["pairs"][pair].get(metric, {})
                row_method1.append(Paragraph(
                    f"M: {method1_stats.get('mean', 0):.4f}\nMed: {method1_stats.get('median', 0):.4f}\nStd: {method1_stats.get('std', 0):.4f}",
                    custom_normal_style))
                row_method2.append(Paragraph(
                    f"M: {method2_stats.get('mean', 0):.4f}\nMed: {method2_stats.get('median', 0):.4f}\nStd: {method2_stats.get('std', 0):.4f}",
                    custom_normal_style))
            pairs_matrix_data_method1.append(row_method1)
            pairs_matrix_data_method2.append(row_method2)

        # Create the pairs matrices for this chunk
        pairs_matrix_method1 = Table(pairs_matrix_data_method1, colWidths=[80] + [50] * len(metrics))
        pairs_matrix_method2 = Table(pairs_matrix_data_method2, colWidths=[80] + [50] * len(metrics))

        # Style for both matrices
        for table in [pairs_matrix_method1, pairs_matrix_method2]:
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),  # Reduce font size
            ]))

        # Create a combined table to hold both pairs matrices side by side
        combined_pairs_matrix = Table([[pairs_matrix_method1, pairs_matrix_method2]], colWidths=[300, 300])

        # Center the combined pairs matrix table
        combined_pairs_matrix.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Add the combined pairs matrix to the story
        story.append(combined_pairs_matrix)
        story.append(Spacer(1, 12))  # Space after each pairs matrix
        story.append(PageBreak())  # Add a page break after each chunk

    for batch_name, batch in statistics_method1["batches"].items():
        story.append(Paragraph(f"Statistics for {batch_name}", custom_heading_style))

        # Display Method 1 Image
        # Create a table for images
        image_data = []
        image_path_method1 = os.path.expanduser(os.path.join(path_method_1, f"{batch_name}_rgb_image.png"))  # Adjust the path as needed
        if os.path.exists(image_path_method1):
            img_method1 = Image(image_path_method1, width=300, height=200)  # Adjust size as needed
        else:
            print(f"Warning: {image_path_method1} not found.")
            img_method1 = Image("", width=300, height=200)  # Placeholder if the image doesn't exist

        # Display Method 2 Image
        image_path_method2 = os.path.expanduser(
            os.path.join(path_method_2, f"{batch_name}_rgb_image.png"))  # Adjust the path as needed
        if os.path.exists(image_path_method2):
            img_method2 = Image(image_path_method2, width=300, height=200)  # Adjust size as needed
        else:
            print(f"Warning: {image_path_method2} not found.")
            img_method2 = Image("", width=300, height=200)  # Placeholder if the image doesn't exist

            # Append images and add spacing
        image_data.append([img_method1, " ", img_method2])  # Adding an empty string for spacing

            # Create a table for the images
        image_table = Table(image_data, colWidths=[300, 5, 300])  # 20 units for spacing

            # Style the image table
        image_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))

            # Add the image table to the story
        story.append(image_table)
        story.append(Spacer(1, 12))  # Space after each images table
        story.append(Paragraph(f"Overall Statistics for {batch_name} across all Pairs", custom_heading_style))
        # Batch Statistics for Method 1
        batch_stats_data_method1 = [
            [Paragraph("Metric", custom_normal_style),
             Paragraph("Mean", custom_normal_style),
             Paragraph("Median", custom_normal_style),
             Paragraph("Std Dev", custom_normal_style)]
        ]
        for metric in ['ncc', 'ssim', 'mi']:
            stats = batch[metric]
            batch_stats_data_method1.append([
                Paragraph(metric.upper(), custom_normal_style),
                Paragraph(f"{stats['mean']:.4f}", custom_normal_style),
                Paragraph(f"{stats['median']:.4f}", custom_normal_style),
                Paragraph(f"{stats['std']:.4f}", custom_normal_style)
            ])

        batch_stats_table_method1 = Table(batch_stats_data_method1, colWidths=[80, 50, 50, 50])
        batch_stats_table_method1.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Batch Statistics for Method 2
        batch_stats_data_method2 = [
            [Paragraph("Metric", custom_normal_style),
             Paragraph("Mean", custom_normal_style),
             Paragraph("Median", custom_normal_style),
             Paragraph("Std Dev", custom_normal_style)]
        ]
        method2_batch_data = statistics_method2["batches"].get(batch_name, {})
        for metric in ['ncc', 'ssim', 'mi']:
            stats = method2_batch_data.get(metric, {})
            batch_stats_data_method2.append([
                Paragraph(metric.upper(), custom_normal_style),
                Paragraph(f"{stats.get('mean', 'N/A'):.4f}", custom_normal_style),
                Paragraph(f"{stats.get('median', 'N/A'):.4f}", custom_normal_style),
                Paragraph(f"{stats.get('std', 'N/A'):.4f}", custom_normal_style)
            ])

        batch_stats_table_method2 = Table(batch_stats_data_method2, colWidths=[80, 50, 50, 50])
        batch_stats_table_method2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Combine batch statistics tables for both methods
        combined_batch_stats_table = Table([[batch_stats_table_method1, batch_stats_table_method2]],
                                           colWidths=[300, 300])
        combined_batch_stats_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        story.append(combined_batch_stats_table)
        story.append(Spacer(1, 12))


        # Define a limit for the number of rows per page
        rows_per_page = 5  # Adjust as necessary

        # Fill in the matrix with mean, median, std values for each pair and each metric for Method 1
        pairs_list = list(statistics_method1["pairs"].keys())
        for i in range(0, len(pairs_list), rows_per_page):
            # Prepare data for this chunk of pairs
            pairs_matrix_data_method1_batch = [
                [Paragraph("Pairs", custom_normal_style)] + [Paragraph(metric.upper(), custom_normal_style) for metric
                                                             in metrics]
            ]
            pairs_matrix_data_method2_batch = [
                [Paragraph("Pairs", custom_normal_style)] + [Paragraph(metric.upper(), custom_normal_style) for metric
                                                             in metrics]
            ]

            # Fill in the matrix for the current chunk
            for pair in pairs_list[i:i + rows_per_page]:
                row_method1 = [Paragraph(str(pair), custom_normal_style)]
                row_method2 = [Paragraph(str(pair), custom_normal_style)]
                for metric in metrics:
                    method1_stats = statistics_method1["batches"][batch_name][pair].get(metric, {})
                    method2_stats = statistics_method2["batches"][batch_name][pair].get(metric, {})
                    row_method1.append(Paragraph(
                        f"M: {method1_stats.get('mean', 0):.4f}\nMed: {method1_stats.get('median', 0):.4f}\nStd: {method1_stats.get('std', 0):.4f}",
                        custom_normal_style))
                    row_method2.append(Paragraph(
                        f"M: {method2_stats.get('mean', 0):.4f}\nMed: {method2_stats.get('median', 0):.4f}\nStd: {method2_stats.get('std', 0):.4f}",
                        custom_normal_style))
                pairs_matrix_data_method1_batch.append(row_method1)
                pairs_matrix_data_method2_batch.append(row_method2)

            # Create the pairs matrices for this chunk
            pairs_matrix_method1_batch = Table(pairs_matrix_data_method1_batch, colWidths=[80] + [50] * len(metrics))
            pairs_matrix_method2_batch = Table(pairs_matrix_data_method2_batch, colWidths=[80] + [50] * len(metrics))

            # Style for both matrices
            for table in [pairs_matrix_method1_batch, pairs_matrix_method2_batch]:
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),  # Reduce font size
                ]))

            # Create a combined table to hold both pairs matrices side by side
            combined_pairs_matrix_batch = Table([[pairs_matrix_method1_batch, pairs_matrix_method2_batch]], colWidths=[300, 300])

            # Center the combined pairs matrix table
            combined_pairs_matrix_batch.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))

            # Add the combined pairs matrix to the story
            story.append(combined_pairs_matrix_batch)
            story.append(Spacer(1, 12))  # Space after each combined matrix

            # Add a page break after each chunk
            story.append(PageBreak())

# Build the PDF
    doc.build(story)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate a PDF report comparing two methods.")
    parser.add_argument('method1', type=str, help='Path to the evaluation results from Method 1')
    parser.add_argument('method2', type=str, help='Path to the evaluation results from Method 2')
    parser.add_argument('output_dir', type=str, default="../output/eval/compare", help='Output directory for the PDF report')

    args = parser.parse_args()
    path_method_1 = args.method1
    path_method_2 = args.method2
    output_dir = os.path.expanduser(args.output_dir)
    eval_utils.create_output_directory(output_dir)

    # Load statistics for two methods
    statistics_file = "statistics.json"
    method1_json = os.path.expanduser(os.path.join(path_method_1, statistics_file))
    method2_json = os.path.expanduser(os.path.join(path_method_2, statistics_file))

    # Debugging: Print the paths
    print("Method 1 JSON path:", method1_json)
    print("Method 2 JSON path:", method2_json)

    # Check if files exist
    if not os.path.exists(method1_json):
        print(f"Warning: {method1_json} does not exist.")

    if not os.path.exists(method2_json):
        print(f"Warning: {method2_json} does not exist.")

    # Load statistics if files exist
    if os.path.exists(method1_json):
        with open(method1_json, 'r') as file:
            method1_stats = json.load(file)  # Load statistics for Method 1

    if os.path.exists(method2_json):
        with open(method2_json, 'r') as file:
            method2_stats = json.load(file)  # Load statistics for Method 2

    # Save results to PDF
    save_results_to_pdf(method1_stats, method2_stats,path_method_1, path_method_2,output_dir)