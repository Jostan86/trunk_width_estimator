from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from env_vars import *

dataset_root = os.environ.get('DATASET_PATH')
data_root = os.environ.get('DATA_PATH')

def evaluate_row(row_num, num_measurements, large_error=0.03):
    """
    Generate the needed information for a row of trees.

    Args:
        row_num (): Row number of trees to evaluate
        large_error (): Threshold for large error in meters

    Returns: Dictionary of information for the row containing:
        abs_error (): Absolute error in meters
        widths_mode (): Mode of the widths across the 5 measurements in meters
        gt_widths (): Ground truth widths in meters
        error_worst (): Indexes of the 5 larges errors
        nan_mask (): Mask of the columns with nan values
        large_error_mask (): Mask of the trees with an error larger than the threshold
        num_trees (): Number of trees in the row
        consistent_cols (): List of columns that have the same value in each measurement
        inconsistent_cols (): List of columns that have at least one value that differs from the others
        difference (): Largest difference between the values in the inconsistent columns
        row_num (): Row number of the trees

    """

    results_root = os.environ.get('RESULTS_PATH')
    data_root = os.environ.get('DATA_PATH')

    # Load 5 .npy files into one array
    for i in range(1, num_measurements + 1):
        if i == 1:
            widths = np.load(results_root + "row{}".format(row_num) + "/widths/widths{}.npy".format(i))
            num_trees = widths.shape[0]
        else:
            widths = np.concatenate((widths, np.load(results_root + "row{}".format(row_num) + "/widths/widths{}.npy".format(i))), axis=0)

    # reshape to 2d array
    widths = np.reshape(widths, (num_measurements, num_trees))

    # Set nan values to 0 and record the column it was in
    nan_cols = []
    for i in range(widths.shape[1]):
        if np.isnan(widths[:, i]).any():
            nan_cols.append(i)
            widths[:, i] = np.nan_to_num(widths[:, i], nan=0.0)

    # Make a nan mask
    nan_mask = np.array([False] * num_trees)
    # Set the values of the nan columns to True
    nan_mask[nan_cols] = True

    # Keep mode of each column, record which columns have all the same values
    widths_mode = np.zeros(num_trees)
    consistent_cols = []
    inconsistent_cols = []
    difference = []
    for i in range(num_trees):
        current_row = widths[:, i]
        # Multiply by 1000000 to convert to micrometers
        current_row = np.round(current_row * 1000000, 0)
        # Convert to int
        current_row = current_row.astype(int)
        # Get mode
        widths_mode[i] = np.bincount(current_row).argmax() / 1000000.0
        # Check if all values are the same, if so, add to consistent_cols, else add to inconsistent_cols and record the difference
        if np.all(current_row == current_row[0]):
            consistent_cols.append(i)
        else:
            inconsistent_cols.append(i)
            difference.append((np.max(current_row) - np.min(current_row))/ 1000000.0)

    # Load the ground truth data
    df = pd.read_csv(data_root + "trunk_gt/row_{}_diameters.csv".format(row_num))
    gt_widths = df['Average'].to_numpy() / 1000.0

    # Save a list of the index of the 5 largest errors, ignoring the nan values
    error = np.abs(np.subtract(widths_mode, gt_widths))
    error_worst = error[:]
    error_worst[nan_mask] = 0.0
    error_worst = np.argsort(error_worst)
    error_worst = error_worst[-5:]
    error_worst = error_worst[::-1]

    # Make a mask of errors larger than 3cm
    large_error_mask = error > large_error

    # Save the data to a dictionary
    return_dict = {"abs_error": error, "consistent_cols": consistent_cols, "inconsistent_cols": inconsistent_cols,
                   "difference": difference, "nan_mask": nan_mask, "widths_mode": widths_mode, "gt_widths": gt_widths,
                   "num_trees": num_trees, "error_worst": error_worst, "large_error_mask": large_error_mask,
                   "row_num": row_num}

    return return_dict

def plot_in_pdf(widths, widths_gt, x, y, pdf, plot_num, scale=1.0):
    """
    Plots the widths vs the ground truth widths and places it in the pdf
    Args:
        widths (): Predicted widths
        widths_gt (): Ground truth widths
        x (): x coordinate of the plot
        y (): y coordinate of the plot
        pdf (): pdf object to place the plot in
        plot_num (): Plot number for saving, should be unique for each plot, if not the same plot will be made each
        time.
        scale (): Scale of the plot, default is 1.0

    Returns: None

    """
    # Plot needs to be at least 5x5, but can be larger if possible
    fig_w = 5 if scale < 1.0 else 5 * scale
    fig_h = 5 if scale < 1.0 else 5 * scale

    fig = plt.figure(figsize=(fig_h, fig_w))
    ax = fig.add_subplot(111)

    # Get min and max of the ground truth widths and set up the truth line
    min_gt = min(widths_gt)
    max_gt = max(widths_gt)
    truth_linex = [min_gt, max_gt]
    truth_liney = [min_gt, max_gt]

    plt.plot(widths_gt, widths, 'g*', truth_linex, truth_liney)
    plt.ylabel('Estimates (m)')
    plt.xlabel('Ground Truth (m)')
    ax.set_title("Predicted vs Ground Truth Widths")
    fig.savefig("temp{}.png".format(plot_num))

    # Get the width and height of the image
    img_width = fig_w * 100 * scale if scale < 1.0 else fig_w * 100
    img_height = fig_h * 100 * scale if scale < 1.0 else fig_h * 100
    y -= img_height

    # Place the image in the pdf
    pdf.drawImage("temp{}.png".format(plot_num), x, y, width=img_width, height=img_height)

    # Erase temp file
    os.remove("temp{}.png".format(plot_num))

def place_image(pdf, x, y, row_num, tree_num, scale=0.37, measure_num=1):
    """
    Places an image in the pdf

    Args:
        pdf (): pdf to place image in
        x (): x value to place image at
        y (): y value to place image at
        row_num (): Tree row number
        tree_num (): Tree number
        scale (): Image scale
        measure_num (): Measurement file number to use

    Returns: None: Places image in pdf

    """

    results_root = os.environ.get('RESULTS_PATH')
    image_path = results_root + "/row{}/images/measure{}/{}".format(row_num, measure_num, tree_num) + '.png'
    scale_96 = scale * 0.75
    scale_97 = scale
    width = 640 * scale_96 if row_num == 96 else 640 * scale_97
    height = 480 * scale_96 if row_num == 96 else 360 * scale_97

    pdf.drawImage(image_path, x, y, width=width, height=height)


def section1(pdf, row_info_all, info):
    """Makes the first page of the pdf, which contains the statistics of the model"""

    # Draw the title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, 750, "CV Model Report - " + info)

    plot_num = 0
    x = 30
    for row_info in row_info_all:

        # Setup the statistics
        statistics = {
            "Row": row_num,
            "Number of Trees": row_info["num_trees"],
            "Mean Error (mm)": np.round(np.mean(row_info["abs_error"]) * 1000, 2),
            "Total Error (mm)": np.round(np.sum(row_info["abs_error"]) * 1000, 1),
            "Std Dev (mm)": np.round(np.std(row_info["abs_error"]) * 1000, 1),
            "Num nans": np.sum(row_info["nan_mask"]),
            "Num large errors (> 30mm)": np.sum(row_info["large_error_mask"])
        }

        # Draw the statistics
        pdf.setFont("Helvetica", 12)
        y = 700
        for stat, value in statistics.items():
            pdf.drawString(x, y, f"{stat}: {value}")
            y -= 18

        # Plot the widths vs ground truth widths
        plot_in_pdf(row_info["widths_mode"][~row_info["nan_mask"] & ~row_info["large_error_mask"]],
                    row_info["gt_widths"][~row_info["nan_mask"] & ~row_info["large_error_mask"]],
                    x-10, y, pdf, plot_num, scale=0.35)
        plot_num += 1
        x += 190

    # Calculate average stats across all rows
    # Put all errors, widths, and ground truth widths in their own array
    all_errors = []
    all_widths = []
    all_gt_widths = []
    for row_info in row_info_all:
        all_errors.append(row_info["abs_error"])
        all_widths.append(row_info["widths_mode"][~row_info["nan_mask"] & ~row_info["large_error_mask"]])
        all_gt_widths.append(row_info["gt_widths"][~row_info["nan_mask"] & ~row_info["large_error_mask"]])
    all_errors = np.concatenate(all_errors, axis=0)
    all_widths = np.concatenate(all_widths, axis=0)
    all_gt_widths = np.concatenate(all_gt_widths, axis=0)

    y -= 240
    x = 30

    pdf.drawString(x, y, "All Rows:")
    y -= 20
    pdf.drawString(x, y, f"Mean Total: {np.round(np.mean(all_errors) * 1000, 2)}")
    y -= 20
    pdf.drawString(x, y, f"Std Dev Total: {np.round(np.std(all_errors) * 1000, 2)}")
    y -= 20

    y += 110
    x += 130
    plot_in_pdf(all_widths, all_gt_widths, x, y, pdf, plot_num, scale=0.7)
    plot_num += 1

def section2(pdf, row_info_all):
    """Makes the second page of the pdf, which contains the largest errors for each row"""
    x_img = 300
    x_text = 30

    for row_info in row_info_all:

        y_img = 730
        y_text = 750
        # Add a page
        pdf.showPage()

        # Draw the title
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(x_text, y_text, "Largest Errors Row {}".format(row_info["row_num"]))

        y_text -= 50
        # Loop over the 5 largest errors
        for i, error_idx in enumerate(row_info["error_worst"]):

            statistics = {
                "Tree Num": error_idx + 1,
                "Width Estimate (mm)": np.round(row_info["widths_mode"][error_idx] * 1000, 2),
                "Ground Truth (mm)": np.round(row_info["gt_widths"][error_idx] * 1000, 2),
                "Error (mm)": np.round(row_info["abs_error"][error_idx] * 1000, 2),
                "Depth (m)": "Not available"
            }

            # Draw the statistics
            pdf.setFont("Helvetica", 12)
            for stat, value in statistics.items():
                pdf.drawString(x_text, y_text, f"{stat}: {value}")
                y_text -= 20

            # Show the image
            y_img -= 140

            place_image(pdf, x_img, y_img, row_info['row_num'], error_idx + 1, scale=0.37)
            y_text -= 40

    pdf.showPage()

def section3(pdf, row_info_all):
    """Make the third section of the report, which contains the trees with nan values for the width"""

    x_text = 30
    x_img = 300

    # Draw the title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(x_text, 750, "Unable to Find Width for These Trees")

    y_img = 590
    y_text = 700
    for row_info in row_info_all:

        # If no nan values in nan mask, skip
        if not np.any(row_info["nan_mask"]):
            continue

        # Get indexes of nan values from nan mask
        nan_idxs = np.where(row_info["nan_mask"])[0]

        # loop over the trees with nan values
        for nan_idx in nan_idxs:
            if y_img < 100:
                y_img = 590
                y_text = 700

                pdf.showPage()
                pdf.setFont("Helvetica-Bold", 16)
                pdf.drawString(x_text, 750, "Unable to Find Width for These Trees")

            statistics = {
                "Row": row_info["row_num"],
                "Tree Num": nan_idx + 1,
                "Width Estimate (mm)": "Not available",
                "Ground Truth (mm)": np.round(row_info["gt_widths"][nan_idx] * 1000, 2),
                "Error (mm)": "Not available",
                "Depth (m)": "Not available"
            }

            # Draw the statistics
            pdf.setFont("Helvetica", 12)
            for stat, value in statistics.items():
                pdf.drawString(x_text, y_text, f"{stat}: {value}")
                y_text -= 20

            place_image(pdf, x_img, y_img, row_num, nan_idx + 1, scale=0.37)
            y_img -= 160
            y_text -= 40

    y_text = 700

    pdf.showPage()

def section4(pdf, row_info_all):
    """Make the fourth section of the report, which shows the trees with inconsistent widths across measurements"""
    x_text = 30

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(x_text, 750, "Inconsistent Widths")

    x1 = 30
    x2 = 200
    x3 = 400
    pdf.setFont("Helvetica", 12)
    pdf.drawString(x1, 700, "Row")
    pdf.drawString(x2, 700, "Num Inconsistent")
    pdf.drawString(x3, 700, "Num Consistent")
    y = 680
    for row_info in row_info_all:

        pdf.drawString(x1, y, str(row_info["row_num"]))
        pdf.drawString(x2, y, str(len(row_info["inconsistent_cols"])))
        pdf.drawString(x3, y, str(len(row_info["consistent_cols"])))
        y -= 20

    y -= 20

    pdf.setFont("Helvetica", 12)
    pdf.drawString(x1, y, "Row")
    pdf.drawString(x2, y, "Tree Num")
    pdf.drawString(x3, y, "Max Difference")
    y -= 20

    for row_info in row_info_all:

        for (tree_num, max_diff) in zip(row_info["inconsistent_cols"], row_info["difference"]):
            if y < 80:
                y = 750
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                pdf.drawString(x1, y, "Row")
                pdf.drawString(x2, y, "Tree Num")
                pdf.drawString(x3, y, "Max Difference (mm)")
                y -= 20
            pdf.drawString(x1, y, str(row_info["row_num"]))
            pdf.drawString(x2, y, str(tree_num + 1))
            pdf.drawString(x3, y, str(np.round(max_diff * 1000, 2)))
            y -= 20

if __name__ == "__main__":

    row_info_saved = []
    for row_num in [97, 98]:
        # Get the row info
        row_info_saved.append(evaluate_row(row_num, num_measurements=1))

    # Get the path to save the PDF
    pdf_path = os.environ.get("RESULTS_PATH") + "report.pdf"

    # Create a new PDF document
    pdf = canvas.Canvas(pdf_path, pagesize=letter)
    # Page is 612 x 792

    info = " "
    section1(pdf, row_info_saved, info)
    section2(pdf, row_info_saved)
    section3(pdf, row_info_saved)
    section4(pdf, row_info_saved)

    # Save the PDF document at the given path, in the results folder
    pdf.save()