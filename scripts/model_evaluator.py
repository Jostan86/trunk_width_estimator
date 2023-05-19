#!/usr/bin/env python3
from trunk_width_estimator import TrunkWidthEstimator
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class TrunkWidthEvaluator:
    def __init__(self):
        self.width_estimator = TrunkWidthEstimator()

    def evaluate_model_test_data(self, row_num, measurement_num, overwrite=False):
        """
        Evaluate the model on the test data for the given row number. This function will save the results in the
        directory specified by the RESULTS_PATH environment variable.
        Args:
            row_num (): Row number to evaluate the model on.
            measurement_num (): Measurement number to evaluate the model on, useful if you want multiple measurements
                for the same row.
            overwrite (): Whether or not to overwrite the results if they already exist.

        Returns:
            None (prints the results)
        """

        # Get the paths to the ground truth file
        gt_filepath = os.environ.get('DATA_PATH') + "trunk_gt/row_{}_diameters.csv".format(row_num)
        # Set the path to the results directory
        results_path = os.environ.get('RESULTS_PATH')
        # Get the path to the test data that corresponds to the ground truth file
        test_data_directory = os.environ.get('DATASET_PATH') + "row{}/".format(row_num)

        # Check if the test data directory exists
        if not os.path.isdir(test_data_directory):
            print("Test data directory does not exist")
            print(test_data_directory)
            # Throw an error
            raise Exception("Test data directory does not exist")


        # Check if the ground truth file exists
        if not os.path.isfile(gt_filepath):
            print("Ground truth file does not exist")
            print(gt_filepath)
            raise Exception("Test data directory does not exist")

        # Set the save directories
        vis_save_dir = results_path + "row{}/images/measure{}/".format(row_num, measurement_num)
        mask_save_dir = results_path + "row{}/mask/measure{}/".format(row_num, measurement_num)
        widths_save_dir = results_path + "row{}/widths/".format(row_num, measurement_num)

        # Check if the save directories already exist, and if they do, abort
        if os.path.isdir(vis_save_dir) or os.path.isdir(mask_save_dir) and not overwrite:
            print("Save directories already exist, aborting. Set overwrite to True to overwrite")
            raise Exception("Test data directory does not exist")
        else:
            # Create the save directories
            os.makedirs(vis_save_dir)
            os.makedirs(mask_save_dir)

        # Make the widths save directory
        if not os.path.isdir(widths_save_dir):
            os.makedirs(widths_save_dir)

        # Get the ground truth
        ground_truth_pd = pd.read_csv(gt_filepath)
        ground_truth_widths = ground_truth_pd['Average'].to_numpy() / 1000.0

        # empty array to save the widths
        calculated_widths = np.zeros_like(ground_truth_widths)

        count = 0

        # Loop over each png file in the test data directory
        for i, filename in enumerate(os.listdir(test_data_directory)):
            if filename.endswith(".png"):
                count += 1
                # Print the filename and status
                print("Starting sample number " + str(count) + " of " + str(len(os.listdir(test_data_directory))/2))
                print("Filename: " + filename)

                # Get the image and pointcloud
                img = cv2.imread(os.path.join(test_data_directory, filename))
                pc_z_vals = np.load(os.path.join(test_data_directory, filename[:-4] + ".npy"))

                # Get the width of the trunk
                width_est, image_masked, mask = self.width_estimator.get_width(img, pc_z_vals)

                # If there is no mask, skip this image
                if width_est is None:
                    print("No mask found")
                    continue

                # Get the image number by removing the .png
                tree_index = int(filename[:-4]) - 1

                # Save the width
                calculated_widths[tree_index] = width_est

                # Save the masked image
                cv2.imwrite(os.path.join(vis_save_dir, filename), image_masked)

                # Save the mask
                np.save(os.path.join(mask_save_dir, filename[:-4] + ".npy"), mask)

        # Save the calculated widths with trial number
        np.save(os.path.join(widths_save_dir, "widths" + str(measurement_num) + ".npy"), calculated_widths)



    def plot_results(self, row_num, measurement_num):

        gt_filepath = os.environ.get('DATA_PATH') + "trunk_gt/row_{}_diameters.csv".format(row_num)
        save_dir = os.environ.get('RESULTS_PATH') + "row{}/widths/".format(row_num)

        # Get the ground truth
        ground_truth_pd = pd.read_csv(gt_filepath)
        ground_truth_widths = ground_truth_pd['Average'].to_numpy() / 1000.0

        min_t = min(ground_truth_widths)
        max_t = max(ground_truth_widths)
        linex = [min_t, max_t]
        liney = [min_t, max_t]

        # Get the calculated widths
        calculated_widths = np.load(os.path.join(save_dir, "widths" + str(measurement_num) + ".npy"))

        width_error = np.sum(np.abs(np.subtract(calculated_widths, ground_truth_widths)))
        print("Total width error:", width_error)

        plt.plot(ground_truth_widths, calculated_widths, 'g*', linex, liney)
        plt.ylabel('Estimates (m)')
        plt.xlabel('Ground Truth (m)')
        plt.show()


if __name__ == "__main__":
    evaluator = TrunkWidthEvaluator()
    evaluator.evaluate_model_test_data(96, 1, overwrite=False)
    evaluator.plot_results(96, 1)

    # for i in [96, 97, 98]:
    #     for j in range(1, 5):
    #         evaluator.evaluate_model_test_data(i, j)

