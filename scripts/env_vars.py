#!/usr/bin/env python3

import os

# Set paths
# Path to the ground truth dataset folder
os.environ['DATASET_PATH'] = '/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/trunk_width_estimation/data/trunk_gt_data/'
# Path to the results folder (where the results will be saved)
os.environ['RESULTS_PATH'] = '/home/jostan/Downloads/datasets/results/Trial8/'
# Path to the folder with the model data
os.environ['DATA_PATH'] = '/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/trunk_width_estimation/data/'
# Path to mask2former folder
os.environ['M2F_PATH'] = '/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/trunk_width_estimation/Mask2Former/'