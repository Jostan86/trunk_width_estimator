#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor

from skimage.measure import find_contours
import glob
from env_vars import *

import numpy as np
from scipy.spatial import distance
from skimage import morphology
from skimage import measure
class TrunkAnalyzer:
    def __init__(self):

        project_path = os.environ.get('M2F_PATH')

        # Add Mask2Former to sys.path so that we can import it
        sys.path.insert(1, project_path)
        from mask2former import add_maskformer2_config

        # Path to the config file of the pretrained model, the weight file, and the datasets
        config_file = project_path + "/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
        weight_file = os.environ.get('DATA_PATH') + "/output20220612_5instances/model_final.pth"

        # Create a configuration object and merge in settings from the specified config files
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_file)

        # Load trained weights from file
        cfg.MODEL.WEIGHTS = weight_file

        # Set number of output classes to 1 and use batch normalization
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        cfg.MODEL.RESNETS.NORM = "BN"

        # Freeze the configuration to prevent any further changes
        cfg.freeze()

        # Initialize predictor using the configuration object
        self.predictor = DefaultPredictor(cfg)

        self.predictions = None
        self.image = None
        self.pointcloud = None

        self.scores = None
        self.masks = None
        self.outputs_kept = None

        self.depth_calculated = False
        self.depth_median = None
        self.depth_percentile = None

        self.width_calculated = False
        self.tree_widths = None

        self.straightness_metrics_calculated = False
        self.angle_diff = None
        self.mean_deviation_sum = None
        self.mean_length = None
        self.mean_deviation_left = None
        self.mean_deviation_right = None
        self.lines = None

        self.color_calculated = False
        self.hsv_value_channel = None

        self.num_instances = None
        self.width = None
        self.height = None
        self.num_pixels = None

        self.classification_calculated = False
        self.classification = None

        self.tree_locations = None

        self.save_count = 0

    def get_mask(self, image):
        """
        Do the prediction and save the mask and score variables
        Args:
            image (): Image that was sent to the model.

        Returns:
            saves the mask and score arrays to the class variables, as well as the number of instances, and the indices
            of the instances that were kept
        """
        # Do the prediction and convert to cpu
        outputs = self.predictor(image)
        self.predictions = outputs["instances"].to("cpu")

        # Save the mask and score arrays to the class variables
        self.scores = self.predictions.scores.numpy() if self.predictions.has("scores") else None
        self.masks = self.predictions.pred_masks.numpy() if self.predictions.has("pred_boxes") else None

        # Save the number of instances
        self.num_instances = len(self.masks) if self.scores is not None else None
        # Initialize the array that will hold the indices of the instances that are being kept
        self.outputs_kept = np.arange(self.num_instances) if self.num_instances is not None else None

    def update_arrays(self, keep_indices):
        """Update all the arrays based on the keep_indices array from a filtering operation.

        Args:
            keep_indices (): Indices of the instances that are being kept.

        Returns:
            Filters all the active arrays based on the keep_indices array.
        """

        self.scores = self.scores[keep_indices]
        self.masks = self.masks[keep_indices]
        self.outputs_kept = self.outputs_kept[keep_indices]
        self.num_instances = len(self.masks)

        if self.depth_calculated:
            self.depth_median = self.depth_median[keep_indices]
            self.depth_percentile = self.depth_percentile[keep_indices]
            self.tree_locations = self.tree_locations[keep_indices]
        if self.width_calculated:
            self.tree_widths = self.tree_widths[keep_indices]
        if self.straightness_metrics_calculated:
            self.angle_diff = self.angle_diff[keep_indices]
            self.mean_deviation_sum = self.mean_deviation_sum[keep_indices]
            self.mean_length = self.mean_length[keep_indices]
            self.mean_deviation_left = self.mean_deviation_left[keep_indices]
            self.mean_deviation_right = self.mean_deviation_right[keep_indices]
        if self.color_calculated:
            self.hsv_value_channel = self.hsv_value_channel[keep_indices]
        if self.classification_calculated:
            self.classification = self.classification[keep_indices]


    def get_st(self, mask):
        """
        Attains the best estimate of the diameter of the tree in pixels.

        Args:
            mask (): Binary mask of the trunk.

        Returns:
            diameter_pixels (): Diameter of the trunk in pixels.
        """

        # The function returns two arrays: medial_axis containing the boolean mask of the medial axis, and return_distance
        # containing the distance transform of the binary mask, which assigns to each pixel the distance to the closest background pixel.
        medial_axis, return_distance = morphology.medial_axis(mask, return_distance=True)

        # Get the number of medial axes in each row
        axes_per_row = medial_axis.sum(axis=1)

        # Find the longest medial axis, where there is only one medial axis in the row for the duration
        pre = 0
        # mlen, mstart, mend = 0, -1, -1
        # tlen, tstart, tend = 0, -1, -1
        # for i in range(axes_per_row.shape[0]):
        #     # If there is one medial axis in the row, then start counting the length of the medial axis
        #     if axes_per_row[i] == 1:
        #         if pre == 0:
        #             pre = 1
        #             tstart = i
        #             tlen = 0
        #         tlen += 1
        #
        #     else:
        #         if axes_per_row[max(i - 1,0)] == 1 and axes_per_row[min(i + 1, image.shape[0] - 1)] == 1:
        #             medial_axis[i] = 1
        #             tlen += 1
        #         elif pre == 1:
        #             pre = 0
        #             if tlen > mlen:
        #                 mlen = tlen
        #                 mend = i
        #                 mstart = tstart
        #
        # if tlen > mlen:
        #     mend = medial_axis.shape[0]
        #     mstart = tstart
        mlen, mstart, mend = 0, -1, -1
        tlen, tstart, tend = 0, -1, -1
        for i in range(axes_per_row.shape[0]):
            # If there is one medial axis in the row, then start counting the length of the medial axis
            if axes_per_row[i] == 1:
                if pre == 0:
                    pre = 1
                    tstart = i
                    tlen = 0
                tlen += 1
            else:
                if axes_per_row[max(i - 1, 0)] == 1 and axes_per_row[min(i + 1, mask.shape[0] - 1)] == 1:
                    medial_axis[i] = 0
                elif pre == 1:
                    pre = 0
                    # print(tlen,tstart,i)
                    # print(mlen,mstart,mend)
                    # print('---')
                    if tlen > mlen:
                        mlen = tlen
                        mend = i
                        mstart = tstart
                    # tlen=0

        if tlen > mlen:
            mlen = tlen
            mend = axes_per_row.shape[0]
            mstart = tstart
            tlen = 0

        # Make 1d mask of the longest medial axis
        b2 = np.zeros_like(axes_per_row)
        b2[mstart:mend] = 1

        # Make 2d mask of the longest medial axis
        medial_axis = medial_axis * b2[:, np.newaxis].astype(bool)
        # Get the distance from the medial axis to the edge of the mask for each row along the medial axis
        return_distance0 = return_distance * medial_axis
        return_distance1 = np.max(return_distance0, axis=1)
        return_distance2 = return_distance1[mstart:mend]

        # Take the cumulative sum, then cut off the first 20% and last 20% of the medial axis
        diff = mend - mstart
        diff2 = int(diff * 0.2)
        return_distance3 = np.cumsum(return_distance2)[:-diff2]
        return_distance3 = return_distance3[diff2:]

        # Take the difference between the cumulative sum and the cumulative sum shifted by 20 pixels
        return_distance4 = return_distance3[20:] - return_distance3[:-20]

        # Find the indices of the 40% of the remaining medial axis with the smallest distance to the edge of the mask
        k = int(return_distance4.shape[0] * 0.4)
        idx1 = np.argpartition(return_distance4, k)[:k]
        real_idx = idx1 + mstart + 10
        real_idx += diff2

        diameter_pixels = return_distance1[real_idx] * 2

        return diameter_pixels

    def calculate_depth(self, top_ignore=0.25, bottom_ignore=0.20, min_num_points=300, depth_filter_percentile=65):
        """
        Calculates the best estimate of the distance between the tree and camera.

        Args:
            top_ignore (): Proportion of the top of the image to ignore mask points in.
            bottom_ignore (): Proportion of the bottom of the image to ignore mask points in.
            min_num_points (): Minimum number of valid pointcloud points needed to keep the mask, if less than this,
            disregard the mask.
            depth_filter_percentile (): Percentile of the depth values to use for the percentile depth estimate. So at
            65, the percentile depth will be farther than 65% of the points in the mask.

        Returns:
            Calculates the median depth and percentile depth for each mask. Also filters out masks that have less than
            min_num_points valid points in the region defined by top_ignore and bottom_ignore.

        """

        # Initialize arrays to store the depth values and the tree locations
        self.depth_median = np.zeros(self.num_instances)
        self.depth_percentile = np.zeros(self.num_instances)
        self.tree_locations = np.zeros((self.num_instances, 2))

        # Make boolean array of indices to keep
        keep = np.ones(self.num_instances, dtype=bool)

        # Replace all nan values with 0
        self.pointcloud = np.where(np.isnan(self.pointcloud), 0, self.pointcloud)

        # Reshape the point cloud array to match the mask dimensions
        reshaped_cloud = self.pointcloud.reshape(-1, 3)

        # Calculate the top and bottom ignore values in pixels
        top_ignore = int(top_ignore * self.height)
        bottom_ignore = self.height - int(bottom_ignore * self.height)

        # Loop through each mask
        for i, mask in enumerate(self.masks):

            # Make copy of mask array
            masked_cloud = mask.copy()

            # Zero out the top and bottom ignore regions
            masked_cloud[:top_ignore, :] = 0
            masked_cloud[bottom_ignore:, :] = 0

            # If there are no points in the mask, remove the segment
            if np.sum(masked_cloud) == 0:
                keep[i] = False
                continue

            # Apply the mask to the reshaped point cloud array
            masked_cloud = reshaped_cloud[masked_cloud.flatten() == 1]

            # Remove rows containing all zero values
            nonzero_points = masked_cloud[~np.all(masked_cloud == 0, axis=1)]

            # If there are less than the min number of points, remove the mask
            if nonzero_points.shape[0] < min_num_points:
                keep[i] = False
                continue

            # Calculate Euclidean distances to the origin
            distances_to_origin = np.linalg.norm(nonzero_points, axis=1)

            # Get the median x value of the points
            median_x = np.median(nonzero_points[:, 0])
            # Get the median z value of the points
            median_z = np.median(nonzero_points[:, 2])

            # Calculate median distance to the origin
            self.depth_median[i] = np.median(distances_to_origin)
            # Calculate the percentile distance to the origin
            self.depth_percentile[i] = np.percentile(distances_to_origin, depth_filter_percentile)
            # Record the x and z values of the tree
            self.tree_locations[i, :] = [median_x, median_z]

        # Update the arrays
        self.depth_calculated = True
        self.update_arrays(keep)


    def calculate_width(self, horz_fov=69.4):
        """
        Calculates the best estimate of the width of the tree in meters.

        Args:
            horz_fov (): Horizontal field of view of the camera in degrees.

        Returns:
            Calculates and stores the width of the tree in meters for each mask.
        """

        self.tree_widths = np.zeros(self.num_instances)

        # Loop through each mask
        for i, (mask, depth) in enumerate(zip(self.masks, self.depth_median)):

            # Get the diameter of the tree in pixels
            diameter_pixels = self.get_st(mask)

            # Calculate the width of the image in meters at the depth of the tree
            image_width_m = depth * np.tan(np.deg2rad(horz_fov / 2)) * 2

            # Calculate the distance per pixel
            distperpix = image_width_m / self.width

            # Calculate the diameter of the tree in meters
            diameter_m = diameter_pixels * distperpix

            # If there are no valid widths, set the width to 0, otherwise set it to the max width
            if len(diameter_m) == 0:
                self.tree_widths[i] = 0
            else:
                self.tree_widths[i] = np.max(diameter_m)

        self.width_calculated = True

    def mask_filter_score(self, score_threshold=0.1):
        """Sort out any outputs that are below the score threshold"""

        if score_threshold != 0:
            keep = self.scores > score_threshold
            self.update_arrays(keep)

    def mask_filter_nms(self, overlap_threshold=0.5):
        """
        Apply non-maximum suppression (NMS) to a set of masks and scores.

        Args:
            overlap_threshold (): Overlap threshold for NMS. If the overlap between two masks is greater than this
            value, the mask with the lower score will be suppressed.

        Returns:
            Updates the class arrays to only include the masks that were not suppressed.
        """

        mask_nms = self.masks.copy()
        score_nms = self.scores.copy()

        # Sort masks by score
        indices = np.argsort(-score_nms)
        mask_nms = mask_nms[indices]

        # Array to keep track of whether an instance is suppressed or not
        suppressed = np.zeros((len(mask_nms)), dtype=bool)

        # For each mask, compute overlap with other masks and suppress overlapping masks if their score is lower
        for i in range(len(mask_nms) - 1):
            # If already suppressed, skip
            if suppressed[i]:
                continue
            # Compute overlap with other masks
            overlap = np.sum(mask_nms[i] * mask_nms[i + 1:], axis=(1, 2)) / np.sum(mask_nms[i] + mask_nms[i + 1:],
                                                                                   axis=(1, 2))
            # Suppress masks that are either already suppressed or have an overlap greater than the threshold
            suppressed[i + 1:] = np.logical_or(suppressed[i + 1:], overlap > overlap_threshold)


        # Get the indices of the masks that were not suppressed
        indices_revert = np.argsort(indices)
        suppressed = suppressed[indices_revert]
        not_suppressed = np.logical_not(suppressed)

        # Update the arrays
        self.update_arrays(not_suppressed)


    def mask_filter_depth(self, depth_threshold=1.5):
        """Sort out any outputs that are beyond a given depth threshold. Note that during depth calcuation any
        segments entirely in the top or bottom portions of the image are removed, and any segments with too few
        points in the point cloud are also removed.

        Args:
            depth_threshold (): Depth threshold in meters. Any masks with a percentile depth greater than this value
            will be removed.

        Returns:
            Updates the class arrays to only include the masks that are within the depth threshold.
        """
        keep = self.depth_percentile < depth_threshold
        self.update_arrays(keep)

    def mask_filter_size(self, large_threshold=0.05, small_threshold=0.01, score_threshold=0.1):
        """Sort out any outputs with masks smaller or larger than the thresholds, based on number of pixels. Also,
        this filter ignores any masks with a score higher than the score threshold"""

        keep = np.zeros(self.num_instances, dtype=bool)

        large_threshold = large_threshold * self.num_pixels
        small_threshold = small_threshold * self.num_pixels

        for i, (score, mask) in enumerate(zip(self.scores, self.masks)):
            if score > score_threshold:
                keep[i] = True
                continue
            area = np.sum(mask)
            if area > large_threshold or area < small_threshold:
                continue
            else:
                keep[i] = True

        self.update_arrays(keep)

    def mask_filter_edge(self, edge_threshold=0.05, size_threshold=0.1):
        """Sort out any outputs with masks that are too close to the edge of the image. Edge threshold is how close
        the mask can be to the edge, as a proportion of the image width. Size threshold is the proportion of the mask
        that must be beyond the edge threshold for the mask to be removed. """

        keep = np.zeros(self.num_instances, dtype=bool)

        edge_threshold = int(edge_threshold * self.width)

        masks_copy = self.masks.copy()

        for i, mask in enumerate(masks_copy):
            left_edge_pixels = mask[:, :edge_threshold].sum()
            right_edge_pixels = mask[:, -edge_threshold:].sum()
            total_mask_pixels = np.sum(mask)
            if left_edge_pixels / total_mask_pixels > size_threshold or right_edge_pixels / total_mask_pixels > size_threshold:
                continue
            else:
                keep[i] = True

        self.update_arrays(keep)

    def mask_filter_position(self, bottom_position_threshold=0.33, score_threshold=0.3):
        """Filter out any masks whose lowest point is above the position threshold. Position threshold is the
        proportion of the image height from the bottom."""

        keep = np.zeros(self.num_instances, dtype=bool)

        bottom_position_threshold = int(bottom_position_threshold * self.height)

        masks_copy = self.masks.copy()

        for i, mask in enumerate(masks_copy):
            if self.scores[i] > score_threshold:
                keep[i] = True
                continue

            bottom_pixels = mask[-bottom_position_threshold:].sum()
            if bottom_pixels  > 0:
                keep[i] = True

        self.update_arrays(keep)
    def mask_filter_stacked(self):
        """Sort out any masks that are stacked vertically, indicating that they are the same object. Keep the trunk
        with the highest score."""

        masks = self.masks.copy()
        scores = self.scores.copy()

        # Sort masks based on scores in descending order
        sorted_indices = np.argsort(scores)[::-1]
        sorted_masks = masks[sorted_indices]

        filtered_masks = []

        # Array to keep track of whether an instance is suppressed or not
        suppressed = np.zeros((len(masks)), dtype=bool)

        for i in range(len(sorted_masks)):

            # Sum the columns into a single row
            mask_sum = np.sum(sorted_masks[i], axis=0)


            for j in range(len(filtered_masks)):
                filtered_sum = np.sum(filtered_masks[j], axis=0)

                if np.any(np.logical_and(mask_sum, filtered_sum)):
                    suppressed[i] = True
                    break

            if not suppressed[i]:
                filtered_masks.append(sorted_masks[i])


        # Restore original ordering of suppressed indices
        indices_revert = np.argsort(sorted_indices)
        suppressed = suppressed[indices_revert]
        not_suppressed = np.logical_not(suppressed)

        self.update_arrays(not_suppressed)

    def calculate_straightness_score(self, contour, y_values, top_remove, bottom_remove, num_points, right):

        # Start a list of the points that will be used to fit the line
        filtered_points = [contour[0]]
        # Set the first y value
        y_val = y_values[0]
        # Set the index of the y value
        j = 0
        # Loop over each point in the contour, and only keep the point with the highest or lowest x value for each y
        # value
        for i in range(len(contour)):
            # If the y value is the same as the current y value, check if the x value is higher or lower than the
            # current x value for that y value, and replace if necessary. If the y value is different, move to the
            # next y value and add the point to the list of points to fit the line to.
            if contour[i, 1] == y_val:
                if right:
                    if contour[i, 0] < filtered_points[j][0]:
                        filtered_points[j] = contour[i]
                else:
                    if contour[i, 0] > filtered_points[j][0]:
                        filtered_points[j] = contour[i]
            else:
                j += 1
                y_val = y_values[j]
                filtered_points.append(contour[i])

        # Convert the list of points to an array
        filtered_points = np.array(filtered_points)

        # Fit a line to the points
        [vx, vy, x0, y0] = cv2.fitLine(filtered_points, cv2.DIST_L1, 0, 0.01, 0.01)

        # Calculate the angle of the line
        angle = abs(np.arctan2(vy, vx) * 180 / np.pi)

        # Get the points along the line of best fit
        # Since we'll remove the top and bottom of the contuor, the points in the line of best fit don't have to extend
        # all the way to the top and bottom of the contour.
        gap = min(top_remove, bottom_remove)
        y_max = int(self.height * (1 - gap))
        y_start = int(self.height * gap)

        # Make points for the line along the line of best fit
        x_step = vx / vy
        x_start = x0 + (y_start - y0) * x_step
        x_end = x0 + (y_max - y0) * x_step
        x = np.linspace(x_start[0], x_end[0], num_points)
        y = np.linspace(y_start, y_max, num_points)
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        xy = np.stack((x, y), axis=1)

        # Calculate the distance between each point in the contour and the line of best fit (this seems like it is
        # probably not the most efficient way to do this, but since it's numpy it seems to do fine)
        distances = distance.cdist(filtered_points, xy)

        # Get the min of each row of distances, so the smallest distance between each point in the contour and the line
        # of best fit
        distances = np.min(distances, axis=1)

        # Add the line of best fit to the list of lines
        self.lines.append(xy)

        return np.mean(distances), angle, len(filtered_points)

    def calculate_straightness_stats(self, top_remove=0.1, bottom_remove=0.05, num_points=1500):
        """Calculate the straightness score for each instance. This is done by removing the top and bottom portions
        of the mask, and then finding the line of best fit along each side of the segment, then finding how much
        deviation there is between the line of best fit and the actual line. The straightness score is the sum of the
        deviation from the left and right sides.

        Args:
            top_remove (): Proportion of the top of the mask to remove.
            bottom_remove (): Proportion of the bottom of the mask to remove.
            num_points (): Number of points to use for the line of best fit.
            """

        # Setup class arrays
        self.lines = []
        self.mean_length = np.zeros(self.num_instances)
        self.mean_deviation_sum = np.zeros(self.num_instances)
        self.angle_diff = np.zeros(self.num_instances)
        self.mean_deviation_left = np.zeros(self.num_instances)
        self.mean_deviation_right = np.zeros(self.num_instances)

        # Loop through each mask and calculate the straightness stats
        for mask_index, mask in enumerate(self.masks):

            # Copy the mask
            contour_mask = mask.copy()

            # Extract contour from the binary mask
            contours, _ = cv2.findContours(contour_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Assuming only one contour is present
            contour = contours[0]

            # Remove axis dimension
            contour = contour.squeeze()

            # Remove the top and bottom portions of the contour, according to the top_remove and bottom_remove values.
            # This is done using the y values of the contour pixels.
            contour_max = contour[:, 1].max()
            contour_min = contour[:, 1].min()
            countour_range = contour_max - contour_min
            contour = contour[contour[:, 1] > contour_min + top_remove * countour_range]
            contour = contour[contour[:, 1] < contour_max - bottom_remove * countour_range]

            # Sort the points based on y values (ascending order)
            contour = contour[contour[:, 1].argsort()]

            # Find the unique y values
            y_values, _ = np.unique(contour[:, 1], return_index=True)

            # Calculate the straightness score for the right and left sides of the contour
            mean_variance_right, angle_right, line_length_right = self.calculate_straightness_score(contour, y_values,
                                                                 top_remove, bottom_remove, num_points, True)

            mean_variance_left, angle_left, line_length_left = self.calculate_straightness_score(contour, y_values,
                                                                   top_remove, bottom_remove, num_points, False)

            # Save the needed values
            self.angle_diff[mask_index] = abs(angle_right - angle_left)
            self.mean_deviation_sum[mask_index] = mean_variance_right + mean_variance_left
            self.mean_length[mask_index] = (line_length_right + line_length_left) / 2
            self.mean_deviation_left[mask_index] = mean_variance_left
            self.mean_deviation_right[mask_index] = mean_variance_right

        self.straightness_metrics_calculated = True

    def find_avg_color(self, image, top_cutoff=0.25, bottom_cutoff=0.1):
        """Convert the image to HSV and find the average color of the mask, excluding the top and bottom portions of the
        mask.

        Args:
            image (): Image to find the average color of.
            top_cutoff (): Proportion of the top of the mask to remove.
            bottom_cutoff (): Proportion of the bottom of the mask to remove.

        Returns:
            Adds the average color to the class array hsv_value_value.
        """

        # Setup class array
        self.hsv_value_channel = np.ones(self.num_instances)

        # Convert the image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert the mask to the appropriate data type

        # Loop through each mask
        for i, mask in enumerate(self.masks):

            color_mask = mask.copy()
            color_mask = color_mask.astype(np.uint8)

            # Remove the top and bottom of the mask
            coordinates = cv2.findNonZero(color_mask)
            top_boundary = coordinates[:, 0, 1].min()
            bottom_boundary = coordinates[:, 0, 1].max()
            range = bottom_boundary - top_boundary
            top_cutoff_pix = int((top_boundary + top_cutoff * range))
            bottom_cutoff_pix = int((bottom_boundary - bottom_cutoff * range))

            color_mask[:top_cutoff_pix, :] = 0
            color_mask[bottom_cutoff_pix:, :] = 0

            # Apply the mask to the HSV image
            masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=color_mask)

            # Calculate the average color values within the masked area
            average_hsv = cv2.mean(masked_hsv, mask=color_mask)

            # Save the value of the hsv value channel
            self.hsv_value_channel[i] = average_hsv[2]
            self.color_calculated = True

    def mask_filter_straightness(self, straightness_threshold=10, score_threshold=0.05):
        """Filter the masks based on the straightness of the left and right sides of the mask.

        Args:
            straightness_threshold (): Threshold for the straightness score. Masks with a straightness score below this
                value will be removed.
            score_threshold (): Threshold for the score. Masks with a score above this value will be kept regardless of
                their straightness score.

        Returns:
            Updates the class arrays to only include the masks that pass the filter.
        """

        keep = np.zeros(self.num_instances, dtype=bool)

        for i, (left_straightness, right_straightness, score) in enumerate(zip(self.mean_deviation_left,
                                                                               self.mean_deviation_right,
                                                                               self.scores)):
            if score > score_threshold:
                keep[i] = True
            elif left_straightness < straightness_threshold and right_straightness < straightness_threshold:
                keep[i] = True

        self.update_arrays(keep)

    def classify_posts_sprinklers(self):
        """Classify the masks as posts or sprinklers based on the straightness score, the average length of the mask,
        and the average width of the mask. Lots of magic numbers here, but they were determined by trial and error."""

        self.classification = np.zeros(self.num_instances)

        for i in range(self.num_instances):

            post_score = 0

            # Add points based on straightness stats
            if self.angle_diff[i] < 2:
                post_score += 1
            if self.mean_deviation_sum[i] < 2.5:
                post_score += 1
            if self.mean_length[i] > 400:
                post_score += 1

            # Add points based on width, if under 0.09, then its not a post
            if self.tree_widths[i] > 0.11:
                post_score += 3
            elif self.tree_widths[i] > 0.09:
                post_score += 1
            else:
                post_score = 0

            if post_score >= 3:
                self.classification[i] = 1
                # continue

            # At this point sprinkler score is decided by the hsv value channel and the width of the mask.
            sprinkler_score = 0
            if self.hsv_value_channel[i] > 130 and self.tree_widths[i] < 0.04:
                self.classification[i] = 2


        self.classification_calculated = True



    def new_image_reset(self, image, pointcloud):
        """ Resets the class variables for a new image, and loads the new image and pointcloud.

        Args:
            image (): Image to be processed
            pointcloud (): Pointcloud that corresponds to the image
        """
        self.image = image
        self.pointcloud = pointcloud

        self.scores = None
        self.masks = None
        self.outputs_kept = None

        self.depth_calculated = False
        self.depth_median = None
        self.depth_percentile = None
        self.tree_locations = None

        self.width_calculated = False
        self.tree_widths = None

        self.straightness_metrics_calculated = False
        self.angle_diff = None
        self.mean_deviation_sum = None
        self.mean_length = None
        self.mean_deviation_left = None
        self.mean_deviation_right = None
        self.lines = None

        self.color_calculated = False
        self.hsv_value_channel = None

        self.classification_calculated = False
        self.classification = None

        self.height = image.shape[0]
        self.width = image.shape[1]
        self.num_pixels = self.height * self.width

        self.get_mask(image)

    def process_image(self):

        if self.masks is None:
            return

        # Send the masks through all the filters, skipping to the end if the number of instances is 0
        self.mask_filter_score(score_threshold=0.001)
        if self.num_instances > 0:
            self.mask_filter_nms(overlap_threshold=0.01)
        if self.num_instances > 0:
            self.calculate_depth(top_ignore=0.25, bottom_ignore=0.20, min_num_points=200) # was 300
        if self.num_instances > 0:
            self.mask_filter_depth(depth_threshold=2.5)
        if self.num_instances > 0:
            self.mask_filter_size(large_threshold=0.05, small_threshold=0.01)
        if self.num_instances > 0:
            self.mask_filter_edge(edge_threshold=0.03)
        if self.num_instances > 1:
            self.mask_filter_stacked()
        if self.num_instances > 0:
            self.calculate_straightness_stats(top_remove=0.1, bottom_remove=0.05, num_points=1000) # was 1500
            self.mask_filter_straightness(straightness_threshold=10, score_threshold=0.05)

            self.find_avg_color(self.image, top_cutoff=0.25, bottom_cutoff=0.1)

            self.calculate_width()

            self.classify_posts_sprinklers()

    def show_filtered_masks(self, image, pointcloud):
        """ Show all the masks and the results after each filter."""

        self.new_image_reset(image, pointcloud)
        if self.masks is None:
            return
        self.show_current_output("og")
        self.mask_filter_score(score_threshold=0.005)
        self.show_current_output("score")

        if self.num_instances > 0:
            self.mask_filter_nms(overlap_threshold=0.01)
            self.show_current_output("Score, NMS")

        if self.num_instances > 0:
            self.calculate_depth(top_ignore=0.05, bottom_ignore=0.05, min_num_points=500, depth_filter_percentile=65)  # was 300
            # self.calculate_depth(top_ignore=0.25, bottom_ignore=0.2, min_num_points=200, depth_filter_percentile=75)
            self.show_current_output("Score, NMS, Depth")

        if self.num_instances > 0:
            self.mask_filter_depth(depth_threshold=2.5)
            self.show_current_output("Score, NMS, Depth, Depth Filter")

        if self.num_instances > 0:
            self.mask_filter_size(large_threshold=0.05, small_threshold=0.01)
            self.show_current_output("Score, NMS, Depth, Depth Filter, Size")

        if self.num_instances > 0:
            self.mask_filter_edge(edge_threshold=0.03)
            self.show_current_output("Score, NMS, Depth, Depth Filter, Size, Edge")

        if self.num_instances > 1:
            self.mask_filter_stacked()
            self.show_current_output("Score, NMS, Depth, Depth Filter, Size, Edge, Stacked")

        if self.num_instances > 0:
            self.mask_filter_position(bottom_position_threshold=0.33, score_threshold=0.9)
            self.show_current_output("Score, NMS, Depth, Depth Filter, Size, Edge, Stacked, Position")

        if self.num_instances > 0:
            self.calculate_straightness_stats(top_remove=0.1, bottom_remove=0.05, num_points=1000)  # was 1500
            self.mask_filter_straightness(straightness_threshold=10, score_threshold=0.05)
            self.show_current_output("Score, NMS, Depth, Depth Filter, Size, Edge, Stacked, Position, Straightness")

            self.find_avg_color(self.image, top_cutoff=0.25, bottom_cutoff=0.1)

            self.calculate_width()

            self.classify_posts_sprinklers()

        # Print some stats about each mask, if there are any, and draw lines of best fit on the image.
        if self.num_instances > 0:

            for i, mask in enumerate(self.masks):

                # Figure out the x location of the mask
                mean_x = np.mean(np.where(mask)[1])

                if self.classification[i] == 0:
                    print("Tree at x = {}".format(mean_x))
                elif self.classification[i] == 1:
                    print("Post at x = {}".format(mean_x))
                elif self.classification[i] == 2:
                    print("Sprinkler at x = {}".format(mean_x))

                print("Width = {}".format(self.tree_widths[i]))
                print("x: {}, z: {}".format(self.tree_locations[i][0], self.tree_locations[i][1]))
                print("--------------------")

        # get mask from output 6, draw lines on it, then draw on image
            for line in self.lines:
                # Define the two endpoints of the line
                point1 = (line[0][0], line[0][1])
                point2 = (line[-1][0], line[-1][1])

                # Reshape the points into the required format
                line_points = np.array([point1, point2], dtype=np.int32).reshape((-1, 1, 2))

                # Draw the line on the canvas
                image = cv2.polylines(image, [line_points], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow('lines', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_current_output(self, label):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        outputs = self.predictions[self.outputs_kept]
        v = Visualizer(image, scale=1.0, )
        out = v.draw_instance_predictions(outputs)

        cv2.imshow(label, out.get_image()[:, :, ::-1])


    def eval_helper(self, image, pointcloud):
        self.new_image_reset(image, pointcloud)
        self.process_image()

        # Get index of lowest absolute medial x value
        min_x_index = np.argmin(np.abs(self.tree_locations[:, 0]))
        keep = np.zeros(self.num_instances, dtype=bool)
        keep[min_x_index] = True
        self.update_arrays(keep)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        v0 = Visualizer(image_rgb, scale=1.0, )
        outputs0 = self.predictions[self.outputs_kept]
        out0 = v0.draw_instance_predictions(outputs0)
        image_masked = out0.get_image()[:, :, ::-1]

        return self.tree_widths[0], image_masked, self.masks[0]

    def ros_helper(self, image, pointcloud):

        self.process_image(image, pointcloud)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.num_instances == 0:
            return None, None, image_rgb

        v0 = Visualizer(image_rgb, scale=1.0, )
        outputs0 = self.predictions[self.outputs_kept]
        out0 = v0.draw_instance_predictions(outputs0)
        image_masked = out0.get_image()[:, :, ::-1]

        return self.tree_widths, self.tree_locations, image_masked

if __name__ == "__main__":
    # estimator.plot_results(num)
    segmenation_model = TrunkAnalyzer()
    post = [3, 4, 20, 31, 62]
    post_98 = [21, 53, 62, 63, 73]
    sprinkler = [33, 44, 59, 68, 74]

    # There is a wierd mask at img 279 of bag file 11

    for i in range(279, 500):
    # for i in sprinkler:
        img_directory = os.environ.get('DATASET_PATH') + 'bag_11/images_pointclouds'
        file_list_img = glob.glob(os.path.join(img_directory, '*_{}.png'.format(i)))
        file_list_pc = glob.glob(os.environ.get('DATASET_PATH') + 'bag_11/images_pointclouds/*_{}.npy'.format(i))

        if len(file_list_img) == 1 and len(file_list_pc) == 1:
            file_path = file_list_img[0]
            pc_path = file_list_pc[0]
            img = cv2.imread(file_path)
            pc = np.load(pc_path)
        else:
            print("Error: More than one file found")
            break
        print(i)
        segmenation_model.save_count = i
        segmenation_model.show_filtered_masks(img, pc)



