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

from env_vars import *

class TrunkSegmenter:
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

    def get_mask(self, image, score_thresh, height_thresh=0.25):
        """
        Get the best mask from the predictions
        Args:
            predictions (): Predictions from the model.
            score_thresh (): Min score threshold for predictions.
            image (): Image that was sent to the model.
            height_thresh (): Fraction of the image height that the mask must be above.

        Returns:
            image (): Image with the mask drawn on it.
            mask (): Best mask from the predictions.
            score (): Score of the best mask.
        """
        outputs = self.predictor(image)

        predictions = outputs["instances"].to("cpu")

        # Extract scores
        scores = predictions.scores if predictions.has("scores") else None
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        # Create index tensor of predictions to keep
        if score_thresh != 0:
            keep = scores > score_thresh
            scores = scores[keep]
            boxes = boxes[keep]

        empty_img = np.zeros_like(image[:, :, ::-1])
        v = Visualizer(empty_img, scale=1.0, )

        # Convert masks to numpy arrays and keep only those above score threshold
        if predictions.has("pred_masks"):
            masks = predictions.pred_masks.numpy()[keep]
            # masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]

        index_use = None

        # If no scores are above threshold, return empty results
        if len(masks) == 0:
            return image,np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),-1,-1,-1

        # Apply NMS to remove overlapping predictions with lower score
        scores = scores.numpy()

        masks, scores = self.nms(scores, masks)

        # Get the index of the mask with the highest score
        index_use = np.argmax(scores)

        colors = [255, 255, 255]
        # Get the contours of the mask
        mask = masks[index_use]
        contours = self.get_mask_contours(mask)
        # Draw the contours on the image
        image = self.draw_mask(image, contours, colors)

        return image, mask, scores[index_use]

        # # (720, 1280, 3)
        # h, w, _ = image.shape
        # image_midpoint = w / 2.0
        # max_offset = 1000000
        # for i in range(len(masks)):
        #     # Get boolean array that is true for columns where mask has at least one True value
        #     mask_columns_bool = np.max(masks[i],axis=0)
        #     # Get the pixel positions of the columns with a true value
        #     mask_columns = np.nonzero(mask_columns_bool)
        #     # Get the mean of the pixel positions, so the center of the mask
        #     mask_midpoint = np.mean(mask_columns)
        #     # Get boolean array that is true for row where mask has at least one True value
        #     mask_row_bool = np.max(masks[i],axis=1)
        #     # Calculate how far the center of the mask is from the center of the image
        #     trunk_offset = abs(image_midpoint - mask_midpoint)
        #     # Presently, it ignores masks that are not at least 25% of the image height, and uses the one that is closest to
        #     # the center.
        #     if sum(mask_row_bool) < image.shape[0] * height_thresh:continue
        #     if trunk_offset < max_offset:
        #         max_offset = trunk_offset
        #         index_use = i
        #
        # if index_use is None:
        #     return image,np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),-1,-1,-1
        #
        # # Setup the colors for the masks
        # colors = [255, 255, 255]
        # # Get the contours of the mask
        # mask = masks[index_use]
        # contours = self.get_mask_contours(mask)
        # # Draw the contours on the image
        # image = self.draw_mask(image, contours, colors)
        #
        # return image, mask, scores[index_use]



            # (720, 1280, 3)
        # id = 0
        # h, w, _ = image.shape
        # mid_w = w / 2.0
        # m_distance = 1000000
        # for i in range(len(masks)):
        #     a = np.max(masks[i].mask, axis=0)
        #     a1 = np.nonzero(a)
        #     a2 = np.mean(a1)
        #     b = np.max(masks[i].mask, axis=1)
        #     t_distance = abs(mid_w - a2)
        #     if sum(b) < image.shape[0] * 0.35: continue
        #     if t_distance < m_distance:
        #         m_distance = t_distance
        #         id = i
        #
        # colors = [255, 255, 255]
        # contours = self.get_mask_contours(masks[id].mask)
        # cnt2 = 0
        # for cnt in contours:
        #     cv2.polylines(image, [cnt], True, colors, 2)
        #     image = self.draw_mask(image, [cnt], colors)
        #
        # bt = boxes.tensor[id].numpy().astype('int')
        # start_point = (bt[0], bt[1])
        # end_point = (bt[2], bt[3])
        # delta = 20
        # btxm = bt[0]
        # if min(bt[3], bt[1]) > image.shape[0] - max(bt[3], bt[1]):
        #     btym = int((bt[3] + bt[1]) / 2)
        #     btxm = bt[2]
        # else:
        #     btym = max(bt[3], bt[1]) + delta
        # return image, masks[id].mask, scores[id], btxm, btym

    def get_mask_contours(self, mask):
        """
        Get contours of the mask.

        Args:
            mask (): mask of the image.

        Returns:
            contours_mask (): List of arrays, each array has the pixels that make up the contour of the mask, as x,
            y pairs.
        """
        # Pad mask with zeros to ensure proper polygons for masks that touch image edges.
        contours_mask = []
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask

        # Find boundary of the mask
        contours = find_contours(padded_mask, 0.5)
        # Subtract the padding and flip (y, x) to (x, y)
        for contour in contours:
            vertices = np.fliplr(contour) - 1
            contours_mask.append(np.array(vertices, np.int32))
        return contours_mask

    def nms(self, scores, masks, threshold=0.5):
        """
        Apply non-maximum suppression (NMS) to a set of masks and scores.
        :param masks: A (N, H, W) array of segmentation masks.
        :param scores: An (N,) array of scores.
        :param threshold: The IOU threshold to use for NMS.
        :return: Arrays of masks and scores with overlapping masks removed.
        """
        # if len(masks) == 0:
        #     return [], []
        # Sort masks and scores by score
        indices = np.argsort(-scores)
        masks = masks[indices]
        scores = scores[indices]
        # Array to keep track of whether an instance is suppressed or not
        suppressed = np.zeros((len(masks)), dtype=np.bool)
        # For each mask, compute overlap with other masks and suppress overlapping
        # masks if their score is lower
        for i in range(len(masks) - 1):
            if suppressed[i]:
                continue
            overlap = np.sum(masks[i] * masks[i + 1:], axis=(1, 2)) / np.sum(masks[i] + masks[i + 1:], axis=(1, 2))
            # suppressed[i + 1:] = np.logical_and(suppressed[i + 1:], overlap > threshold)
            suppressed[i + 1:] = overlap > threshold
        # Keep masks and scores that are not suppressed
        masks = masks[~suppressed]
        scores = scores[~suppressed]
        return masks, scores

    # def get_mask_contours(self, mask):
    #     # mask = masks[:, :, i]
    #     # Mask Polygon
    #     # Pad to ensure proper polygons for masks that touch image edges.
    #     contours_mask = []
    #     padded_mask = np.zeros(
    #         (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    #     padded_mask[1:-1, 1:-1] = mask
    #     contours = find_contours(padded_mask, 0.5)
    #     for verts in contours:
    #         # Subtract the padding and flip (y, x) to (x, y)
    #         verts = np.fliplr(verts) - 1
    #         contours_mask.append(np.array(verts, np.int32))
    #     return contours_mask
    # def draw_mask(self, img, pts, color, alpha=0.5):
    #     h, w, _ = img.shape
    #
    #     overlay = img.copy()
    #     output = img.copy()
    #
    #     cv2.fillPoly(overlay, pts, color)
    #     output = cv2.addWeighted(overlay, alpha, output, 1 - alpha,
    #                              0, output)
    #     return output
    def draw_mask(self, img, contours, color, alpha=0.5):
        """
        Draw the mask on the image.
        Args:
            img (): Original image
            contour_pts (): Points that make up the contour of the mask.
            color (): Color to draw the mask with.
            alpha (): Transparency of the mask.

        Returns:
            masked_img (): Image with the mask drawn on it.

        """
        # Draw boundary
        cv2.drawContours(img, contours, -1, color, 2)

        # Get image dimensions
        h, w, _ = img.shape

        # Make copies of the image
        overlay = img.copy()
        masked_img = img.copy()

        # Fill polygon
        cv2.fillPoly(overlay, contours, color)
        masked_img = cv2.addWeighted(overlay, alpha, masked_img, 1 - alpha, 0, masked_img)
        return masked_img

    def show_all_segs(self, image, threshold=0.1):
        """
        Show all the segments of the image with probability greater than threshold
        Args:
            predictor_output (): Output from the predictor
            threshold (float, optional): Score threshold to show segments. Defaults to 0.1.

        Returns:
            None (shows the image)
        """
        # Get the predictor output
        predictor_output = self.predictor(image)

        output = predictor_output["instances"].to("cpu")
        # Loop over the instances and save the ones with probability greater than threshold
        segs_to_show = []
        for i in range(len(output)):
            if output.scores[i] > threshold:
                segs_to_show.append(i)

        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        v = Visualizer(image, scale=1.0, )

        out = v.draw_instance_predictions(output[segs_to_show])
        # Show masks on image
        cv2.imshow('image', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # estimator.plot_results(num)
    segmenation_model = TrunkSegmenter()

    img = cv2.imread(os.environ.get('DATASET_PATH') + 'row96/20.png')
    pc = cv2.imread(os.environ.get('DATASET_PATH') + 'row96/20.npy')

    # width, masked_image = segmenation_model.show_all_segs(img, threshold=0.4)

