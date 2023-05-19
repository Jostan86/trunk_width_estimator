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

    def make_prediction(self, image):
        outputs = self.predictor(image)
        return outputs

    def get_mask(self, image, score_thresh):
        """
        Get the best mask from the predictions, using tieqiao's method
        Args:
            image (): Image that was sent to the model.
            score_thresh (): Min score threshold for predictions.

        Returns:
            image (): Image with the mask drawn on it.
            mask (): Best mask from the predictions.
            score (): Score of the best mask.
        """
        outputs = self.make_prediction(image)

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
            masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]

        index_use = None

        # If no scores are above threshold, return empty results
        if len(masks) == 0:
            return image,np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),-1,-1,-1

        id = 0
        h, w, _ = image.shape
        mid_w = w / 2.0
        m_distance = 1000000
        for i in range(len(masks)):
            a = np.max(masks[i].mask, axis=0)
            a1 = np.nonzero(a)
            a2 = np.mean(a1)
            b = np.max(masks[i].mask, axis=1)
            t_distance = abs(mid_w - a2)
            if sum(b) < image.shape[0] * 0.35: continue
            if t_distance < m_distance:
                m_distance = t_distance
                id = i

        colors = [255, 255, 255]
        contours = self.get_mask_contours(masks[id].mask)
        cnt2 = 0
        for cnt in contours:
            cv2.polylines(image, [cnt], True, colors, 2)
            image = self.draw_mask(image, [cnt], colors)

        bt = boxes.tensor[id].numpy().astype('int')
        start_point = (bt[0], bt[1])
        end_point = (bt[2], bt[3])
        delta = 20
        btxm = bt[0]
        if min(bt[3], bt[1]) > image.shape[0] - max(bt[3], bt[1]):
            btym = int((bt[3] + bt[1]) / 2)
            btxm = bt[2]
        else:
            btym = max(bt[3], bt[1]) + delta
        return image, masks[id].mask, scores[id], btxm, btym

    def get_mask_top_score(self, image, score_thresh):
        """
        Get the best mask with the highest score from the predictions
        Args:
            score_thresh (): Min score threshold for predictions.
            image (): Image that was sent to the model.

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

        empty_img = np.zeros_like(image[:, :, ::-1])
        v = Visualizer(empty_img, scale=1.0, )

        # Convert masks to numpy arrays and keep only those above score threshold
        if predictions.has("pred_masks"):
            masks = predictions.pred_masks.numpy()[keep]

        # If no scores are above threshold, return empty results
        if len(masks) == 0:
            return image,np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),-1,-1,-1

        # Apply NMS to remove overlapping predictions with lower score
        scores = scores.numpy()

        # Get the index of the mask with the highest score
        index_use = np.argmax(scores)

        colors = [255, 255, 255]
        # Get the contours of the mask
        mask = masks[index_use]
        contours = self.get_mask_contours(mask)
        # Draw the contours on the image
        image = self.draw_mask(image, contours, colors)

        return image, mask, scores[index_use]

    def get_mask_contours(self, mask):
        # mask = masks[:, :, i]
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        contours_mask = []
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            contours_mask.append(np.array(verts, np.int32))
        return contours_mask

    def draw_mask(self, img, pts, color, alpha=0.5):
        h, w, _ = img.shape

        overlay = img.copy()
        output = img.copy()

        cv2.fillPoly(overlay, pts, color)
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                                 0, output)
        return output

    def show_all_segs(self, image, threshold=0.1):
        """
        Show all the segments of the image with probability greater than threshold
        Args:
            image (np.array): Image to show segments on.
            threshold (float, optional): Score threshold to show segments. Defaults to 0.1.

        Returns:
            None (shows the image)
        """
        # Get the predictor output
        predictor_output = self.make_prediction(image)

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

    segmenation_model = TrunkSegmenter()

    img = cv2.imread(os.environ.get('DATASET_PATH') + 'row96/24.png')
    pc = cv2.imread(os.environ.get('DATASET_PATH') + 'row96/24.npy')

    width, masked_image = segmenation_model.show_all_segs(img, threshold=0.1)
