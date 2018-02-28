# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import cv2
import time
import numpy as np
from .test import im_detect, nms
from .config import cfg, cfg_from_file as set_config


def detect(net, imdb, max_per_image=20, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t = time.time()
        scores, boxes = im_detect(net, im)
        boxes = boxes.reshape(-1, 21, 4)
        _im_detect = time.time() - _t
        _t = time.time()

        total_boxes = 0
        for j in xrange(1, imdb.num_classes):  # skip j = 0 (background class)
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j, :]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets  # B X 5
            total_boxes += cls_dets.shape[0]

        if max_per_image > 0 and total_boxes > max_per_image:
            if len(image_scores) > max_per_image:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_classes)])
                print(len(image_scores))
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _misc = time.time() - _t

        if i % 100 == 0:
            print("%d/%d (%d x %d) %.5fs %.5fs" % (i + 1, num_images, im.shape[0], im.shape[1], _im_detect, _misc))

    return all_boxes
