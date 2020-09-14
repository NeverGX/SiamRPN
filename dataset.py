import torch
import glob
import cv2
import os
import numpy as np
from torch.utils.data.dataset import Dataset
from config import config
from utils import crop_and_pad, compute_iou, box_transform
np.random.seed(123456)

class GOT_10KDataset(Dataset):
    def __init__(self, data_dir, z_transforms, x_transforms, anchors):

        self.data_dir = data_dir
        self.videos = os.listdir(data_dir)
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        self.num = config.pairs_per_video * len(self.videos)
        self.anchors = anchors


    def __getitem__(self, index):
        index = index % len(self.videos)
        video = self.videos[index]
        video_path = os.path.join(self.data_dir, video)
        n_frames = len(os.listdir(video_path))
        z_id = np.random.choice(n_frames)
        z_path = glob.glob(os.path.join(video_path, "{:0>8d}.x**.jpg".format(z_id+1)))[0]
        z = cv2.imread(z_path, cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        low_limit = max(0, z_id - config.frame_range)
        up_limit = min(n_frames, z_id + config.frame_range)
        x_id = np.random.choice(range(low_limit, up_limit))
        x_path = glob.glob(os.path.join(video_path, "{:0>8d}.x**.jpg".format(x_id+1)))[0]

        w_z = float(z_path.split('=')[1])
        h_z = float(z_path.split('=')[2])
        w_x = float(x_path.split('=')[1])
        h_x = float(x_path.split('=')[2])

        x = cv2.imread(x_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if np.random.rand(1) < config.gray_ratio: # data augmentation for gray image to color image
            z = cv2.cvtColor(z, cv2.COLOR_RGB2GRAY)
            z = cv2.cvtColor(z, cv2.COLOR_GRAY2RGB)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        z = crop_and_pad(z, (z.shape[1]-1)/2, (z.shape[0]-1)/2, config.exemplar_size, config.exemplar_size)
        x, gt_w, gt_h = self.RandomStretch(x, w_x, h_x)
        cy, cx, gt_cx, gt_cy = self.RandomTranslate(x)
        x = crop_and_pad(x, cx, cy, config.instance_size, config.instance_size)
        z = self.z_transforms(z)
        x = self.x_transforms(x)
        regression_label, classification_label = self.compute_target(self.anchors,
                                                             np.array(list(map(round, [gt_cx*config.total_stride, gt_cy*config.total_stride, gt_w, gt_h]))))
        return z, x, regression_label, classification_label

    def __len__(self):
        return self.num

    def RandomStretch(self, sample, w_x, h_x, max_stretch=config.max_stretch):
        scale_h = 1.0 + np.random.uniform(-max_stretch, max_stretch)
        scale_w = 1.0 + np.random.uniform(-max_stretch, max_stretch)
        h, w = sample.shape[:2]
        shape = (int(h * scale_h), int(w * scale_w))
        gt_w = w_x * scale_w
        gt_h = h_x * scale_h
        return cv2.resize(sample, shape, cv2.INTER_LINEAR), gt_w, gt_h

    def RandomTranslate(self, sample):
        h, w = sample.shape[:2]
        cy_o = (h - 1) // 2
        cx_o = (w - 1) // 2
        cy = cy_o + np.random.randint(- config.max_translate, config.max_translate + 1)
        cx = cx_o + np.random.randint(- config.max_translate, config.max_translate + 1)
        gt_cx = cx_o - cx
        gt_cy = cy_o - cy
        return cy, cx, gt_cx, gt_cy

    def compute_target(self, anchors, box):
        regression_label = box_transform(anchors, box)
        iou = compute_iou(anchors, box).flatten()
        pos_index = np.where(iou > config.pos_threshold)[0]
        neg_index = np.where(iou < config.neg_threshold)[0]
        classification_label = np.ones_like(iou, dtype = np.float32) * -1
        classification_label[pos_index] = 1
        classification_label[neg_index] = 0
        return regression_label, classification_label



