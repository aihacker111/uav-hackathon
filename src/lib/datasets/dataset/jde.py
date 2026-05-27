import glob
import math
import os
import os.path as osp
import random
import copy
import time
import warnings

import cv2
import numpy as np
import PIL.Image
import torch

from collections import OrderedDict, defaultdict
from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.datasets.uav_augment import build_aerial_mot_transforms
from lib.utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
from lib.tracker.multitracker import id2cls

# ImageNet mean/std (CxHxW, float32) — must match the Normalize() used in train.py.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# for inference
class LoadImages:
    def __init__(self, path, img_size=(1088, 608)):
        """
        :param path:
        :param img_size:
        """
        self.frame_rate = 10  # no actual meaning here

        if type(path) == str:
            if os.path.isdir(path):
                image_format = ['.jpg', '.jpeg', '.png', '.tif']
                self.files = sorted(glob.glob('%s/*.*' % path))
                self.files = list(filter(lambda x: os.path.splitext(x)[
                                                       1].lower() in image_format, self.files))
            elif os.path.isfile(path):
                self.files = [path]
        elif type(path) == list:
            self.files = path

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        if self.count == self.nF:
            raise StopIteration

        img_path = self.files[self.count]

        # Read image
        img_0 = cv2.imread(img_path)  # BGR
        assert img_0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # Normalize RGB — must match train.py: /255 then ImageNet mean/std
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        return img_path, img, img_0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img_0 = cv2.imread(img_path)  # BGR
        assert img_0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # Normalize RGB: BGR -> RGB and H×W×C -> C×H×W, then ImageNet mean/std
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        return img_path, img, img_0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self,
                 path,
                 img_size=(1088, 608)):
        """
        :param path:
        :param img_size:
        """
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration

        # Read image
        res, img_0 = self.cap.read()  # BGR
        assert img_0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img_0 = cv2.resize(img_0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # Normalize RGB — must match train.py: /255 then ImageNet mean/std
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB and HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        return self.count, img, img_0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self,
                 path,
                 img_size=(1088, 608),
                 augment=False,
                 transforms=None):
        """
        :param path:
        :param img_size:
        :param augment:
        :param transforms:
        """
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids')
                            .replace('.png', '.txt')
                            .replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files

        self.width = img_size[0]
        self.height = img_size[1]

        self.augment = augment
        self.transforms = transforms

        # PIL augmentation pipeline (pre-letterbox).  None during inference.
        # Replaces apply_weather_light_color + bias_crop + random_affine.
        self._pil_aug = build_aerial_mot_transforms() if augment else None

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path, width=None, height=None):
        """
        Image data format conversion + augmentation; label formatting.

        Augmentation path (self.augment=True):
          1. Load raw BGR image.
          2. Load raw labels (norm. xywh) and convert to pixel xyxy of original image.
          3. Apply PIL augmentation pipeline (geometric + appearance, pre-letterbox).
             Pipeline includes: RandomHorizontalFlip, ScaleBiasedCrop,
             RandomPerspective, ColorJitter, NightMode/Fog/Glare (OneOf),
             MotionBlur/GaussianBlur (OneOf), JPEG compression, sensor noise,
             occlusion patches, multi-scale resize.
          4. Letterbox the augmented image to final network resolution.
          5. Re-map augmented boxes to letterbox pixel coords.
          6. Convert xyxy → normalised cxcywh.

        Inference path (self.augment=False):
          Standard letterbox + label coord conversion; no augmentation.

        :param img_path:   path to input image
        :param label_path: path to label .txt (normalised xywh, 6 columns)
        :param height:     target network height  (default: self.height)
        :param width:      target network width   (default: self.width)
        :return:           img (C×H×W tensor), labels (N×6), img_path, (h_orig, w_orig)
        """
        if height is None or width is None:
            height = self.height
            width = self.width

        # Read raw image (BGR numpy)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))

        h_orig, w_orig = img.shape[:2]  # original dims — returned to caller

        # ── Augmentation path ─────────────────────────────────────────────────
        if self.augment and self._pil_aug is not None:

            # Load raw labels (normalised xywh, cols: cls_id track_id cx cy bw bh)
            if os.path.isfile(label_path):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels_0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            else:
                labels_0 = np.zeros((0, 6), dtype=np.float32)

            # Convert normalised xywh → pixel xyxy in original image space
            if len(labels_0) > 0:
                boxes = np.stack([
                    (labels_0[:, 2] - labels_0[:, 4] / 2) * w_orig,  # x1
                    (labels_0[:, 3] - labels_0[:, 5] / 2) * h_orig,  # y1
                    (labels_0[:, 2] + labels_0[:, 4] / 2) * w_orig,  # x2
                    (labels_0[:, 3] + labels_0[:, 5] / 2) * h_orig,  # y2
                ], axis=1).astype(np.float32)
                target_dict = {
                    "boxes":  torch.from_numpy(boxes),
                    "labels": torch.from_numpy(labels_0[:, 0].astype(np.int64)),
                    "ids":    torch.from_numpy(labels_0[:, 1].astype(np.int64)),
                    "size":   torch.tensor([h_orig, w_orig]),
                }
            else:
                target_dict = {
                    "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros(0, dtype=torch.int64),
                    "ids":    torch.zeros(0, dtype=torch.int64),
                    "size":   torch.tensor([h_orig, w_orig]),
                }

            # Apply PIL augmentation pipeline (BGR → RGB PIL → augment → BGR numpy)
            pil_img = PIL.Image.fromarray(img[:, :, ::-1])
            pil_img, target_dict = self._pil_aug(pil_img, target_dict)
            img = np.array(pil_img)[:, :, ::-1].copy()  # RGB PIL → BGR numpy

            # Letterbox the augmented image to final network resolution
            img, ratio, pad_w, pad_h = letterbox(img, height=height, width=width)

            # Re-map augmented boxes to letterbox pixel coordinates
            aug_boxes = target_dict["boxes"].numpy()
            if len(aug_boxes) > 0:
                labels = np.zeros((len(aug_boxes), 6), dtype=np.float32)
                labels[:, 0] = target_dict["labels"].numpy()
                labels[:, 1] = target_dict["ids"].numpy()
                labels[:, 2] = np.clip(aug_boxes[:, 0] * ratio + pad_w, 0, width)   # x1
                labels[:, 3] = np.clip(aug_boxes[:, 1] * ratio + pad_h, 0, height)  # y1
                labels[:, 4] = np.clip(aug_boxes[:, 2] * ratio + pad_w, 0, width)   # x2
                labels[:, 5] = np.clip(aug_boxes[:, 3] * ratio + pad_h, 0, height)  # y2
                # Drop degenerate boxes produced by aggressive crops/perspective
                keep = (labels[:, 4] - labels[:, 2] > 2) & (labels[:, 5] - labels[:, 3] > 2)
                labels = labels[keep]
            else:
                labels = np.array([])

        # ── Inference / no-augmentation path ──────────────────────────────────
        else:
            img, ratio, pad_w, pad_h = letterbox(img, height=height, width=width)

            if os.path.isfile(label_path):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels_0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

                labels = labels_0.copy()
                labels[:, 2] = ratio * w_orig * (labels_0[:, 2] - labels_0[:, 4] / 2) + pad_w  # x1
                labels[:, 3] = ratio * h_orig * (labels_0[:, 3] - labels_0[:, 5] / 2) + pad_h  # y1
                labels[:, 4] = ratio * w_orig * (labels_0[:, 2] + labels_0[:, 4] / 2) + pad_w  # x2
                labels[:, 5] = ratio * h_orig * (labels_0[:, 3] + labels_0[:, 5] / 2) + pad_h  # y2
            else:
                labels = np.array([])

        # ── Debug visualisation (disabled by default) ─────────────────────────
        plot_flag = False
        if plot_flag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            if len(labels) > 0:
                plt.plot(labels[:, [2, 4, 4, 2, 2]].T,
                         labels[:, [3, 3, 5, 5, 3]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        # ── Convert xyxy → normalised cxcywh ──────────────────────────────────
        num_labels = len(labels)
        if num_labels > 0:
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height

        # ── BGR → RGB, apply tensor transforms ────────────────────────────────
        # Note: RandomHorizontalFlip is already part of the PIL pipeline above;
        # the standalone lr_flip block has been removed to avoid double-flipping.
        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h_orig, w_orig)

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img,
              height=608,
              width=1088,
              color=(127.5, 127.5, 127.5)):
    """
    resize a rectangular image to a padded rectangular
    :param img:
    :param height:
    :param width:
    :param color:
    :return:
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])

    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width - new_shape[0]) * 0.5  # width padding
    dh = (height - new_shape[1]) * 0.5  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resized, no border
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def random_affine(img, targets=None,
                  degrees=(-10, 10),
                  translate=(.1, .1),
                  scale=(.9, 1.1),
                  shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(
        img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * \
              img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * \
              img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * \
                    (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)),
                            abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

        return imw, targets, M
    else:
        return imw


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)


# ----------

class JointDataset(LoadImagesAndLabels):  # for training
    """
    joint detection and embedding dataset
    """
    mean = None
    std = None

    def __init__(self,
                 opt,
                 root,
                 paths,
                 img_size=(1088, 608),
                 augment=False,
                 transforms=None):
        """
        :param opt:
        :param root:
        :param paths:
        :param img_size:
        :param augment:
        :param transforms:
        """
        self.opt = opt
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = len(opt.reid_cls_ids.split(','))  # C5: car, bicycle, person, cyclist, tricycle

        # make sure img_size equal to opt.input_wh
        if opt.input_wh[0] != img_size[0] or opt.input_wh[1] != img_size[1]:
            opt.input_wh[0], opt.input_wh[1] = img_size[0], img_size[1]

        # default input width and height
        self.default_input_wh = opt.input_wh

        # net input width and height
        self.width = self.default_input_wh[0]
        self.height = self.default_input_wh[1]

        # PIL augmentation pipeline — instantiated once, reused per sample.
        # Replaces apply_weather_light_color + bias_crop + random_affine.
        self.augment = augment
        self._pil_aug = build_aerial_mot_transforms() if augment else None

        # ----- generate img and label file path lists
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [x.replace('images', 'labels_with_ids')
                                    .replace('.png', '.txt')
                                    .replace('.jpg', '.txt')
                                    for x in self.img_files[ds]]

            print('Total {} image files in {} dataset.'.format(len(self.label_files[ds]), ds))

        if opt.id_weight > 0:  # If do ReID calculation
            # @even: for MCMOT training
            for ds, label_paths in self.label_files.items():
                max_ids_dict = defaultdict(int)  # cls_id => max track id

                for lp in label_paths:
                    if not os.path.isfile(lp):
                        print('[Warning]: invalid label file {}.'.format(lp))
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        lb = np.loadtxt(lp)
                        if len(lb) < 1:
                            continue

                        lb = lb.reshape(-1, 6)
                        for item in lb:
                            if item[1] > max_ids_dict[int(item[0])]:
                                max_ids_dict[int(item[0])] = item[1]

                # track id number
                self.tid_num[ds] = max_ids_dict

            # @even: for MCMOT training
            self.tid_start_idx_of_cls_ids = defaultdict(dict)
            last_idx_dict = defaultdict(int)
            for k, v in self.tid_num.items():
                for cls_id, id_num in v.items():
                    self.tid_start_idx_of_cls_ids[k][cls_id] = last_idx_dict[cls_id]
                    last_idx_dict[cls_id] += id_num

            # @even: for MCMOT training
            self.nID_dict = defaultdict(int)
            for k, v in last_idx_dict.items():
                self.nID_dict[k] = int(v)

        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.max_objs = opt.K
        self.transforms = transforms

        print('dataset summary')
        print(self.tid_num)

        if opt.id_weight > 0:
            for k, v in self.nID_dict.items():
                print('Total {:d} IDs of {}'.format(v, id2cls[k]))

            for k, v in self.tid_start_idx_of_cls_ids.items():
                for cls_id, start_idx in v.items():
                    print('Start index of dataset {} class {:d} is {:d}'
                          .format(k, int(cls_id), int(start_idx)))

    def __getitem__(self, idx):
        for i, c in enumerate(self.cds):
            if idx >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][idx - start_index]
        label_path = self.label_files[ds][idx - start_index]

        # Get image data and label
        imgs, labels, img_path, (input_h, input_w) = self.get_data(img_path, label_path)

        # @even: for MCMOT training — remap track ids per sub-dataset
        if self.opt.id_weight > 0:
            for i, _ in enumerate(labels):
                if labels[i, 1] > -1:
                    cls_id = int(labels[i][0])
                    start_idx = self.tid_start_idx_of_cls_ids[ds][cls_id]
                    labels[i, 1] += start_idx

        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio

        num_objs = labels.shape[0]

        # --- GT of detection
        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs,), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs,), dtype=np.uint8)

        if self.opt.id_weight > 0:
            # --- GT of ReID
            ids = np.zeros((self.max_objs,), dtype=np.int64)

            # @even: each class has its own track-id map
            cls_tr_ids = np.zeros((self.num_classes, output_h, output_w), dtype=np.int64)

            # @even, class id map
            cls_id_map = np.full((1, output_h, output_w), -1, dtype=np.int64)

        # Gauss function definition
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        for k in range(num_objs):
            label = labels[k]

            bbox = label[2:]  # center_x, center_y, bbox_w, bbox_h

            cls_id = int(label[0])

            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)

            w, h = bbox[2], bbox[3]

            if h > 0 and w > 0:
                # heat-map radius
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius

                # bbox center coordinate
                ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # draw gauss weight for heat-map
                draw_gaussian(hm[cls_id], ct_int, radius)

                # --- GT of detection
                wh[k] = float(w), float(h)

                ind[k] = ct_int[1] * output_w + ct_int[0]

                reg[k] = ct - ct_int
                reg_mask[k] = 1

                # --- GT of ReID
                if self.opt.id_weight > 0:
                    cls_id_map[0][ct_int[1], ct_int[0]] = cls_id

                    cls_tr_ids[cls_id][ct_int[1]][ct_int[0]] = label[1] - 1

                    ids[k] = label[1] - 1

        if self.opt.id_weight > 0:
            ret = {'input': imgs,
                   'hm': hm,
                   'reg': reg,
                   'wh': wh,
                   'ind': ind,
                   'reg_mask': reg_mask,
                   'ids': ids,
                   'cls_id_map': cls_id_map,
                   'cls_tr_ids': cls_tr_ids}
        else:  # only for detection
            ret = {'input': imgs,
                   'hm': hm,
                   'reg': reg,
                   'wh': wh,
                   'ind': ind,
                   'reg_mask': reg_mask}

        return ret