"""SOS dataset access + the official CBD-Net training augmentation.

Augmentation is copied from third_party/CBDNet-main/data.py (official code) with one
compatibility fix: `np.math` (removed in NumPy 2) -> `math`. Nothing else is changed.

SOS layout on disk:
  train/<sensor>/<id>_sat.jpg, <id>_mask.png       (images and masks in one folder)
  test/<sensor>/sat/<id>_sat.jpg, gt/<id>_mask.png  (separate folders)
Masks: white (255) = oil, black (0) = background.
"""
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SOS_ROOT = os.environ.get("SOS_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "SOS_dataset"))


def list_split(split, sensor, root=SOS_ROOT):
    """Return list of (id, sat_path, mask_path) for split in {'train','test'}."""
    if split == "train":
        d = os.path.join(root, "train", sensor)
        ids = sorted(f[:-8] for f in os.listdir(d) if f.endswith("_sat.jpg"))
        return [(i, os.path.join(d, f"{i}_sat.jpg"), os.path.join(d, f"{i}_mask.png")) for i in ids]
    d = os.path.join(root, "test", sensor)
    ids = sorted(f[:-8] for f in os.listdir(os.path.join(d, "sat")) if f.endswith("_sat.jpg"))
    return [(i, os.path.join(d, "sat", f"{i}_sat.jpg"), os.path.join(d, "gt", f"{i}_mask.png")) for i in ids]


def list_from_split_csv(path, partitions):
    """Items from a group-disjoint split CSV (C1). partitions e.g. ['train','val','calib'] or ['test']."""
    import csv
    import re
    keep = set(partitions)

    def local(p):  # the CSV stores absolute paths of the original machine; re-root them under SOS_ROOT if needed
        if os.path.exists(p):
            return p
        rel = re.split(r"[\\/]SOS_dataset[\\/]", p, maxsplit=1)[-1]
        return os.path.join(SOS_ROOT, *re.split(r"[\\/]", rel))
    return [(r["tile_id"], local(r["sat"]), local(r["mask"])) for r in csv.DictReader(open(path)) if r["partition"] in keep]


# ---------------- official augmentation (data.py), np.math -> math ----------------
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift % 256)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def randomShiftScaleRotate(image, mask, shift_limit=(-0.0, 0.0), scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape
        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
        cc = math.cos(angle / 180 * math.pi) * sx
        ss = math.sin(angle / 180 * math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
        mat = cv2.getPerspectiveTransform(box0.astype(np.float32), box1.astype(np.float32))
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=borderMode, borderValue=(0, 0, 0))
    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image, mask = np.rot90(image), np.rot90(mask)
    return image, mask


def normalise(img_bgr_uint8):
    """Official CBD-Net input scaling: [0,255] -> [-1.6, 1.6], CHW float32."""
    return np.array(img_bgr_uint8, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6


class SOSTrain(Dataset):
    def __init__(self, items, augment=True, aug="cbdnet"):
        """aug='cbdnet': official CBD-Net pipeline; aug='flips': flips + 90-degree rotations only."""
        self.items, self.augment, self.aug = items, augment, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        _, sp, mp = self.items[k]
        img = cv2.imread(sp)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if self.augment and self.aug == "flips":
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
        elif self.augment:
            img = randomHueSaturationValue(img, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))
            img, mask = randomShiftScaleRotate(img, mask, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1),
                                               aspect_limit=(-0.1, 0.1), rotate_limit=(-0, 0))
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
        mask = (np.array(mask, np.float32)[None] / 255.0 >= 0.5).astype(np.float32)
        return torch.from_numpy(normalise(np.ascontiguousarray(img))), torch.from_numpy(mask)


def load_eval(sat_path, mask_path):
    img = cv2.imread(sat_path)
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
    return img, gt
