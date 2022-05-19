import logging

import numpy as np
import torch

from detectron2.data import transforms as T
from detectron2.data.detection_utils import \
    annotations_to_instances as d2_anno_to_inst
from detectron2.data.detection_utils import \
    transform_instance_annotations as d2_transform_inst_anno
import random

def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):

    annotation = d2_transform_inst_anno(
        annotation,
        transforms,
        image_size,
        keypoint_hflip_indices=keypoint_hflip_indices,
    )

    if "beziers" in annotation:
        beziers = transform_beziers_annotations(annotation["beziers"], transforms)
        annotation["beziers"] = beziers
    return annotation


def transform_beziers_annotations(beziers, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        beziers (list[float]): Nx16 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    beziers = np.asarray(beziers, dtype="float64").reshape(-1, 2)
    beziers = transforms.apply_coords(beziers).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError("Flipping text data is not supported (also disencouraged).")

    return beziers


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    instance = d2_anno_to_inst(annos, image_size, mask_format)

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)

    return instance


def build_augmentation(cfg, is_train):
    """
    With option to don't use hflip

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)

    augmentation = []
    # if is_train:
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        if cfg.INPUT.HFLIP_TRAIN:
            augmentation.append(T.RandomFlip())
        if cfg['augment']:
            T.RandomApply(T.RandomCrop(crop_type=cfg.INPUT.CROP.TYPE, crop_size=cfg.INPUT.CROP.SIZE), 0.5)
            # augmentation.append(random.choice([T.RandomBrightness(0.5, 2),
            #                                    T.RandomContrast(0.5, 2),
            #                                    T.RandomSaturation(0.5, 2),
            #                                    T.RandomCrop(crop_type=cfg.INPUT.CROP.TYPE, crop_size=cfg.INPUT.CROP.SIZE),
            #                                    T.RandomRotation(angle=[90, 180, 270], sample_style="choice")]))
            # augmentation.extend([T.RandomBrightness(0.2, 2),
            #                     T.RandomContrast(0.2, 2),
            #                     T.RandomSaturation(0.2, 2),
            #                     # T.RandomApply(T.RandomCrop(crop_type=cfg.INPUT.CROP.TYPE, crop_size=cfg.INPUT.CROP.SIZE), 0.5)
            #                     # T.RandomApply(T.RandomRotation(angle=[90, 180, 270], sample_style="choice"), 0.2)
            #                     ])

        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation

build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""
