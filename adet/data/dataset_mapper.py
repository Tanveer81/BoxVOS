import copy
import logging
import os
import os.path as osp
import cv2
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
# import lycon
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .augmentation import RandomCropWithInstance
from .detection_utils import (annotations_to_instances, build_augmentation,
                              transform_instance_annotations)

from adet.data.video_data.util import visualize, visualize_bbox, visualize_polygon, visualize_segmentation

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        root = cfg['root']
        self.morphological_closing = cfg['morphological_closing']
        if self.morphological_closing:
            self.morphological_kernel = np.ones((5, 5), np.uint8)
        self.percentile_filter = float(cfg['percentile_filter']) if cfg['percentile_filter'] else cfg['percentile_filter']
        self.binary_motion = float(cfg['binary_motion']) if cfg['binary_motion'] else cfg['binary_motion']
        self.similarity_map_net = int(cfg['similarity_map_net']) if cfg['similarity_map_net'] else None
        self.is_train = is_train
        self.augment = cfg['augment']
        self.motion = cfg['motion']
        self.motion_resnet = int(cfg['motion_resnet']) if cfg['motion_resnet'] else -2
        motion_type = cfg['motion_type']
        self.motion_path = f'{root}/youtubeVOS/train/{motion_type}/'
        self.img_dir = f'{root}/youtubeVOS/train/JPEGImages/'
        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            # start_time = time.time()
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            # print("---PIL %s seconds ---" % (time.time() - start_time))
            '''Lycon loads in RGB format, we need BGR format so the flipping [:, :, ::-1] is added'''
            # start_time = time.time()
            # image = lycon.load(dataset_dict["file_name"])[:, :, ::-1]
            # print("---Lycon %s seconds ---" % (time.time() - start_time))
            motion_enabled = (self.motion != 'None' and self.is_train) or self.motion_resnet > -2 or self.similarity_map_net is not None
            if motion_enabled:
                if "motion_file_name" in dataset_dict:
                    motion = np.load(dataset_dict["motion_file_name"])[dataset_dict["frame_id"]]
                else:  # for valid dataset
                    video_id = dataset_dict["file_name"].split('/')[-2]
                    frame_list = os.listdir(os.path.join(self.img_dir, video_id))
                    frame_id = dataset_dict["file_name"].split('/')[-1].split('.')[0]
                    motion_path = os.path.join(self.motion_path, video_id, 'motions.npy')
                    motion = np.load(motion_path)[frame_list.index(frame_id + '.jpg')]

                if self.percentile_filter:
                    motion = (motion >= np.percentile(motion, self.percentile_filter)).astype(np.uint8) * motion
                elif self.binary_motion:
                    motion = (motion >= np.percentile(motion, 95)).astype(np.uint8)
                elif self.morphological_closing:
                    motion = (motion >= np.percentile(motion, 95)).astype('uint8')
                    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, self.morphological_kernel)

        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        # Same spatial augmentation applied to motion
        if motion_enabled:
            for transform in transforms.transforms:
                if any(substring in str(transform) for substring in ['Resize', 'Flip', 'Rotation', 'Crop']):
                    motion = transform.apply_image(motion)

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        if 'dense' in self.motion_path and motion_enabled: # take only dense motion magnitude
            motion = motion.transpose(2, 0, 1)

        if motion_enabled:
            dataset_dict["motion"] = torch.as_tensor(
                np.ascontiguousarray(motion)
            )

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                        .replace("train2017", "thing_train2017")
                        .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                        .replace("coco", "lvis")
                        .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict
