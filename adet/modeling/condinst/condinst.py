# -*- coding: utf-8 -*-
import logging

import cv2
import numpy as np
from PIL import Image
from skimage import color

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask

from .dynamic_mask_head import build_dynamic_mask_head
from .mask_branch import build_mask_branch

from adet.utils.comm import aligned_bilinear
from adet.data.video_data.util import visualize, visualize_bbox, visualize_polygon, visualize_segmentation

from detectron2.layers import ShapeSpec

__all__ = ["CondInst"]


logger = logging.getLogger(__name__)


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation, motion=None, dense_dot_product=''):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(images, kernel_size=kernel_size, dilation=dilation)

    if images.shape[1] == 2 and dense_dot_product=='True': #handle dense optical flow
        diff = -torch.einsum("ijklm,ijplm->ijplm", (images[:, :, None], unfolded_images))
    else:
        diff = images[:, :, None] - unfolded_images

    if motion=='NormChannel':
        norm = torch.norm(torch.cat((torch.norm(diff[:, :-1, :, :], dim=1), diff[:, -1, :, :] * (374 / 255))), dim=0)[None, ...]
    else:
        norm = torch.norm(diff, dim=1)

    similarity = torch.exp(-norm * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights


@META_ARCH_REGISTRY.register()
class CondInst(nn.Module):
    """
    Main class for CondInst architectures (see https://arxiv.org/abs/2003.05664).
    """
    def __init__(self, cfg):
        super().__init__()
        self.dense_dot_product = cfg['dense_dot_product']
        self.motion_type = cfg['motion_type']
        if cfg['pairwise_motion_thresh']:
            self.register_buffer("pairwise_motion_thresh", torch.tensor(float(cfg['pairwise_motion_thresh'])))
        self.similarity_map_net = int(cfg['similarity_map_net']) if cfg['similarity_map_net'] else None
        self.barlow_twin_pretrain = cfg['barlow_twin_pretrain']
        self.motion_resnet = int(cfg['motion_resnet']) if cfg['motion_resnet'] else -2
        self.motion_pairwise_size = int(cfg['motion_pairwise_size']) \
            if cfg['motion_pairwise_size'] else cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.motion_pairwise_dilation = int(cfg['motion_pairwise_dilation']) \
            if cfg['motion_pairwise_dilation'] else cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.train_from_scratch = cfg['train_from_scratch']
        self.resume = cfg['resume']
        self.visualize = cfg['visualize']
        self.motion = cfg['motion']
        if cfg['color_weight']:
            self.color_weight = float(cfg['color_weight'])
        self.device = torch.device(cfg.MODEL.DEVICE)
        if self.motion == 'NormChannel':
            self.channel_mins = torch.tensor([0, -128, -128, 0]).to(self.device)[None, :, None, None]
            self.channel_range = torch.tensor([100, 255, 255, 255]).to(self.device)[None, :, None, None]
        if self.motion == 'NormChannel2':
            self.channel_weights = torch.tensor([self.color_weight]*3 + [1-self.color_weight]).to(self.device)[None, :, None, None]

        # if self.motion_resnet > -1 and (self.resume or self.train_from_scratch): #TODO: implement train from scratch
        #     self.backbone = build_backbone(cfg, ShapeSpec(channels=4))
        # else:
        self.backbone = build_backbone(cfg)

        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())

        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE

        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS
        self.topk_proposals_per_im = cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module

        self.controller = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        # When using pre-trained models in Detectron1 or any MSRA models,
        # std has been absorbed into its conv1 weights, so the std needs to be set 1.
        # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
        if self.barlow_twin_pretrain:
            PIXEL_STD = [57.375, 57.120, 58.395]
        else:
            PIXEL_STD = cfg.MODEL.PIXEL_STD
        if self.motion_resnet > -2:
            pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN + [62.167]).to(self.device).view(4, 1, 1)
            pixel_std = torch.Tensor(PIXEL_STD + [40.2]).to(self.device).view(4, 1, 1)
        else:
            pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
            pixel_std = torch.Tensor(PIXEL_STD).to(self.device).view(3, 1, 1)

        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        '''
        visualize(original_images[0].permute(1,2,0).cpu())
        visualize(motions[0].cpu())
        '''
        original_images = [x["image"].to(self.device) for x in batched_inputs]
        if (self.motion!='None' and self.training) or self.motion_resnet > -2 or self.similarity_map_net is not None:
            motions = [x["motion"].to(self.device) for x in batched_inputs]
        else:
            motions = None

        # visualize(original_images[0].permute(1, 2, 0).cpu())
        # visualize(motions[0].cpu())
        # visualize(((motions[0] > 63.75).to(torch.uint8)*motions[0]).cpu())
        # visualize(((motions[0] > 127.5).to(torch.uint8)*motions[0]).cpu())
        # visualize(((motions[0] > 242.25).to(torch.uint8)*motions[0]).cpu())

        # If similarity map is not put into resnet
        if self.similarity_map_net is None:
            if self.motion_resnet > -2:
                temp_image = []
                for i in range(len(original_images)):
                    temp_image.append(torch.cat((original_images[i], motions[i][None])))
                original_images = temp_image
            # normalize images
            images_norm = [self.normalizer(x) for x in original_images]
            images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)

            features = self.backbone(images_norm.tensor)

            if self.motion_resnet > -2:
                for i in range(len(original_images)):
                    original_images[i] = original_images[i][:-1, ...]
                images_norm.tensor = images_norm.tensor[:, :-1, ...]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if self.boxinst_enabled:
                original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images]

                # mask out the bottom area where the COCO dataset probably has wrong annotations
                for i in range(len(original_image_masks)): #TODO: remove this feature for video_data
                    im_h = batched_inputs[i]["height"]
                    pixels_removed = int(
                        self.bottom_pixels_removed *
                        float(original_images[i].size(1)) / float(im_h)
                    )
                    if pixels_removed > 0:
                        original_image_masks[i][-pixels_removed:, :] = 0

                original_images = ImageList.from_tensors(original_images, self.backbone.size_divisibility)
                if (self.motion!='None' and self.training) or self.motion_resnet > -2:
                    motions = ImageList.from_tensors(motions, self.backbone.size_divisibility)
                original_image_masks = ImageList.from_tensors(
                    original_image_masks, self.backbone.size_divisibility, pad_value=0.0
                )
                self.add_bitmasks_from_boxes(
                    gt_instances, original_images.tensor, original_image_masks.tensor,
                    original_images.tensor.size(-2), original_images.tensor.size(-1), motions.tensor if motions else motions
                )

            else:
                self.add_bitmasks(gt_instances, images_norm.tensor.size(-2), images_norm.tensor.size(-1))
        else:
            gt_instances = None

        # If similarity map is put into resnet
        if self.similarity_map_net is not None:
            if gt_instances:
                image_color_similarity = torch.cat([x[0].image_color_similarity[:,:,0,...] for x in gt_instances])
            else:
                temp_image = []
                for i in range(len(original_images)):
                    temp_image.append(torch.cat((original_images[i], motions[i][None])))
                original_images = temp_image
                original_images = ImageList.from_tensors(original_images, self.backbone.size_divisibility)
                images = original_images.tensor
                stride = self.mask_out_stride
                start = int(stride // 2)
                downsampled_images = F.avg_pool2d(
                    images.float(), kernel_size=stride,
                    stride=stride, padding=0
                )[:, [2, 1, 0, 3]]
                motions = downsampled_images[:, -1, :, :]
                downsampled_images = downsampled_images[:, :-1, :, :]

                image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images]
                image_masks = ImageList.from_tensors(image_masks, self.backbone.size_divisibility, pad_value=0.0)
                image_masks = image_masks.tensor[:, start::stride, start::stride]

                for im_i in range(len(downsampled_images)):
                    images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy())
                    images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
                    images_lab = images_lab.permute(2, 0, 1)[None]

                    images_color_similarity = get_images_color_similarity(
                        images_lab, image_masks[im_i],
                        self.pairwise_size, self.pairwise_dilation, self.motion
                    )

                    images_motion_similarity = get_images_color_similarity(
                        motions[im_i][None, None, ...], image_masks[im_i],
                        self.motion_pairwise_size, self.motion_pairwise_dilation
                    )
                    image_color_similarity = torch.cat((images_color_similarity[:,0], images_motion_similarity[:,0]))[None]

            motion_weights = (image_color_similarity[:, 1, ...] >= self.pairwise_motion_thresh).float()
            weights = (image_color_similarity[:, 0, ...] >= self.pairwise_color_thresh).float()

            weights = weights.to(torch.int8) & motion_weights.to(torch.int8)
            weights = weights.to(torch.float32)
            intersection_weights = (weights.to(torch.int8) & motion_weights.to(torch.int8)).to(torch.float32)

            if not gt_instances:
                temp_images = []
                for i in range(len(original_images)):
                    temp_images.append(original_images.tensor[i][:-1, ...])
                original_images = temp_images

            # normalize images
            images_norm = [self.normalizer(x) for x in original_images]
            images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)

            features = self.backbone((images_norm.tensor, intersection_weights))

        mask_feats, sem_losses = self.mask_branch(features, gt_instances)

        proposals, proposal_losses = self.proposal_generator(
            images_norm, features, gt_instances, self.controller
        )

        if self.training:
            mask_losses = self._forward_mask_heads_train(proposals, mask_feats, gt_instances, original_images, motions)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update(mask_losses)
            return losses
        else:
            pred_instances_w_masks = self._forward_mask_heads_test(proposals, mask_feats)

            padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images_norm.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                instances_per_im = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w
                )

                processed_results.append({
                    "instances": instances_per_im
                })

            return processed_results

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances, original_images=None, motions=None):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]

        assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."
        if self.max_proposals != -1:
            if self.max_proposals < len(pred_instances):
                inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
                logger.info("clipping proposals from {} to {}".format(
                    len(pred_instances), self.max_proposals
                ))
                pred_instances = pred_instances[inds[:self.max_proposals]]
        elif self.topk_proposals_per_im != -1:
            num_images = len(gt_instances)

            kept_instances = []
            for im_id in range(num_images):
                instances_per_im = pred_instances[pred_instances.im_inds == im_id]
                if len(instances_per_im) == 0:
                    kept_instances.append(instances_per_im)
                    continue

                unique_gt_inds = instances_per_im.gt_inds.unique()
                num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)

                for gt_ind in unique_gt_inds:
                    instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]

                    if len(instances_per_gt) > num_instances_per_gt:
                        scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
                        ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                        inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
                        instances_per_gt = instances_per_gt[inds]

                    kept_instances.append(instances_per_gt)

            pred_instances = Instances.cat(kept_instances)

        pred_instances.mask_head_params = pred_instances.top_feats

        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances, original_images, motions
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, mask_feats):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat

        pred_instances_w_masks = self.mask_head(
            mask_feats, self.mask_branch.out_stride, pred_instances
        )

        return pred_instances_w_masks

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else: # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full

    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w, motions):
        stride = self.mask_out_stride
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        if motions is not None:
            dense_flow = len(motions.shape) == len(images.shape)
            if dense_flow: #handle dense opt flow case
                images = torch.cat((images, motions), 1)
            else:
                images = torch.cat((images, motions[:, None, :, :]), 1)

        if motions is not None:
            if dense_flow:
                downsampled_images = F.avg_pool2d(
                    images.float(), kernel_size=stride,
                    stride=stride, padding=0
                )[:, [2, 1, 0, 3, 4]]
                motions = downsampled_images[:, [3, 4], :, :]
            else:
                downsampled_images = F.avg_pool2d(
                    images.float(), kernel_size=stride,
                    stride=stride, padding=0
                )[:, [2, 1, 0, 3]]
                motions = downsampled_images[:, 3, :, :]
            downsampled_images = downsampled_images[:, [0, 1, 2], :, :]
        else:
            downsampled_images = F.avg_pool2d(
                images.float(), kernel_size=stride,
                stride=stride, padding=0
            )[:, [2, 1, 0]]

        image_masks = image_masks[:, start::stride, start::stride]

        for im_i, per_im_gt_inst in enumerate(instances):
            if 'MotionOnly' in self.motion:
                images_color_similarity = get_images_color_similarity(
                    motions[im_i][None, None, ...], image_masks[im_i],
                    self.pairwise_size, self.pairwise_dilation
                )
            else:
                images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy())
                images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
                images_lab = images_lab.permute(2, 0, 1)[None]
                if self.motion!='None' and 'Decoupled' not in self.motion: #channel
                    images_lab = torch.cat((images_lab, motions[im_i][None, None, ...]), 1)
                # if self.motion == 'NormChannel':
                    # images_lab -= self.channel_mins
                    # images_lab[:,-1,:,:] *= (374/255)#self.channel_range
                if self.motion == 'NormChannel2':
                    images_lab *= self.channel_weights
                images_color_similarity = get_images_color_similarity(
                    images_lab, image_masks[im_i],
                    self.pairwise_size, self.pairwise_dilation, self.motion
                )
                if 'Decoupled' in self.motion: #'Decoupled', 'DecoupledUnion', 'DecoupledIntersection'
                    images_motion_similarity = get_images_color_similarity(
                        motions[im_i][None, ...] if dense_flow else motions[im_i][None, None, ...], image_masks[im_i],
                        self.motion_pairwise_size, self.motion_pairwise_dilation, None, self.dense_dot_product
                    )
                    images_color_similarity = torch.cat((images_color_similarity, images_motion_similarity), 0).unsqueeze(0)

            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            per_im_bitmasks = []
            per_im_bitmasks_full = []
            for per_box in per_im_boxes:
                bitmask_full = torch.zeros((im_h, im_w)).to(self.device).float()
                bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0

                bitmask = bitmask_full[start::stride, start::stride]

                assert bitmask.size(0) * stride == im_h
                assert bitmask.size(1) * stride == im_w

                per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)

            per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            per_im_gt_inst.image_color_similarity = torch.cat([
                images_color_similarity for _ in range(len(per_im_gt_inst))
            ], dim=0)

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results
