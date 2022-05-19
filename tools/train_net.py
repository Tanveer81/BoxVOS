# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import detectron2.utils.comm as comm
import torch
from util import build_detection_train_loader
from detectron2.data import MetadataCatalog, build_detection_test_loader #build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup, hooks, launch#, DefaultTrainer
from tools.defaults import DefaultTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.video_data.yvos_register import register_yvos
from adet.data.video_data.davis_register import register_davis
from adet.modeling.backbone.saliency_map import visualize_saliency
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

from detectron2.layers import Conv2d, get_norm
import random
import numpy as np


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def resume_or_load(self, resume=True, train_from_scratch=False):
        if not isinstance(self.checkpointer, AdetCheckpointer):
            # support loading a few other backbones
            self.checkpointer = AdetCheckpointer(
                self.model,
                self.cfg.OUTPUT_DIR,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        if not train_from_scratch or (train_from_scratch and resume):
            super().resume_or_load(resume=resume)

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`  with a customized
        DatasetMapper, which includes motion if used as input channel
        """
        mapper = DatasetMapperWithBasis(cfg, False)
        test_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return test_loader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def args_opts_value(args, key, delete=True):
    try:
        value = args.opts[args.opts.index(key) + 1]
    except ValueError:
        return None
    if delete:
        del args.opts[args.opts.index(key)+1]
        del args.opts[args.opts.index(key)]
    return value


def check_video_data(args, cfg):
    if args_opts_value(args, 'YVOS') == 'True':
        cfg.merge_from_file(args_opts_value(args, 'video-config-file'))
        '''motion, motion_75, motion_150, motion_200, motion_200_2_2_2, motion_2_dense'''
        cfg['motion_type'] = args_opts_value(args, 'motion_type')
        cfg['test_set'] = args_opts_value(args, 'test_set')
        if cfg['test_set']: # 'yvos_test', 'yvos_actual_test', 'yvos_actual_valid'
            cfg['DATASETS']['TEST'] = (cfg['test_set'],)
        if cfg['motion_type'] is None:
            cfg['motion_type'] = 'motion'
        avoid_first_frame = args_opts_value(args, 'avoid_first_frame')
        cfg['condinst'] = args_opts_value(args, 'condinst')
        if cfg['condinst']:
            cfg.MODEL.BOXINST.ENABLED = False
        cfg['no_class'] = args_opts_value(args, 'no_class')
        cfg['actual-train'] = args_opts_value(args, 'actual_train')
        register_yvos(cfg['motion_type'], cfg['test_set'], avoid_first_frame, cfg['condinst'], cfg['no_class'], cfg['actual-train'])
    elif args_opts_value(args, 'DAVIS') == 'True':
        cfg.merge_from_file(args_opts_value(args, 'video-config-file'))
        '''motion, motion_75, motion_150, motion_200, motion_200_2_2_2, motion_2_dense'''
        cfg['motion_type'] = args_opts_value(args, 'motion_type')
        cfg['test_set'] = args_opts_value(args, 'test-set')
        if cfg['test_set']:
            cfg['DATASETS']['TEST'] = ('davis_' + cfg['test_set'],)
        if cfg['motion_type'] is None:
            cfg['motion_type'] = 'motion'
        avoid_first_frame = args_opts_value(args, 'avoid_first_frame')
        cfg['condinst'] = args_opts_value(args, 'condinst')
        if cfg['condinst']:
            cfg.MODEL.BOXINST.ENABLED = False
        cfg['coco'] = args_opts_value(args, 'coco')
        register_davis(cfg['motion_type'], cfg['test_set'], avoid_first_frame, cfg['coco'])

    else:
        cfg['motion_type'] = None
        args_opts_value(args, 'video-config-file')

    ''' Motions Options: 'MotionOnly', 'MotionOnly2','Channel', 'NormChannel', 
        'NormChannel2', 'Decoupled' 'DecoupledUnion' 'DecoupledIntersection' '''
    cfg['motion'] = args_opts_value(args, 'Motion')
    cfg['visualize'] = args_opts_value(args, 'Visualize')
    cfg['pairwise_motion_thresh'] = args_opts_value(args, 'Pairwise_Motion_Thresh')
    cfg['color_weight'] = args_opts_value(args, 'Color_Weight')
    cfg['augment'] = args_opts_value(args, 'Augment')
    cfg['train_from_scratch'] = args_opts_value(args, 'Train_From_Scratch')
    cfg['resume'] = args.resume
    '''Motion Resnet options: Motion_Resnet_[Conv1/MaxPool] - 0,1
    , Motion_Resnet_res[2,3,4,5] - 2,3,4,5'''
    cfg['motion_resnet'] = args_opts_value(args, 'Motion_Resnet')
    cfg['motion_resnet_1_1'] = args_opts_value(args, 'Motion_Resnet_1_1')
    cfg['motion_pairwise_size'] = args_opts_value(args, 'Motion_Pairwise_Size')
    cfg['motion_pairwise_dilation'] = args_opts_value(args, 'Motion_Pairwise_Dilation')
    cfg['barlow_twin_ckp'] = args_opts_value(args, 'Barlow_Twin_Ckp')
    cfg['barlow_twin_pretrain'] = args_opts_value(args, 'barlow_twin_pretrain')
    cfg['find_unused_parameters'] = args_opts_value(args, 'find_unused_parameters')
    cfg['similarity_map_net'] = args_opts_value(args, 'similarity_map_net')
    if cfg['similarity_map_net']:
        cfg['motion_resnet'] = -3
    cfg['dense_dot_product'] = args_opts_value(args, 'dense_dot_product')
    cfg['no_shuffle'] = args_opts_value(args, 'no_shuffle')
    cfg['percentile_filter'] = args_opts_value(args, 'percentile_filter')
    cfg['binary_motion'] = args_opts_value(args, 'binary_motion')
    cfg['morphological_closing'] = args_opts_value(args, 'morphological_closing')
    cfg['saliency_map'] = args_opts_value(args, 'saliency_map')
    cfg['patience'] = args_opts_value(args, 'patience')
    cfg['confidence_threshold'] = args_opts_value(args, 'confidence_threshold')
    if cfg['confidence_threshold']:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(cfg['confidence_threshold'])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(cfg['confidence_threshold'])
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = float(cfg['confidence_threshold'])
        cfg.MODEL.MEInst.INFERENCE_TH_TEST = float(cfg['confidence_threshold'])
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = float(cfg['confidence_threshold'])
    cfg['root'] = args_opts_value(args, 'root')


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    check_video_data(args, cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)
    if cfg['saliency_map']:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        visualize_saliency(model.backbone)
        return

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume, train_from_scratch = cfg['train_from_scratch'])

    # Include 4th channel in the first conv layer for motion as 4th channel
    # In case of resume checkpoint or train from scratch it is done inside condinst
    inject_motion_at = int(cfg['motion_resnet']) if cfg['motion_resnet'] else -2
    if inject_motion_at == -1 and not (args.resume or cfg['train_from_scratch']):
        if args.num_gpus < 2:
            weight = trainer.model.backbone.bottom_up.stem.conv1.weight.clone()
            trainer.model.backbone.bottom_up.stem.conv1 = Conv2d(4, 64, kernel_size=7, stride=2,
                padding=3, bias=False, norm=get_norm(cfg.MODEL.RESNETS.NORM, 64)).to(weight.device)
            with torch.no_grad():
                trainer.model.backbone.bottom_up.stem.conv1.weight[:, :3] = weight
                trainer.model.backbone.bottom_up.stem.conv1.weight[:, 3] = torch.mean(weight, 1)
        else:
            weight = trainer.model.module.backbone.bottom_up.stem.conv1.weight
            del trainer.model.module.backbone.bottom_up.stem.conv1
            trainer.model.module.backbone.bottom_up.stem.conv1 = Conv2d(4, 64, kernel_size=7,
                stride=2, padding=3, bias=False, norm=get_norm(cfg.MODEL.RESNETS.NORM, 64)).to(weight.device)
            with torch.no_grad():
            # trainer.model.module.backbone.bottom_up.stem.conv1.weight[:, :3] = weight
            # trainer.model.module.backbone.bottom_up.stem.conv1.weight[:, 3] = torch.mean(weight, 1)
                trainer.model.module.backbone.bottom_up.stem.conv1.weight = torch.nn.Parameter(torch.cat((weight, torch.mean(weight, 1)[:,None,...]), dim=1))
            del weight
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    gpus = args_opts_value(args, 'cuda_visible_devices')
    block = args_opts_value(args, 'Block')
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
               dist_url=args.dist_url, args=(args,), )
