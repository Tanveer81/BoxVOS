# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from glob import glob
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy
import tqdm
from adet.data.video_data.util import visualize
from detectron2.utils.visualizer import ColorMode
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg
# constants
WINDOW_NAME = "COCO detections"
from adet.data.video_data.util import *

def args_opts_value(args, key, delete=True):
    try:
        value = args.opts[args.opts.index(key) + 1]
    except ValueError:
        return None
    if delete:
        del args.opts[args.opts.index(key)+1]
        del args.opts[args.opts.index(key)]
    return value


def check_YVOS(args, cfg):
    if args_opts_value(args, 'YVOS') == 'True':
        cfg.merge_from_file(args_opts_value(args, 'video-config-file'))
        '''motion, motion_75, motion_150, motion_200, motion_200_2_2_2, motion_2_dense'''
        cfg['motion_type'] = args_opts_value(args, 'motion_type')
        if cfg['motion_type'] is None:
            cfg['motion_type'] = 'motion'
        # register_yvos(cfg['motion_type'])
    else:
        args_opts_value(args, 'video-config-file')

    ''' Motions Options: 'MotionOnly', 'MotionOnly2','Channel', 'NormChannel', 
        'NormChannel2', 'Decoupled' 'DecoupledUnion' 'DecoupledIntersection' '''
    cfg['motion'] = args_opts_value(args, 'Motion')
    cfg['visualize'] = args_opts_value(args, 'Visualize')
    cfg['pairwise_motion_thresh'] = args_opts_value(args, 'Pairwise_Motion_Thresh')
    cfg['color_weight'] = args_opts_value(args, 'Color_Weight')
    cfg['augment'] = args_opts_value(args, 'Augment')
    cfg['train_from_scratch'] = args_opts_value(args, 'Train_From_Scratch')
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
    cfg['threshold_motion'] = args_opts_value(args, 'threshold_motion')
    cfg['binary_motion'] = args_opts_value(args, 'binary_motion')
    cfg['morphological_closing'] = args_opts_value(args, 'morphological_closing')
    cfg['resume'] = False


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    check_YVOS(args, cfg)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    numpy.random.seed(0)
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    cfg['motion_type'] = None
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION)

    meta_path = '../data/DAVIS_2019_unsupervised_480/trainval/ImageSets/2017/val.txt'
    with open(meta_path) as temp_file:
        video_id_names = [line.rstrip('\n') for line in temp_file]
    for i, video in enumerate(video_id_names):
        Path(f"{args.output}/{video}").mkdir(parents=True, exist_ok=True)
        imgs_paths = np.sort(glob(os.path.join(args.input[0], video, '*.jpg'))).tolist()
        for i in range(len(imgs_paths)):
            img = read_image(imgs_paths[i], format="BGR")
            predictions, visualized_output = demo.run_on_image(img)
            out_filename = f"{args.output}/{video}/{imgs_paths[i].split('/')[-1]}"
            visualized_output.save(out_filename)


