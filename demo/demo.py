# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
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
    numpy.random.seed(4)
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    cfg['motion_type'] = None
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            # visualize(img)
            start_time = time.time()

            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                visualize(visualized_output.get_image()[:, :, ::-1])
                # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                # if cv2.waitKey(0) == 27:
                #     break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
