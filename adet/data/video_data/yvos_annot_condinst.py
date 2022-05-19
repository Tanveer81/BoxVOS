import json
import time
from glob import glob
from pathlib import Path
from adet.data.video_data.util import *
from PIL import Image, ImageFont, ImageDraw
import os
import random


categories = ['airplane', 'ape', 'bear', 'bike', 'bird', 'boat', 'bucket', 'bus', 'camel', 'cat',
              'cow', 'crocodile', 'deer', 'dog', 'dolphin', 'duck', 'eagle', 'earless_seal',
              'elephant', 'fish', 'fox', 'frisbee', 'frog', 'giant_panda', 'giraffe', 'hand',
              'hat', 'hedgehog', 'horse', 'knife', 'leopard', 'lion', 'lizard', 'monkey',
              'motorbike', 'mouse', 'others', 'owl', 'paddle', 'parachute', 'parrot', 'penguin',
              'person', 'plant', 'rabbit', 'raccoon', 'sedan', 'shark', 'sheep', 'sign',
              'skateboard', 'snail', 'snake', 'snowboard', 'squirrel', 'surfboard', 'tennis_racket',
              'tiger', 'toilet', 'train', 'truck', 'turtle', 'umbrella', 'whale', 'zebra']


def generate_detectron2_annotations(annot_path, detectron2_annos_path, meta_path, split):
    f = open(meta_path, )
    data = json.load(f)
    f.close()
    video_id_names = list(data.keys())
    video_id_names.sort()
    detectron2_annos = {}
    detectron2_annos['annos'] = []
    detectron2_annos['bbox_mode'] = 'BoxMode.XYXY_ABS'
    errors = {}
    start_time = time.time()
    cont = False
    for i, video_id in enumerate(video_id_names):
        annotations_paths = np.sort(glob(os.path.join(annot_path, video_id, '*.png'))).tolist()
        for j in range(0, len(annotations_paths)):
            try:
                file_name = annotations_paths[j]  # path
                mask_image = Image.open(file_name)

                # Create the annotations
                sub_masks = create_sub_masks(mask_image)
                annotations = []
                for object_id, sub_mask in sub_masks.items():
                    segmentation = create_sub_mask_annotation(sub_mask)
                    segmentation = [s for s in segmentation if len(s) >= 6]
                    if len(segmentation)==0 and split=='val':
                        cont=True
                        break

                    bbox, area = submask_to_box(sub_mask, True)  # xyxy format

                    category_id = categories.index(data[video_id]['objects'][object_id]['category'])
                    iscrowd = 0  # TODO: calculate this
                    annotation = {
                        'iscrowd': iscrowd,
                        'bbox': bbox,
                        'category_id': category_id,
                        'segmentation': segmentation,
                        'object_id': object_id
                    }
                    annotations.append(annotation)
                    if cont:
                        cont = False
                        continue

                anno = {
                    'video_id': video_id,
                    'frame_id': file_name.split('/')[-1].split('.')[0],
                    'height': mask_image.height,
                    'width': mask_image.width,
                    'image_id': i + j,
                    'annotations': annotations
                }
                detectron2_annos['annos'].append(anno)
            except Exception as e:
                frame = file_name.split('/')[-1].split('.')[0]
                try:
                    errors[video_id].append(frame)
                except KeyError:
                    errors[video_id] = [frame]
                print(f'video_id: {video_id}, frame_id: {frame}')
                print(f"An exception occurred: {e}")

        print(f'{i + 1}/{len(video_id_names)}: {video_id} : {(time.time() - start_time)} seconds')
    print(f'Total Time : {(time.time() - start_time)} seconds')

    with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-balanced-2.json', 'w') as outfile:
        json.dump(detectron2_annos, outfile)


def main():
    root = '../data'
    # img_path = f'{root}/youtubeVOS/train/JPEGImages/'
    annot_path = f'{root}/youtubeVOS/train/Annotations/'
    detectron2_annos_path = f'{root}/youtubeVOS/train/'
    for split in ['valid', 'train', 'test']:
        meta_path = f'{root}/youtubeVOS/train/train-{split}-meta-balanced.json'
        generate_detectron2_annotations(annot_path, detectron2_annos_path, meta_path, split)


if __name__ == '__main__':
    main()
