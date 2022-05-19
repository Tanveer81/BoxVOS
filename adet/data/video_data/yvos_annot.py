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
    for i, video_id in enumerate(video_id_names):
        # video_id = '65b7dda462'
        # print(f'{i+1}/{len(sequences_names)}: {seq}')
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

    if split == 'train':
        with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-balanced.json', 'w') as outfile:
            json.dump(detectron2_annos, outfile)
    else:
        with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-full-balanced.json', 'w') as outfile:
            json.dump(detectron2_annos, outfile)

    with open(f'{detectron2_annos_path}/detectron2-annotations-errors-{split}-full-balanced.json', 'w') as outfile:
        json.dump(errors, outfile)


def refine_detectron2_annotations(detectron2_annos_path):
    for split in ['valid', 'test']:
        with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-full-balanced.json') as json_file:
            data = json.load(json_file)

        for anno in data['annos']:
            cont = False
            for annot in anno['annotations']:
                for s in annot['segmentation']:
                    if len(s) <= 6:
                        cont = True
                        continue
                if cont:
                    continue
            if cont:
                data['annos'].remove(anno)
                continue

        with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-balanced.json', 'w') as outfile:
            json.dump(data, outfile)


def refine_detectron2_annotations_for_condinst(detectron2_annos_path):
    # as condinst uses polygons while training, remove the polygons that has less than or equal to 6 edges.
    with open(f'{detectron2_annos_path}/detectron2-annotations-train-balanced.json') as json_file:
        data = json.load(json_file)

    for anno in data['annos']:
        cont = False
        for annot in anno['annotations']:
            for s in annot['segmentation']:
                if len(s) % 2 == 0 and len(s) <= 6:
                    cont = True
                    continue
            if cont:
                continue
        if cont:
            data['annos'].remove(anno)
            continue

    with open(f'{detectron2_annos_path}/detectron2-annotations-train-balanced-condinst.json', 'w') as outfile:
        json.dump(data, outfile)


def debug_detectron2_annotations(detectron2_annos_path, img_path):
    with open(f'{detectron2_annos_path}/detectron2-annotations-valid-balanced.json') as json_file:
        data = json.load(json_file)
    # 49 sign, 6 bucket
    seen_videos = []
    for cat in [6, 26, 36, 49, 58]:
        print(categories[cat])
        for annos in data['annos']:
            if annos['video_id'] in seen_videos:
                continue
            for anno in annos['annotations']:
                if anno['category_id'] == cat:
                    image = Image.open(img_path + annos['video_id'] + '/' + annos['frame_id'] + '.jpg')
                    # print(categories[anno['category_id'] ])
                    visualize_bbox(image, anno['bbox'])
                    seen_videos.append(annos['video_id'])


def debug_detectron2_annotations2(detectron2_annos_path, img_path):
    with open(f'{detectron2_annos_path}/detectron2-annotations-valid-balanced.json') as json_file:
        data = json.load(json_file)

    for idx, annos in enumerate(data['annos']):
        if annos['video_id'] == 'a2e6608bfa':
            print(idx)
            break

    print()


def generate_barlow_twin_annotations(img_path, meta_path, out_path, frame_dist):
    f = open(meta_path, )
    data = json.load(f)
    f.close()
    video_id_names = list(data.keys())
    video_id_names.sort()
    frame_pairs = []
    start_time = time.time()
    for i, video_id in enumerate(video_id_names[:3]):
        img_paths = np.sort(glob(os.path.join(img_path, video_id, '*.jpg'))).tolist()
        if len(img_paths) <= frame_dist:
            continue
        categories = list(data[video_id]['objects'].values())
        frames = [a.split('/')[-1].split('.')[0] for a in img_paths]
        frame_categories = dict((a, []) for a in frames)

        for j, category in enumerate(categories):
            for frame in category['frames']:
                frame_categories[frame].append(j)

        number_of_pairs = int(len(img_paths)/frame_dist) * 100
        attempts = 3 * number_of_pairs
        while True:
            start = random.randint(0, len(img_paths) - frame_dist-1)
            end = random.randint(start + frame_dist, len(img_paths)-1)
            key = f'{video_id}/{frames[start]}.jpg'
            if key not in frame_pairs and set(frame_categories[frames[start]]) == set(frame_categories[frames[end]]):
                frame_pairs.append((key, f'{video_id}/{frames[end]}.jpg'))
                number_of_pairs -= 1
                if number_of_pairs == 0:
                    break
            attempts -= 1
            if attempts == 0:
                break

        print(f'{i + 1}/{len(video_id_names)}: {video_id} : {(time.time() - start_time)} seconds')

    print(f'Total Time : {(time.time() - start_time)} seconds')
    print(f'Saving pairs as {out_path}barlow_twins_pairs.txt')
    with open(f'{out_path}barlow_twins_pairs_test.txt', 'w') as fp:
        fp.write('\n'.join('%s %s' % x for x in frame_pairs))


def debug_barlow_twin_annotations(out_path):
    with open(f'{out_path}barlow_twins_pairs.json') as json_file:
        data = json.load(json_file)
        print()


def actual_train():
    root = '../data'
    img_path = f'{root}/youtubeVOS/train/JPEGImages/'
    annot_path = f'{root}/youtubeVOS/train/Annotations/'
    detectron2_annos_path = f'{root}/youtubeVOS/train/'
    data = []
    for split in ['valid', 'train', 'test']:
        with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-balanced-2.json') as json_file:
            data.append(json.load(json_file))
    data[0]['annos'].extend(data[1]['annos'])
    data[0]['annos'].extend(data[2]['annos'])
    data = data[0]
    with open(f'{detectron2_annos_path}/detectron2-annotations-actual-train-balanced.json', 'w') as outfile:
        json.dump(data, outfile)


def main():
    root = '../data'
    img_path = f'{root}/youtubeVOS/train/JPEGImages/'
    annot_path = f'{root}/youtubeVOS/train/Annotations/'
    detectron2_annos_path = f'{root}/youtubeVOS/train/'
    # for split in ['valid', 'train', 'test']:
    #     meta_path = f'{root}/youtubeVOS/train/train-{split}-meta-balanced.json'
    #     # Total Time : 2758.5989196300507 seconds
    #     generate_detectron2_annotations(annot_path, detectron2_annos_path, meta_path, split)
    # refine_detectron2_annotations(detectron2_annos_path)
    # refine_detectron2_annotations_for_condinst(detectron2_annos_path)
    debug_detectron2_annotations2(detectron2_annos_path, img_path)

    # Barlow_twins
    # meta_path = f'{root}/youtubeVOS/train/train-train-meta-balanced.json'
    # frame_dist = 5
    # generate_barlow_twin_annotations(img_path, meta_path, detectron2_annos_path, frame_dist)
    # debug_barlow_twin_annotations(detectron2_annos_path)


if __name__ == '__main__':
    actual_train()
