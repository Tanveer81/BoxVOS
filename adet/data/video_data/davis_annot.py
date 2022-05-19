import json
import time
from glob import glob
from adet.data.video_data.util import *
from PIL import Image
import os

categories = ['object']


def generate_detectron2_annotations(annot_path, detectron2_annos_path, meta_path, split):
    with open(meta_path) as temp_file:
        video_id_names = [line.rstrip('\n') for line in temp_file]
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
                    if len(segmentation)==0:
                        cont=True
                        break

                    bbox, area = submask_to_box(sub_mask, True)  # xyxy format

                    category_id = 0
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
                    cont=False
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
    with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-balanced.json', 'w') as outfile:
        json.dump(detectron2_annos, outfile)


def generate_detectron2_annotations_test(img_path, detectron2_annos_path, meta_path, split):
    with open(meta_path) as temp_file:
        video_id_names = [line.rstrip('\n') for line in temp_file]
    video_id_names.sort()
    detectron2_annos = {}
    detectron2_annos['annos'] = []
    detectron2_annos['bbox_mode'] = 'BoxMode.XYXY_ABS'
    errors = {}
    start_time = time.time()
    cont = False
    for i, video_id in enumerate(video_id_names):
        img_paths = np.sort(glob(os.path.join(img_path, video_id, '*.jpg'))).tolist()
        for j in range(0, len(img_paths)):
            try:
                file_name = img_paths[j]  # path
                image = Image.open(file_name)
                if cont:
                    cont=False
                    continue
                anno = {
                    'video_id': video_id,
                    'frame_id': file_name.split('/')[-1].split('.')[0],
                    'height': image.height,
                    'width': image.width,
                    'image_id': i + j,
                    'annotations': None
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
    with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-balanced.json', 'w') as outfile:
        json.dump(detectron2_annos, outfile)


def test():
    root = '../data/'
    for split in ['test-dev', 'test-challenge']:
        img_path = f'{root}/DAVIS-2017-{split}-480p/DAVIS/JPEGImages/480p/'
        meta_path = f'{root}/DAVIS-2017-{split}-480p/DAVIS/ImageSets/2017/{split}.txt'
        detectron2_annos_path = f'{root}/DAVIS-2017-{split}-480p/DAVIS'
        generate_detectron2_annotations_test(img_path, detectron2_annos_path, meta_path, split)


def train():
    root = '../data/DAVIS_2019_unsupervised_480/trainval'
    img_path = f'{root}/JPEGImages/480p/'
    annot_path = f'{root}/Annotations_unsupervised/480p/'
    detectron2_annos_path = root
    for split in ['val', 'train']:
        meta_path = f'{root}/ImageSets/2017/{split}.txt'
        generate_detectron2_annotations(annot_path, detectron2_annos_path, meta_path, split)


if __name__ == '__main__':
    test()


'''

'''