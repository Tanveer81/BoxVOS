import json
import os
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from adet.data.video_data.davis_annot import categories
from detectron2.data.datasets import register_coco_instances
#TODO: save dataset in coco dictionary with this function later
from detectron2.data.datasets.coco import convert_to_coco_dict


def get_davis_dicts(img_dir, detectron2_annos_path, motion_path, split, avoid_first_frame=False):
    # with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-balanced-condinst.json') as json_file:
    with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-balanced.json') as json_file:
        data = json.load(json_file)

    dataset_dicts = []
    for idx, v in enumerate(data['annos']):
        if not v['annotations'] and 'test' not in split: # check if empty annotations
            continue
        # avoid first frames while training because they have bad motion maps
        if avoid_first_frame and v['frame_id'] == '00000' and split=='train':
            continue
        record = {}
        record["file_name"] = os.path.join(img_dir, v['video_id'], v['frame_id'] + '.jpg')
        record["motion_file_name"] = os.path.join(motion_path, v['video_id'], 'motions.npy') if motion_path else motion_path
        record["image_id"] = idx
        record["height"] = v['height']
        record["width"] = v['width']

        frame_list = os.listdir(os.path.join(img_dir, v['video_id']))
        frame_list.sort()
        record['frame_id'] = frame_list.index(v['frame_id'] + '.jpg')

        objs = []
        if 'test' not in split:
            for anno in v['annotations']:
                obj = {
                    "bbox": [anno['bbox'][0], anno['bbox'][1], anno['bbox'][2], anno['bbox'][3]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": anno['segmentation'],
                    "category_id": anno['category_id'],
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_davis(motion_type='motion', test_set=None, avoid_first_frame=False, coco=False):
    root = '../data/DAVIS_2019_unsupervised_480'
    davis_val_coco_format = f'{root}/trainval/davis_val_coco_format.json'
    img_path = f'{root}/trainval/JPEGImages/480p/'
    annot_path = f'{root}/Annotations_unsupervised/480p/'
    detectron2_annos_path = f'{root}/trainval/'
    motion_path = f'{root}/trainval/{motion_type}'

    for d in ["train"]:#, "val"]:
        DatasetCatalog.register("davis_" + d,
                                lambda d=d: get_davis_dicts(img_path, detectron2_annos_path, motion_path, d, avoid_first_frame))
        MetadataCatalog.get("davis_" + d).set(thing_classes=categories)

    # MetadataCatalog.get("davis_val").set(evaluator_type='coco')
    if coco:
        coco_val = '../data/coco/annotations/instances_val2017.json'
        register_coco_instances(f"davis_val", {}, coco_val, "~")
        MetadataCatalog.get(f"davis_val").set(thing_classes=categories)
    elif test_set:
        register_coco_instances(f"davis_{test_set}", {}, f'{root}/{test_set}/davis_coco_format.json', "~")
        MetadataCatalog.get(f"davis_{test_set}").set(thing_classes=categories)
    else:
        register_coco_instances("davis_val", {}, davis_val_coco_format, "~")
        MetadataCatalog.get("davis_" + "val").set(thing_classes=categories)


def create_coco(avoid_first_frame=False):
    root = '../data/'
    for d in ["test-dev", "test-challenge"]:
        img_path = f'{root}/DAVIS-2017-{d}-480p/DAVIS/JPEGImages/480p/'
        detectron2_annos_path = f'{root}/DAVIS-2017-{d}-480p/DAVIS'
        DatasetCatalog.register("davis_" + d,
                                lambda d=d: get_davis_dicts(img_path, detectron2_annos_path, None, d,
                                                            avoid_first_frame))
        MetadataCatalog.get("davis_" + d).set(thing_classes=categories)

        davis_train_coco = convert_to_coco_dict(f'davis_{d}')
        davis_coco_format = f'{root}/DAVIS-2017-{d}-480p/DAVIS/davis_coco_format.json'
        with open(davis_coco_format, 'w') as fp:
            json.dump(davis_train_coco, fp)


def main():
    # register_yvos()
    # create_coco_dict()
    split = 'train'  # 'val'
    root = '../data/DAVIS_2019_unsupervised_480/trainval'
    img_path = f'{root}/JPEGImages/480p/'
    annot_path = f'{root}/Annotations_unsupervised/480p/'
    detectron2_annos_path = root
    motion_path = f'{root}/motion_200_2_2_2'
    davis_val_coco_format = f'{root}/davis_val_coco_format.json'
    davis_train_coco_format = f'{root}/davis_train_coco_format.json'
    davis_test_dev_coco_format = f'{root}/davis_test_dev_coco_format.json'
    davis_test_challange_coco_format = f'{root}/davis_test_challange_coco_format.json'

    # a = get_davis_dicts(img_path, detectron2_annos_path, motion_path, 'val')
    # b = get_davis_dicts(img_path, detectron2_annos_path, motion_path, 'train')
    register_davis(motion_type='motion', test_set=None, avoid_first_frame=False)

    # davis_val_coco = convert_to_coco_dict('davis_val')
    # with open(davis_val_coco_format, 'w') as fp:
    #     json.dump(davis_val_coco, fp)

    davis_train_coco = convert_to_coco_dict('davis_train')
    with open(davis_train_coco_format, 'w') as fp:
        json.dump(davis_train_coco, fp)


if __name__ == '__main__':
    # main()
    create_coco()