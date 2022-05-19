import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from adet.data.video_data.yvos_annot import categories
from detectron2.data.datasets import register_coco_instances
#TODO: save dataset in coco dictionary with this function later
from detectron2.data.datasets.coco import convert_to_coco_dict

root = '../data'


def get_yvos_dicts(img_dir, detectron2_annos_path, motion_path, split, avoid_first_frame=False, condinst=None, no_class=None, actual_train=None):
    if actual_train and split=='train':
        with open(f'{detectron2_annos_path}/detectron2-annotations-actual-train-balanced.json') as json_file:
            data = json.load(json_file)
    elif condinst and split=='train':
        with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-balanced-2.json') as json_file:
            data = json.load(json_file)
    elif split == 'actual-test':
        with open(f'{detectron2_annos_path}/detectron2-annotations-actual-test-balanced.json') as json_file:
            data = json.load(json_file)
    elif split == 'actual-valid':
        with open(f'{detectron2_annos_path}/detectron2-annotations-actual-valid-balanced.json') as json_file:
            data = json.load(json_file)
    elif split == 'all-frame-actual-valid':
        with open(f'{detectron2_annos_path}/detectron2-annotations-all-frame-actual-valid.json') as json_file:
            data = json.load(json_file)
    else:
        with open(f'{detectron2_annos_path}/detectron2-annotations-{split}-balanced.json') as json_file:
            data = json.load(json_file)

    dataset_dicts = []
    for idx, v in enumerate(data['annos']):
        if not v['annotations'] and 'actual' not in split: # check if empty annotations
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
        if 'actual' not in split:
            for anno in v['annotations']:
                # for idx, s in enumerate(anno['segmentation']):
                #     if len(s) < 6:
                #         anno['segmentation'][idx] = [item for item in s for i in range(2)]
                obj = {
                    "bbox": [anno['bbox'][0], anno['bbox'][1], anno['bbox'][2], anno['bbox'][3]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": anno['segmentation'],
                    "category_id": 0 if no_class else anno['category_id'],
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_yvos(motion_type='motion', test_set=None, avoid_first_frame=False, condinst=True, no_class=None, actual_train=False):
    img_path = f'{root}/youtubeVOS/train/JPEGImages/'
    detectron2_annos_path = f'{root}/youtubeVOS/train/'
    motion_path = f'{root}/youtubeVOS/train/{motion_type}/'
    for d in ["train"]:#, "valid", "test"]:
        DatasetCatalog.register("yvos_" + d,
                                lambda d=d: get_yvos_dicts(img_path, detectron2_annos_path, motion_path, d, avoid_first_frame, condinst, no_class, actual_train))
        MetadataCatalog.get("yvos_" + d).set(thing_classes=['object'] if no_class else categories)
    # if test_set:
        # MetadataCatalog.get("yvos_test").set(evaluator_type='coco')
    # else:
        # MetadataCatalog.get("yvos_valid").set(evaluator_type='coco')

    if no_class:
        # yvos_train_coco_format = f'{root}/boxinst/output/yvos_train_coco_format_no_class.json'
        yvos_valid_coco_format = f'{root}/boxinst/output/yvos_valid_coco_format_no_class.json'
        yvos_test_coco_format = f'{root}/boxinst/output/yvos_test_coco_format_no_class.json'
    else:
        # yvos_train_coco_format = f'{root}/boxinst/output/yvos_train_coco_format.json'
        yvos_valid_coco_format = f'{root}/boxinst/output/yvos_valid_coco_format.json'
        yvos_test_coco_format = f'{root}/boxinst/output/yvos_test_coco_format.json'

    # register_coco_instances("yvos_train", {}, yvos_train_coco_format, "~")
    if test_set:
        if test_set=='yvos_test': # 'yvos_test', 'yvos_actual_test', 'yvos_actual_valid'
            register_coco_instances("yvos_test", {}, yvos_test_coco_format, "~")
            MetadataCatalog.get("yvos_" + "test").set(thing_classes=['object'] if no_class else categories)
        elif test_set=='yvos_actual_test': # 'yvos_test', 'yvos_actual_test', 'yvos_actual_valid'
            yvos_actual_test_coco_format = f'{root}/boxinst/output/yvos_actual-test_coco_format_no_class.json'
            register_coco_instances("yvos_actual_test", {}, yvos_actual_test_coco_format, "~")
            MetadataCatalog.get("yvos_actual_test").set(thing_classes=['object'])
        elif test_set=='yvos_actual_valid':
            yvos_actual_valid_coco_format = f'{root}/boxinst/output/yvos_actual-valid_coco_format_no_class.json'
            register_coco_instances("yvos_actual_valid", {}, yvos_actual_valid_coco_format, "~")
            MetadataCatalog.get("yvos_actual_valid").set(thing_classes=['object'])
        elif test_set=='yvos_all-frame-actual-valid':
            yvos_actual_valid_coco_format = f'{root}/boxinst/output/yvos_all-frame-actual-valid_coco_format_no_class.json'
            register_coco_instances("yvos_all-frame-actual-valid", {}, yvos_actual_valid_coco_format, "~")
            MetadataCatalog.get("yvos_all-frame-actual-valid").set(thing_classes=['object'])
    else:
        register_coco_instances("yvos_valid", {}, yvos_valid_coco_format, "~")
        MetadataCatalog.get("yvos_" + "valid").set(thing_classes=['object'] if no_class else categories)
        MetadataCatalog.get("yvos_" + "valid").set(thing_colors="red")


def create_coco(motion_type='motion', test_set=None, avoid_first_frame=False, condinst=True, no_class=None):
    img_path = f'{root}/youtubeVOS/train/JPEGImages/'
    detectron2_annos_path = f'{root}/youtubeVOS/train/'
    motion_path = f'{root}/youtubeVOS/train/{motion_type}/'

    for d in ["valid"]:
        DatasetCatalog.register("yvos_" + d,
                                lambda d=d: get_yvos_dicts(img_path, detectron2_annos_path, motion_path, d, avoid_first_frame, condinst, no_class))
        MetadataCatalog.get("yvos_" + d).set(thing_classes=['object'] if no_class else categories)

        train_coco = convert_to_coco_dict('yvos_'+d)

        if no_class:
            path = f'{root}/boxinst/output/yvos_{d}_coco_format_no_class.json'
        else:
            path = f'{root}/boxinst/output/yvos_{d}_coco_format.json'

        with open(path, 'w') as fp:
            json.dump(train_coco, fp)


def create_coco_actual_test(motion_type='motion', test_set=None, avoid_first_frame=False, condinst=True, no_class=None):
    img_path = f'{root}/youtubeVOS/valid/JPEGImages/'
    detectron2_annos_path = f'{root}/youtubeVOS/train/'

    for d in ["actual-valid"]: #"actual-test"
        DatasetCatalog.register("yvos_" + d,
                                lambda d=d: get_yvos_dicts(img_path, detectron2_annos_path, None, d, avoid_first_frame, condinst, no_class))
        MetadataCatalog.get("yvos_" + d).set(thing_classes=['object'] if no_class else categories)

        train_coco = convert_to_coco_dict('yvos_'+d)

        if no_class:
            path = f'{root}/boxinst/output/yvos_{d}_coco_format_no_class.json'
        else:
            path = f'{root}/boxinst/output/yvos_{d}_coco_format.json'

        with open(path, 'w') as fp:
            json.dump(train_coco, fp)


def create_coco_all_frame_valid(motion_type='motion', test_set=None, avoid_first_frame=False, condinst=True, no_class=None):
    img_path = f'{root}/youtubeVOS/all_frames/valid_all_frames/JPEGImages'
    detectron2_annos_path = f'{root}/youtubeVOS/train/'

    for d in ["all-frame-actual-valid"]: #"actual-test"
        DatasetCatalog.register("yvos_" + d,
                                lambda d=d: get_yvos_dicts(img_path, detectron2_annos_path, None, d, avoid_first_frame, condinst, no_class))
        MetadataCatalog.get("yvos_" + d).set(thing_classes=['object'] if no_class else categories)

        train_coco = convert_to_coco_dict('yvos_'+d)

        if no_class:
            path = f'{root}/boxinst/output/yvos_{d}_coco_format_no_class.json'
        else:
            path = f'{root}/boxinst/output/yvos_{d}_coco_format.json'

        with open(path, 'w') as fp:
            json.dump(train_coco, fp)


if __name__ == '__main__':
    # register_yvos()
    # create_coco_dict()
    root = '../data'
    split = 'train'  # 'valid'
    img_path = f'{root}/youtubeVOS/train/JPEGImages/'
    # annot_path = f'{root}/youtubeVOS/{split}/Annotations/'
    motion_path = f'{root}/youtubeVOS/train/motion/'
    detectron2_annos_path = f'{root}/youtubeVOS/train/'
    # a = get_yvos_dicts(img_path, detectron2_annos_path, motion_path, 'valid')
    # register_yvos(motion_type='motion', test_set=None, avoid_first_frame=False, no_class=True)
    # create_coco(motion_type='motion', test_set=None, avoid_first_frame=False,no_class=True)
    # create_coco_actual_test(motion_type='motion', test_set=None, avoid_first_frame=False, no_class=True)
    create_coco_all_frame_valid(motion_type='motion', test_set=None, avoid_first_frame=False, no_class=True)
    print()