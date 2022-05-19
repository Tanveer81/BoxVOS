import json
import sys
import time
from glob import glob
from adet.data.video_data.util import *
from PIL import Image
import os


def generate_detectron2_annotations(img_path, detectron2_annos_path, meta_path, split):
    f = open(meta_path, )
    data = json.load(f)
    f.close()
    video_id_names = list(data['videos'].keys())
    video_id_names.sort()
    detectron2_annos = {}
    detectron2_annos['annos'] = []
    detectron2_annos['bbox_mode'] = 'BoxMode.XYXY_ABS'
    errors = {}
    start_time = time.time()
    for i, video_id in enumerate(video_id_names):
        imgs_paths = np.sort(glob(os.path.join(img_path, video_id, '*.jpg'))).tolist()
        for j in range(0, len(imgs_paths)):
            try:
                file_name = imgs_paths[j]  # path
                image = Image.open(file_name)

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

    with open(f'{detectron2_annos_path}/detectron2-annotations-actual-{split}-balanced.json', 'w') as outfile:
        json.dump(detectron2_annos, outfile)

    with open(f'{detectron2_annos_path}/detectron2-annotations-errors-actual-{split}-full-balanced.json', 'w') as outfile:
        json.dump(errors, outfile)


def main():
    for split in ['valid', 'test']:
        root = '../data'
        img_path = f'{root}/youtubeVOS/{split}/JPEGImages/'
        detectron2_annos_path = f'{root}/youtubeVOS/train/'
        meta_path = f'{root}/youtubeVOS/{split}/meta.json'
        generate_detectron2_annotations(img_path, detectron2_annos_path, meta_path, split)


if __name__ == '__main__':
    main()
