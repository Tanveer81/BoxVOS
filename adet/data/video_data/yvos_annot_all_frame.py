import json
import time
from glob import glob
from adet.data.video_data.util import *
from PIL import Image
import os

categories = ['object']


def generate_detectron2_annotations_test(img_path, detectron2_annos_path):
    video_id_names = next(os.walk(img_path))[1]
    video_id_names.sort()
    detectron2_annos = {}
    detectron2_annos['annos'] = []
    detectron2_annos['bbox_mode'] = 'BoxMode.XYXY_ABS'
    errors = {}
    start_time = time.time()
    for i, video_id in enumerate(video_id_names):
        img_paths = np.sort(glob(os.path.join(img_path, video_id, '*.jpg'))).tolist()
        for j in range(0, len(img_paths)):
            try:
                file_name = img_paths[j]  # path
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
    with open(f'{detectron2_annos_path}/detectron2-annotations-all-frame-actual-valid.json', 'w') as outfile:
        json.dump(detectron2_annos, outfile)


def main():
    root = '../data/youtubeVOS/all_frames/valid_all_frames'
    img_path = f'{root}/JPEGImages/'
    detectron2_annos_path = '../data/youtubeVOS/train/'
    generate_detectron2_annotations_test(img_path, detectron2_annos_path)


if __name__ == '__main__':
    main()
