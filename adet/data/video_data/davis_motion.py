import sys
import time
from glob import glob
from pathlib import Path
from adet.data.video_data.util import *
from PIL import Image
import os
import random

root = '../data/DAVIS_2019_unsupervised_480/trainval'
img_path = f'{root}/JPEGImages/480p/'
annot_path = f'{root}/Annotations_unsupervised/480p/'


def generate_motion(annot_path, img_path, motion_path, three_frames=True, motion_thresh=200, bbox=False, forward_backward=True, sift=None, motion_match_points=100):
    Path(motion_path).mkdir(parents=True, exist_ok=True)
    sequences_names = os.listdir(img_path)
    # random.shuffle(sequences_names)
    for j, seq in enumerate(sequences_names):
        print(f'{j + 1}/{len(sequences_names)}: {seq}')
        frames = []
        previous_frames = []
        next_frames = []
        images_paths = np.sort(glob(os.path.join(img_path, seq, '*.jpg'))).tolist()
        annotations_paths = np.sort(glob(os.path.join(annot_path, seq, '*.png'))).tolist()
        mask_images = [] if bbox else None
        for i in range(0, len(images_paths)):
            if i == len(images_paths)-1 and three_frames:
                previous_index = i - 1
                next_index = i -2
            else:
                previous_index = 1 if i == 0 else i - 1
                next_index = 2 if i == 0 else i + 1
            frame = np.array(Image.open(images_paths[i]))
            previous_frame = np.array(Image.open(images_paths[previous_index]))
            if three_frames:
                next_frame = np.array(Image.open(images_paths[next_index]))
            if bbox:
                mask_image = Image.open(annotations_paths[i])
                sub_masks = create_sub_masks(mask_image)
                mask_image = np.ones_like(mask_image)
                for object_id, sub_mask in sub_masks.items():
                    bbox, _ = submask_to_box(sub_mask, True)  # xyxy format
                    cv2.rectangle(mask_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 0, -1)
                mask_images.append(mask_image)

            frames.append(frame)
            previous_frames.append(previous_frame)
            if three_frames:
                next_frames.append(next_frame)
        motions = motion_heat_map(frames, previous_frames, next_frames, False, motion_thresh, mask_images,
                                  forward_backward, sift, motion_match_points, three_frames)

        # save motion maps as ndarray
        Path(f'{motion_path}{seq}').mkdir(parents=True, exist_ok=True)
        motions = np.array(motions)
        with open(f'{motion_path}{seq}/motions.npy', 'wb') as f:
            np.save(f, motions)


def visualize_motion(motion_path):
    Path(motion_path).mkdir(parents=True, exist_ok=True)
    sequences_names = os.listdir(motion_path)
    random.shuffle(sequences_names)
    for i, seq in enumerate(sequences_names):
        print(f'{i + 1}/{len(sequences_names)}: {seq}')
        seq = 'rhino'
        frames = []
        masks = []
        images_paths = np.sort(glob(os.path.join(img_path, seq, '*.jpg'))).tolist()
        annotations_paths = np.sort(glob(os.path.join(annot_path, seq, '*.png'))).tolist()

        for i in range(0, len(annotations_paths)):
            frame = np.array(Image.open(images_paths[i]))
            mask = np.array(Image.open(annotations_paths[i]))
            frames.append(frame)
            masks.append(mask)
        with open(f'{motion_path}{seq}/motions.npy', 'rb') as f:
            motions = np.load(f)

        visualize(frames[0])
        visualize(motions[0])
        visualize(frames[3])
        visualize(motions[3])
        visualize(frames[-1])
        visualize(motions[-1])
        print()


def motion_base():
    # None
    three_frames = False
    motion_thresh = 200
    bbox = False
    forward_backward = False
    sift = None#cv2.xfeatures2d.SIFT_create()
    motion_path = f'{root}/motion_base/'
    start_time = time.time()
    motion_match_points = 100
    generate_motion(annot_path, img_path, motion_path, three_frames, motion_thresh, bbox, forward_backward, sift, motion_match_points)
    # visualize_motion(annot_path, img_path, motion_path)
    total = time.time() - start_time
    print(f'total time: {total}s')


def motion_200_2_2_2():
    # ALL
    three_frames = True
    motion_thresh = 200
    bbox = True
    forward_backward = True
    sift = None#cv2.xfeatures2d.SIFT_create()
    motion_path = f'{root}/motion_all/'
    start_time = time.time()
    motion_match_points = 200
    generate_motion(annot_path, img_path, motion_path, three_frames, motion_thresh, bbox, forward_backward, sift, motion_match_points)
    # visualize_motion(annot_path, img_path, motion_path)
    total = time.time() - start_time
    print(f'total time: {total}s')


# ony three frame sim
def motion_3f():
    three_frames = True
    motion_thresh = 200
    bbox = False
    forward_backward = False
    sift = None#cv2.xfeatures2d.SIFT_create()
    motion_match_points = 200
    motion_path = f'{root}/motion_3f/'
    start_time = time.time()
    generate_motion(annot_path, img_path, motion_path, three_frames, motion_thresh, bbox, forward_backward, sift, motion_match_points)
    # visualize_motion(annot_path, img_path, motion_path)
    total = time.time() - start_time
    print(f'total time: {total}s')


def motion_b3f():
    three_frames = True
    motion_thresh = 200
    bbox = True
    forward_backward = False
    sift = None#cv2.xfeatures2d.SIFT_create()
    motion_match_points = 200
    motion_path = f'{root}/motion_b3f/'
    start_time = time.time()
    generate_motion(annot_path, img_path, motion_path, three_frames, motion_thresh, bbox, forward_backward, sift, motion_match_points)
    # visualize_motion(annot_path, img_path, motion_path)
    total = time.time() - start_time
    print(f'total time: {total}s')


def motion_fb3f():
    three_frames = True
    motion_thresh = 200
    bbox = False
    forward_backward = True
    sift = None#cv2.xfeatures2d.SIFT_create()
    motion_match_points = 200
    motion_path = f'{root}/motion_fb3f_fix/'
    start_time = time.time()
    generate_motion(annot_path, img_path, motion_path, three_frames, motion_thresh, bbox, forward_backward, sift, motion_match_points)
    # visualize_motion(annot_path, img_path, motion_path)
    total = time.time() - start_time
    print(f'total time: {total}s')


def motion_b():
    three_frames = False
    motion_thresh = 200
    bbox = True
    forward_backward = False
    sift = None#cv2.xfeatures2d.SIFT_create()
    motion_match_points = 200
    motion_path = f'{root}/motion_b/'
    start_time = time.time()
    generate_motion(annot_path, img_path, motion_path, three_frames, motion_thresh, bbox, forward_backward, sift, motion_match_points)
    # visualize_motion(annot_path, img_path, motion_path)
    total = time.time() - start_time
    print(f'total time: {total}s')


def motion_fb():
    three_frames = False
    motion_thresh = 200
    bbox = False
    forward_backward = True
    sift = None#cv2.xfeatures2d.SIFT_create()
    motion_match_points = 200
    motion_path = f'{root}/motion_fb_fix/'
    start_time = time.time()
    generate_motion(annot_path, img_path, motion_path, three_frames, motion_thresh, bbox, forward_backward, sift, motion_match_points)
    # visualize_motion(annot_path, img_path, motion_path)
    total = time.time() - start_time
    print(f'total time: {total}s')


def main():
    motion_type = sys.argv[1]
    if motion_type=='motion_200_2_2_2':
        motion_200_2_2_2()
    elif motion_type=='motion_3f':
        motion_3f()
    elif motion_type=='motion_b3f':
        motion_b3f()
    elif motion_type=='motion_fb3f':
        motion_fb3f()
    elif motion_type=='motion_b':
        motion_b()
    elif motion_type=='motion_fb':
        motion_fb()
    elif motion_type == 'motion_base':
        motion_base()


if __name__ == '__main__':
    main()
    # visualize_motion(motion_path = f'{root}/motion_200_2_2_2/')
