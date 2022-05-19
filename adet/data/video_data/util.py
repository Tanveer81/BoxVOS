import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pack_padded_sequence as PACK
from PIL import Image  # (pip install Pillow)
from imantics import Polygons, Mask  # pip install imantics
import torch.nn.utils.rnn as rnnutils

# parameters for the motion calculation
motion_smoothing_kernel_size = 5
# motion_match_points = 200
motion_minimum_match_points = 15
motion_partial_affine = False#True
motion_reduce_factor = 14.0
motion_high_factor = 20.0


def calculate_motion(frame, previous_frame, motion_match_points = 100, as_channel=False, mask_image=None, forward_backward=True, sift=None):
    # convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY)
    frame_gray_blurred = cv2.blur(frame_gray, ksize=(
        motion_smoothing_kernel_size, motion_smoothing_kernel_size))
    previous_gray_blurred = cv2.blur(previous_gray, ksize=(
        motion_smoothing_kernel_size, motion_smoothing_kernel_size))

    # get the tracking points for creating the sparse optical flow
    if sift:
        tracking_points_previous, descriptors_1 = sift.detectAndCompute(previous_frame, mask_image)
        if len(tracking_points_previous) < motion_minimum_match_points:
            tracking_points_previous, descriptors_1 = sift.detectAndCompute(previous_frame, None)
        tracking_points_previous = cv2.KeyPoint_convert(tracking_points_previous)
        if len(tracking_points_previous) > 0:
            tracking_points_previous = tracking_points_previous[:, None,:]
    else:
        tracking_points_previous = cv2.goodFeaturesToTrack(previous_gray, maxCorners=motion_match_points,
                                                       qualityLevel=0.01, minDistance=15, corners=None, mask=mask_image)

    # check if there are enough features to create a sparse optical flow
    if tracking_points_previous is not None and len(tracking_points_previous) > motion_minimum_match_points:
        affine_matrix = get_affine_matrix(forward_backward, frame_gray, previous_gray, tracking_points_previous)

        if affine_matrix is None and mask_image is not None: #calculate without mask
            tracking_points_previous = cv2.goodFeaturesToTrack(previous_gray, maxCorners=motion_match_points,
                                                               qualityLevel=0.01, minDistance=15, corners=None,)

            if tracking_points_previous is not None and tracking_points_previous.shape[0] > motion_minimum_match_points:
                affine_matrix = get_affine_matrix(forward_backward, frame_gray, previous_gray, tracking_points_previous)

        # warp the blurred previous image to the current image with the affine transform
        if affine_matrix is not None:
            previous_warped = cv2.warpAffine(previous_gray_blurred, affine_matrix, frame_gray.shape[::-1])
        else:
            # no optical flow possible, default to a simple difference
            previous_warped = previous_gray_blurred

    else:
        # no optical flow possible, default to a simple difference
        previous_warped = previous_gray_blurred

    motion = np.abs(previous_warped.astype('float32') - frame_gray_blurred.astype('float32'))
    # Rubens special post processing
    motion[motion > 5] = np.clip(motion[motion > 5] * motion_high_factor, 0,
                                 255)  # .astype(np.uint8)
    motion[motion <= 5] = np.clip(motion[motion <= 5] * motion_reduce_factor, 0,
                                  255)  # .astype(np.uint8)
    if as_channel:
        return np.concatenate([frame, motion[:, :, None]], axis=-1)
    else:
        return motion.astype(np.uint8)


def get_affine_matrix(forward_backward, frame_gray, previous_gray, tracking_points_previous):
    tracking_points = cv2.calcOpticalFlowPyrLK(previous_gray, frame_gray, tracking_points_previous, None, maxLevel=10)[
        0]
    # forward backward optical flow
    if forward_backward:
        tracking_points_backward = \
        cv2.calcOpticalFlowPyrLK(frame_gray, previous_gray, tracking_points, None, maxLevel=10)[0]
        l2norm = np.linalg.norm(tracking_points_previous[:, 0, :] - tracking_points_backward[:, 0, :], axis=1)
        median = np.median(l2norm)
        tracking_points_previous = tracking_points_previous[l2norm <= median]
        tracking_points = tracking_points[l2norm <= median]
    if motion_partial_affine:
        affine_matrix = cv2.estimateAffinePartial2D(tracking_points_previous, tracking_points)[
            0]
    else:
        affine_matrix = cv2.estimateAffine2D(tracking_points_previous, tracking_points, maxIters=100, refineIters=100)[
            0]
    return affine_matrix


def calculate_motion_2(frame, previous_frame, next_frame, as_channel=False, motion_thresh=200, prev_mask_image=None,
                       next_mask_image=None, forward_backward=True, sift=None, three_frames=False, motion_match_points=100):
    m1 = calculate_motion(frame, previous_frame, motion_match_points, as_channel, prev_mask_image, forward_backward, sift)
    if three_frames:
        m2 = calculate_motion(frame, next_frame, motion_match_points, as_channel, next_mask_image, forward_backward, sift)
        motion = ((m1 - m2 < motion_thresh).astype(np.uint8)) * m1
        return motion
    return m1


def motion_heat_map(frames, previous_frames, next_frames, as_channel=False, motion_thresh=200, mask_images=None,
                      forward_backward=True, sift=None, motion_match_points=100, three_frames=False):
    '''for video_data'''
    motions = []
    for i in range(len(frames)):
        if i == 0:
            motions.append(calculate_motion_2(frames[i], previous_frames[i], next_frames[i] if three_frames else None, as_channel,
                                              motion_thresh, mask_images[1] if mask_images else mask_images,
                                              mask_images[2] if mask_images else mask_images, forward_backward,
                                              sift, three_frames, motion_match_points))
        elif i == len(frames)-1:
            motions.append(calculate_motion_2(frames[i], previous_frames[i], next_frames[i] if three_frames else None, as_channel,
                                              motion_thresh, mask_images[i-1] if mask_images else mask_images,
                                              mask_images[i-2] if mask_images else mask_images, forward_backward,
                                              sift, three_frames, motion_match_points))
        else:
            motions.append(calculate_motion_2(frames[i], previous_frames[i], next_frames[i] if three_frames else None, as_channel,
                                              motion_thresh, mask_images[i-1] if mask_images else mask_images,
                                              mask_images[i+1] if mask_images else mask_images, forward_backward,
                                              sift, three_frames, motion_match_points))
    return motions


def motion_heat_map_davis(frames, as_channel=False):
    '''for davis'''
    motions = []
    for i in range(len(frames) - 1):
        if i == 0:
            motions.append(calculate_motion(frames[i], frames[i + 1], as_channel))
        else:
            motions.append(calculate_motion(frames[i + 1], frames[i], as_channel))

    return motions


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def raft_flow(previous_warped, frame_gray_blurred):
    return previous_warped


def dense_optical_flow(previous_frame, previous_warped, frame_gray_blurred, algorithm='lucaskanade_dense'):

    hsv = np.zeros_like(previous_frame)
    hsv[..., 1] = 255
    # Calculate Optical Flow
    #pip install opencv-contrib-python
    if algorithm == "lucaskanade_dense":
        params = []
        flow = cv2.optflow.calcOpticalFlowSparseToDense(previous_warped, frame_gray_blurred, None, *params)
    elif algorithm == "farneback":
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback's algorithm parameters
        flow = cv2.calcOpticalFlowFarneback(previous_warped, frame_gray_blurred, None, *params)
    elif algorithm == "rlof":
        flow = cv2.optflow.calcOpticalFlowDenseRLOF(previous_warped, frame_gray_blurred, None)
    elif algorithm == 'RAFT':
        flow = raft_flow(previous_warped, frame_gray_blurred)

    # Encoding: convert the algorithm's output into Polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Use Hue and Saturation to encode the Optical Flow
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def visualize(img, cmap='binary'):
    plt.imshow(img, cmap=cmap)
    plt.show(block=True)


def visualize_bbox(image, bbox):
    # image = copy.deepcopy(image)
    image = np.ascontiguousarray(image)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    visualize(image)


def visualize_segmentation(img, mask, seq='title'):
    plt.subplot(2, 1, 1)
    plt.title(seq)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.imshow(mask)
    plt.show(block=True)


def visualize_polygon(image, pts):
    # image = copy.deepcopy(image)
    isClosed = True
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    for p in pts:
        p = np.array(p, np.int32).reshape((-1, 2))[:, None, :]
        cv2.polylines(image, [p], isClosed, color, thickness)
    visualize(image)


def seg_to_box(mot_0, img_0, mask_0, visualize=False, motion_avg=False, seq=(0, 0), seg_mask=True):
    lbl_0 = label(mask_0)
    props = regionprops(lbl_0)  # min_row, min_col, max_row, max_col

    masks = []
    for i, prop in enumerate(props):
        cv2.rectangle(img_0, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]),
                      (255, 0, 0), 2)
        # cv2.rectangle(mask_0, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
        mask = np.zeros(mot_0.shape, dtype="uint8")
        masks.append(
            cv2.rectangle(mask, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]),
                          i + 1, -1))

    if len(props) != 0:
        masks = (np.sum(np.array(masks), axis=0) > 0).astype("uint8")
        background_mask = 1 - masks
        if seg_mask:
            background_mask = masks * (1 - (mask_0 > 0).astype("uint8"))
            masks = mask_0
        mot_1 = cv2.bitwise_and(mot_0, mot_0, mask=masks)
        background_motion = cv2.bitwise_and(mot_0, mot_0, mask=background_mask)

        if visualize:
            os.makedirs(f'../../../frames/{seq[0]}', exist_ok=True)
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            axs[0, 0].set_title('Frame')
            if seg_mask:
                axs[0, 1].set_title('Background Motion')
            else:
                axs[0, 1].set_title('Segmentation Mask')
            axs[1, 0].set_title('Motion Map of the whole frame')
            axs[1, 1].set_title('Motion Map enclosed in derived bounding box')

            axs[0, 0].imshow(img_0)
            if seg_mask:
                axs[0, 1].imshow(background_motion, cmap='gray')
            else:
                axs[0, 1].imshow(mask_0, cmap='gray')
            axs[1, 0].imshow(mot_0, cmap='gray')  # background_motion
            if len(props) != 0:
                axs[1, 1].imshow(mot_1, cmap='gray')

            plt.show(block=True)
            # plt.savefig(f'../../../frames/{seq[0]}/{seq[1]}.png')
            plt.close(fig)

        if motion_avg:
            foreground_mean = mot_1.sum() / np.count_nonzero(mot_1)
            foreground_mean = foreground_mean / len(props)
            background_mean = background_motion.sum() / np.count_nonzero(background_motion)
            return props, foreground_mean, background_mean
        else:
            return props

    else:
        return props, 0, 0


def frames_to_video(image_folder, video_name, delete_images=False):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 6, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    if delete_images:
        for file in os.listdir(image_folder):
            os.remove(image_folder + file)


def simple_collate(batch):
    videos = [item[0] for item in batch]
    motions = [item[1] for item in batch]
    annots = [item[2] for item in batch]
    return [videos, motions, annots]


# https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/
class PadSequence:
    def __call__(self, batch):
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        video_sequences = [x[0] for x in sorted_batch]
        motion_sequences = [x[1] for x in sorted_batch]
        annot_sequences = [x[2] for x in sorted_batch]
        video_sequences_padded = torch.nn.utils.rnn.pad_sequence(video_sequences, batch_first=True)
        motion_sequences_padded = torch.nn.utils.rnn.pad_sequence(motion_sequences,
                                                                  batch_first=True)
        annot_sequences_padded = torch.nn.utils.rnn.pad_sequence(annot_sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in video_sequences])

        # # Don't forget to grab the labels of the *sorted* batch
        # labels = torch.LongTensor(map(lambda x: x[1], sorted_batch))
        return video_sequences_padded, motion_sequences_padded, annot_sequences_padded, lengths


def unpack_sequence(packed_sequence, lengths):
    assert isinstance(packed_sequence, rnnutils.PackedSequence)
    head = 0
    trailing_dims = packed_sequence.data.shape[1:]
    unpacked_sequence = [torch.zeros(l, *trailing_dims) for l in lengths]
    # l_idx - goes from 0 - maxLen-1
    for l_idx, b_size in enumerate(packed_sequence.batch_sizes):
        for b_idx in range(b_size):
            unpacked_sequence[b_idx][l_idx] = packed_sequence.data[head]
            head += 1
    return unpacked_sequence


def unpack_batch(video_sequences_padded, motion_sequences_padded, annot_sequences_padded, lengths):
    videos = PACK(video_sequences_padded, lengths, batch_first=True)
    videos = unpack_sequence(videos, lengths)

    motion = PACK(motion_sequences_padded, lengths, batch_first=True)
    motion = unpack_sequence(motion, lengths)

    annot = PACK(annot_sequences_padded, lengths, batch_first=True)
    annot = unpack_sequence(annot, lengths)

    return videos, motion, annot


def numpy_to_video(frames):
    width = frames.shape[0]
    hieght = frames.shape[1]
    if len(frames.shape) == 3:
        channel = 1
    else:
        channel = frames.shape[3]

    fps = 30
    sec = 5

    # Syntax: VideoWriter_fourcc(c1, c2, c3, c4) # Concatenates 4 chars to a fourcc code
    #  cv2.VideoWriter_fourcc('M','J','P','G') or cv2.VideoWriter_fourcc(*'MJPG)

    fourcc = cv2.VideoWriter_fourcc(
        *'MP42')  # FourCC is a 4-byte code used to specify the video codec.
    # A video codec is software or hardware that compresses and decompresses digital video.
    # In the context of video compression, codec is a portmanteau of encoder and decoder,
    # while a device that only compresses is typically called an encoder, and one that only
    # decompresses is a decoder. Source - Wikipedia

    # Syntax: cv2.VideoWriter( filename, fourcc, fps, frameSize )
    video = cv2.VideoWriter('test.mp4v', fourcc, float(fps), (width, hieght))

    for frame in frames:
        # img = np.random.randint(0, 255, (hieght, width, channel), dtype=np.uint8)
        video.write(frame[..., None])

    video.release()


# https://www.immersivelimit.com/create-coco-annotations-from-scratch
def create_sub_masks(mask_image):
    # param mask_image: PIL Image
    # return dict{object_id : int, sub_mask : PIL Image}
    mask_image = np.array(mask_image)
    object_ids = np.unique(np.array(mask_image))
    object_ids = np.delete(object_ids, 0)
    sub_masks = {}
    for object_id in object_ids:
        sub_mask = np.where(mask_image == object_id, 1, 0)
        sub_mask = Image.fromarray(sub_mask.astype(np.uint8))
        sub_masks[str(object_id)] = sub_mask

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    polygons = Mask(sub_mask).polygons()
    return polygons.segmentation  # , list(bbox)


def submask_to_box(mask_0, area=False):
    lbl_0 = label(np.array(mask_0))
    props = regionprops(lbl_0)  # min_row, min_col, max_row, max_col or YXYX
    if len(props) == 1:
        min_row, min_col, max_row, max_col = props[0].bbox  # YXYX
    else:
        # A single object (iscrowd=0) may require multiple polygons to represent, such as this object is blocked in the image.
        # merge multiple bbox
        min_row = min([props[i].bbox[0] for i in range(len(props))])
        min_col = min([props[i].bbox[1] for i in range(len(props))])
        max_row = max([props[i].bbox[2] for i in range(len(props))])
        max_col = max([props[i].bbox[3] for i in range(len(props))])

    if area:
        return (min_col, min_row, max_col, max_row), props[0].area  # XYXY

    return min_col, min_row, max_col, max_row  # XYXY
