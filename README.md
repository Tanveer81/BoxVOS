# Box Supervised Video Segmentation Proposal Network

https://user-images.githubusercontent.com/14295248/169266022-ffcb41d1-1c9d-4726-b2bd-625a64fc07d5.mp4


## Installation

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

*Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2.*

Then build the project with:

```
pip install fvcore==0.1.1.dev200512
python setup.py build develop
mkdir & cd .torch/fvcore_cache/detectron2/ImageNetPretrained/MSRA
wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl
Install Lycon from https://github.com/ethereon/lycon
```

# Download Imagenet Pretrained Model Weights
```
cd weights
wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl
wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl
cd ..
```

# Data Directory Setup
```
Put YoutubeVos and Davis datset folders inside data folder.
Then run the follwoing files to create COCO style JSON files and Motion
    * adet/data/video_data/davis_anot.py
    * adet/data/video_data/davis_motion.py
    * adet/data/video_data/yvos_annot{*}.py
    * adet/data/video_data/yvos_motion.py
```

# Train On YoutubeVOS Dataset
```
python tools/train_net.py 
--config-file ../configs/MBVOS/MS_R_101_BiFPN_dcni3_1x.yaml
video-config-file ../configs/BoxInst/Base-YVOS.yaml 
SOLVER.IMS_PER_BATCH 12 
YVOS True 
Motion DecoupledIntersection 
OUTPUT_DIR output/YVOS_101 
TEST.EVAL_PERIOD 10000 
motion_type motion_200_2_2_2 
pairwise_motion_thresh 0.3 
no_class True
root ../data
```
# Finetune on DAVIS Dataset

```
python tools/train_net.py 
--config-file ../configs/MBVOS/MS_R_101_BiFPN_dcni3_1x.yaml 
video-config-file ../configs/MBVOS/Base-DAVIS.yaml
SOLVER.IMS_PER_BATCH 12 
DAVIS True 
Motion DecoupledIntersection 
OUTPUT_DIR output/DAVIS_101 
pairwise_motion_thresh 0.3 
TEST.EVAL_PERIOD 350 
SOLVER.CHECKPOINT_PERIOD 350 
MODEL.WEIGHTS YVOS_101/model_0069999.pth 
SOLVER.BASE_LR 0.0001 
SOLVER.MAX_ITER 6300
root ../data
```

# Evaluate Model

```
python tools/train_net.py 
--eval-only
--config-file ../configs/MBVOS/MS_R_101_BiFPN_dcni3_1x.yaml 
video-config-file ../configs/MBVOS/Base-DAVIS.yaml
DAVIS True 
Motion DecoupledIntersection 
OUTPUT_DIR output/DAVIS_101_EVAL 
MODEL.WEIGHTS DAVIS_101/model_0079999.pth 
root ../data
```

# Visualize Predictions

```
python demo/demo.py 
--config-file ../configs/MBVOS/MS_R_101_BiFPN_dcni3_1x.yaml 
--input input_image_path
--output output_image_path 
--opts 
MODEL.WEIGHTS output/DAVIS_101/model_0006299.pth  
motion None 
video-config-file ../configs/MBVOS/Base-YVOS_no_class.yaml 
YVOS True
```

# Pretrained Weights
```
The weights for both the Davis and Youtube-VOS datasets can be found at this  [link]([[https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md](https://syncandshare.lrz.de/getlink/fiW3kjxv4XB6KvrXVVXCAMBn/)].

```
