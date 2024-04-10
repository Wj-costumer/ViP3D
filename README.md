# ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries (CVPR 2023)
### [Paper](https://arxiv.org/abs/2208.01582) | [Webpage](https://tsinghua-mars-lab.github.io/ViP3D/)
- This is the official repository of the paper: **ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries** (CVPR 2023).

[//]: # (## Getting Started)

[//]: # (- Installation)

[//]: # (- Prepare Dataset)

[//]: # (- Training and Evaluation)

##  Installation
Use the following commands to prepare the python environment. 
#### 1) Create conda environment
```bash
conda create -n vip3d python=3.6
```
Supported python versions are 3.6, 3.7, 3.8. 
#### 2) Install pytorch
```bash
conda activate vip3d
pip install torch==1.10+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
#### 3) Install mmcv, mmdet
```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install mmdet==2.24.1
```

#### 4) Install other packages
```bash
pip install -r requirements.txt
```

#### 5) Install mmdet3d
```bash
cd ~
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install / pip install -e .
pip install -r requirements/runtime.txt  # Install packages for mmdet3d
```

### Quick start with Docker (Optional)
We also provide a docker image of ViP3D, which has installed all required packages. The docker image is built from NVIDIA container image for PyTorch. Make sure you have installed docker and nvidia docker.

```bash
docker pull gentlesmile/vip3d
docker run --name vip3d_container -it --gpus all --ipc=host gentlesmile/vip3d
```

## Prepare Dataset
#### 1) Download nuScenes full dataset (v1.0) and map expansion [here](https://www.nuscenes.org/download).
Only need to download Keyframe blobs and Radar blobs.


#### 2) Structure
After downloading, the structure is as follows:
```
ViP3D
├── mmdet3d/
├── plugin/
├── tools/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── v1.0-trainval/
│   │   ├── lidarseg/
```

#### 3) Prepare data infos
Suppose nuScenes data is saved at ```data/nuscenes/```.
```bash
python tools/data_converter/nusc_tracking.py
```

##  Training and Evaluation

### Training
Train ViP3D using 3 historical frames and the ResNet50 backbone. It will load a pre-trained detector for weight initialization. Suppose the detector is at ```ckpts/detr3d_resnet50.pth```. It can be downloaded from [here](https://drive.google.com/drive/folders/18q2sQ-J-AxqeCO8FaAWKQ9Fi13PPv_MR?usp=drive_link).
```bash
bash tools/dist_train.sh plugin/configs/vip3d_resnet50_3frame.py 8 --work-dir=work_dirs/vip3d_resnet50_3frame.1
python tools/train.py plugin/hivt/configs/hivt_resnet50_3frame.py --work-dir=hivt_results.1 --gpus=1 
```
The training stage requires ~ 17 GB GPU memory, and takes ~ 3 days for 24 epochs on 8× 3090 GPUS.

### Evaluation

Run evaluation using the following command:
```bash
PYTHONPATH=. python tools/test.py plugin/vip3d/configs/vip3d_resnet50_3frame.py work_dirs/vip3d_resnet50_3frame.1/epoch_24.pth --eval bbox

python tools/test.py plugin/vip3d/configs/vip3d_resnet50_3frame.py ./ckpts/epoch_24.pth --eval bbox
```
The checkpoint ```epoch_24.pth``` can be downloaded from [here](https://drive.google.com/drive/folders/18q2sQ-J-AxqeCO8FaAWKQ9Fi13PPv_MR?usp=drive_link).

Expected AMOTA using ResNet50 as backbone: 0.291

Then test prediction metrics:
```bash
unzip ./nuscenes_prediction_infos_val.zip
```
```bash
python tools/prediction_eval.py --result_path 'work_dirs/vip3d_resnet50_3frame.1/results_nusc.json'
```

Expected results: minADE: 1.47, minFDE: 2.21, MR: 0.237, EPA: 0.245

## License
The code and assets are under the Apache 2.0 license.

## Citation
If you find our work useful for your research, please consider citing the paper:
```bash
@inproceedings{vip3d,
  title={ViP3D: End-to-end visual trajectory prediction via 3d agent queries},
  author={Gu, Junru and Hu, Chenxu and Zhang, Tianyuan and Chen, Xuanyao and Wang, Yilun and Wang, Yue and Zhao, Hang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5496--5506},
  year={2023}
}
```
