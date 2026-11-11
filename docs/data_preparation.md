# Data Preparation

## 1. nuScenes

Download the [nuScenes v1.0 full dataset](https://www.nuscenes.org/download) and the nuScenes-lidarseg extension to `data/nuscenes`.

## 2. Annotation files

Generate the info files from the AGO root with the mmdetection3d checkout from the [installation](installation.md):

```shell
cd /path/to/AGO
PYTHONPATH=/path/to/mmdetection3d python /path/to/mmdetection3d/tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

## 3. Occ3D-nuScenes ground truth

Download `gts` from [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D#occ3d-nuscenes) and place it in `data/nuscenes/occ3d`.

## 4. Pseudo labels

Pseudo labels are required for training only. Download them from [Hugging Face](https://huggingface.co/datasets/EdwardLeeLPZ/AGO-Occ3D-nuScenes-pseudo-labels) and extract them:

```shell
wget https://huggingface.co/datasets/EdwardLeeLPZ/AGO-Occ3D-nuScenes-pseudo-labels/resolve/main/occ3d_pseudo_labels.tar
tar xf occ3d_pseudo_labels.tar -C data/nuscenes/occ3d
```

## 5. Pretrained backbone

```shell
mkdir -p checkpoints
wget -P checkpoints https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tpvformer/tpvformer_8xb1-2x_nus-seg/tpvformer_pretrained_fcos3d_r101_dcn.pth
```

## Folder structure

```
AGO
├── checkpoints
│   └── tpvformer_pretrained_fcos3d_r101_dcn.pth
└── data
    └── nuscenes
        ├── lidarseg
        ├── maps
        ├── samples
        ├── sweeps
        ├── v1.0-trainval
        ├── nuscenes_infos_train.pkl
        ├── nuscenes_infos_val.pkl
        └── occ3d
            ├── gts
            │   └── [scene_name]/[sample_token]/labels.npz
            └── pseudo_labels
                └── [scene_name]/[sample_token]/labels.npz
```
