# BEVTrack_vistool
The visualization project of the BEV model inference result, including detBox and trackingLine!

# 3DDetection_DETR3D
Private The reproduction project of the 3D Detection model # DETR3D, which includes some code annotation work

Thanks for the BEVFusion authors！[Paper](https://arxiv.org/abs/2110.06922) | [Code](https://github.com/WangYueFt/detr3d)

## 🌵Necessary File Format
- mmdetection3d/ # https://github.com/open-mmlab/mmdetection3d/
- data/nuscenes/
  - maps/
  - samples/
  - sweeps/
  - v1.0-test/
  - v1.0-trainval/
- pretrained/
- projects/
  - configs/
  - mmdet3d_plugin/
- tools/
- work_dirs/detr3d_res101_gridmask/

## 🌵Build Envs



## 🌵Data create

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```



## 🌵Train Code



## 🌵Key Model Files
