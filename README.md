# BEVTrack_vistool
The visualization project of the BEV model inference result, including detBox and trackingLine!
> We use Nuscenes v1.0-mini DATASET for example



Thanks for the BEVFusion authors！[Paper](https://arxiv.org/abs/2110.06922) | [Code](https://github.com/WangYueFt/detr3d)

## 🌵Necessary File Format
- data/nuscenes/
  - v1.0-mini/
    - maps/
    - samples/
    - sweeps/
    - v1.0-mini/
- projects/
- tools/

## 🌵Build Envs



## 🌵Data create

```
python tools/create_data.py nuscenes-tracking --root-path data/nuscenes/v1.0-mini --out-dir data/nuscenes/v1.0-mini --extra-tag tracking_forecasting --version v1.0-mini --forecasting
```


## 🌵Camera Code

```
tools\camera_visualization.py --result mini_track.json --show-dir work_dirs/cam_visualization/
```

`mini_track.json` is the inference json with `tracking_id`

## 🌵Key Model Files
