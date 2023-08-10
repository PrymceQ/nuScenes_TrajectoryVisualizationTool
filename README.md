# nuScenes_TrajectoryVisualizationTool
The visualization project of the BEV model inference result on [NuScenes Dataset](https://www.nuscenes.org/), including detBox and trackingLine!

First of all, I need to thank some Github authors! [PF-Track](https://github.com/TRI-ML/PF-Track) | [SimpleTrack](https://github.com/tusen-ai/SimpleTrack) | [StreamPETR](https://github.com/exiawsh/StreamPETR)

> We use Nuscenes v1.0-mini DATASET for example

<img src="work_dirs/cam_visualization/fcbccedd61424f1b85dcbf8f897f9754/video 00_00_00-00_00_30.gif" width="660px">

<img src="work_dirs/tracking_visualization/fcbccedd61424f1b85dcbf8f897f9754/videobev%2000_00_00-00_00_30.gif" width="660px">


## 🌵Necessary File Format
- data/nuscenes/
  - v1.0-mini/
    - maps/
    - samples/
    - sweeps/
    - v1.0-mini/
- projects/
- tools/
- SimpleTrack/

## 🌵Build Envs

You can refer to the [PF-Track](https://github.com/TRI-ML/PF-Track) configuration environment documentation. 

Or use the Conda env configuration file we provide.

```
conda env create -f nuScenesTrajectoryVisualizationTool_env.yaml
```

as for `mot_3d` package,

```
cd SimpleTrack
pip install -e ./
```

## 🌵Data Create
You need to create `pkl` files for v1.0mini: 

`tracking_forecasting-mini_infos_train.pkl`

`tracking_forecasting-mini_infos_val.pkl`

```
python tools/create_data.py nuscenes-tracking --root-path data/nuscenes/v1.0-mini --out-dir data/nuscenes/v1.0-mini --extra-tag tracking_forecasting --version v1.0-mini --forecasting
```

## 🌵Data Prepare

The final `json` data structure should like this, it should be notice that key `tracking_id` is necessary.

<img src="https://github.com/PrymceQ/nuScenes_TrajectoryVisualizationTool/blob/master/work_dirs/jpg1.png" width="460px">

You can use the [code](https://github.com/PrymceQ/BEVModel_StreamPETR) here to prepare the `json` file with `tracking_id` key from the test result `json` file. 

## 🌵Camera Visualization Code

```
tools/camera_visualization.py --result mini_track.json --show-dir work_dirs/cam_visualization/
```

## 🌵BEV Visualization Code

```
tools/bev_traj_visualization.py projects/configs/tracking/petr/f3_q500_800x320.py --result mini_track.json --show-dir work_dirs/tracking_visualization/
```
