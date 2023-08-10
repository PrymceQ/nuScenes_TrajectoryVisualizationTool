# nuScenes_TrajectoryVisualizationTool
The visualization project of the BEV model inference result on [NuScenes Dataset](https://www.nuscenes.org/), including detBox and trackingLine!
> We use Nuscenes v1.0-mini DATASET for example

First of all, I need to thank some Github authors! [PF-Track](https://github.com/TRI-ML/PF-Track) | [SimpleTrack](https://github.com/tusen-ai/SimpleTrack) | [StreamPETR](https://github.com/exiawsh/StreamPETR)

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
