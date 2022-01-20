# HAILO README
A repository for training CenterPose.
forked from [tensorboy/centerpose](https://github.com/tensorboy/centerpose).

## Changes
- Added support to "FPN" meta architecture
- Added support for [pycls](https://github.com/facebookresearch/pycls) networks
- Added support for [RepVGG](https://github.com/DingXiaoH/RepVGG) networks
- Added support for tensorboard
- Added export.py to export onnx model
- Minor bug fixes

## Installation
- Follow the original instructions below. You can use a more modern pytorch (1.5/1.6) than listed.
- Install [pycls](https://github.com/facebookresearch/pycls/blob/master/docs/GETTING_STARTED.md) for regnet backbone support
- (Optional) Install tensorboard via pip: `pip install tensorboard`. If using really old pytorch, install tensorboardX instead.

## Usage
### Setting up TRAINS
- Install [TRAINS](https://allegro.ai/blog/setting-up-allegro-ai-platform/)
- If you don't have one, setup a [TRAINS server](https://allegro.ai/docs/deploying_trains/trains_server_config/)
- [Configure TRAINS](https://allegro.ai/docs/deploying_trains/trains_config_for_trains_server/) to connect to your server 
- Go to `train.py` and look for Task.init call. Change the parameters to fit your experiment.
### Train centerpose regnet_fpn
- Create an experiment configuration, such as: experiments/regnet_fpn.yaml
- Choose a backbone (e.g. RegNetX-3.2GF_dds_8gpu)
  - Update CONFIG_STRING in the .yaml file to match your backbone
  - Download the model YAML & pyth from the [Model Zoo](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md) and place them in the root directory
- Start training using the instructions below
### Export to onnx
- run demo.py, e.g.: `python tools/demo.py --cfg experiments/regnet_fpn.yaml --TEST regnet3.2_fpn_best.pth`

# ORIGINAL README
# The repo is based on [CenterNet](https://arxiv.org/abs/1904.07850), which aimed for push the boundary of human pose estimation
multi person pose estimation using center point detection:
![](readme/fig2.png)

## Main results

### Keypoint detection on COCO validation 2017
<p align="center"> <img src='readme/performance.png' align="center" height="512px"></p>

| Backbone     |  AP       |  FPS         | TensorRT Speed | GFLOPs |Download |
|--------------|-----------|--------------|----------|----------|----------|
|DLA-34        | 62.7      |    23      |  - |  - |[model](https://drive.google.com/open?id=1IahJ3vpjTVu1p-Okf6lcn-bM7fVKNg6N)  |
|Resnet-50     | 54.5     |    28      |  33 |  - |[model](https://drive.google.com/open?id=1oBgWrfigo2fGtpQJXQ0stADTgVFxPWGq)  |
|MobilenetV3   | 46.0      |    30      |  - |  - |[model](https://drive.google.com/open?id=1snJnADAD1NUzyO1QXCftuZu1rsr8095G)  |
|ShuffleNetV2  | 43.9      |    25      |  - |  - |[model](https://drive.google.com/open?id=1FK7YQzCB6mLcb0v4SOmlqtRJfA-PQSvN)  |
|[HRNet_W32](https://drive.google.com/open?id=1mJoK7KEx35Wgf6uAZ-Ez5IwAeOk1RYw0)| 63.8 |    16      |  - |  - |[model](https://drive.google.com/open?id=1X0yxGeeNsD4VwU2caDo-BaH_MoCAnU_J)  |
|[HardNet](https://github.com/PingoLH/FCHarDNet)| 46.0    |    30        | -  |  - |[model](https://drive.google.com/open?id=1CFc_qAAT4NFfrAG8JOxRVG8CAw9ySuYp)  |
|[Darknet53]()| 34.2    |    30        | -  |  - |[model](https://drive.google.com/open?id=1S8spP_QKHqIYmWpfF9Bb4-4OoUXIOnkh)  |
|[EfficientDet]()| 38.2    |    30        | -  |  - |[model](https://drive.google.com/open?id=1S8spP_QKHqIYmWpfF9Bb4-4OoUXIOnkh)  |

## Installation

git submodule init&git submodule update
Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use CenterNet

We support demo for image/ image folder, video, and webcam. 

First, download the model [DLA-34](https://drive.google.com/open?id=1OkHjjViB0dzbuicdtIam-YcoT0sYpmjP)
from the [Model zoo](https://drive.google.com/open?id=1UG2l8XtjOfBtG_GLpSdxlWS2wxFR8hQF) and put them in anywhere.

Run:
    
~~~
cd tools; python demo.py --cfg ../experiments/dla_34_512x512.yaml --TESTMODEL /your/model/path/dla34_best.pth --DEMOFILE ../images/33823288584_1d21cf0a26_k.jpg --DEBUG 1
~~~
The result for the example images should look like:
<p align="center"> <img src='readme/multi_pose_screenshot_27.11.2019.png' align="center" height="512px"></p>

## Evaluation 
~~~
cd tools; python evaluate.py --cfg ../experiments/dla_34_512x512.yaml --TESTMODEL /your/model/path/dla34_best.pth --DEMOFILE --DEBUG 0
~~~

## Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets.

We provide config files for all the experiments in the [experiments](experiments) folder.

```
cd ./tools python -m torch.distributed.launch --nproc_per_node 4 train.py --cfg ../experiments/*yalm
```

## Demo

the demo files located in the `demo` directory, which is would be a very robust human detection+tracking+face reid system.

<p align="left">
<img src="./readme/demo.gif", width="720">
</p>

## License

MIT License (refer to the LICENSE file for details).

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2019objects,
      title={Objects as Points},
      author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:1904.07850},
      year={2019}
    }
