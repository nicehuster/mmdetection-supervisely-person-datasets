
## mmdetection-supervisely


We used mmdetection(mask rcnn) to train supervisely person dataset.

![demo image](pretrained_supervise_res.jpg)

### Installation

```shell
git clone https://github.com/nicehuster/mmdetection-supervisely-person-datasets.git

cd mmdetection-supervisely-person-datasets

export PYTHONPATH=$PYTHONPATH:{pwd}
```
### Data

You can download supervisely person dataset from the [official website](https://supervise.ly/), after download the dataset, place it in data folder, and The dataset is structured as follows：
```
.
├── ds1
│   ├── ann
│   └── img
...
├── ds11
│   ├── ann
│   └── img
├── ds12
│   ├── ann
│   └── img
├── ds13
│   ├── ann
│   └── img
└── meta.json
```

### Test image

You can download the pretrained model trained on supervisely person dataset in [baidu,passwd:ytiv](https://pan.baidu.com/s/1b8buEocVXX9Lp0M7HDnguQ)

```
python demox.py
```

### train a model

```
python tools/train.py configs/supervisely.py --gpus 2 
```

**Important**: 
* you need to set the learning rate proportional to the GPU num. E.g., modify lr to 0.01 for 4 GPUs or 0.005 for 2 GPUs.
* In this repository, the pytorch version is 0.4.1

