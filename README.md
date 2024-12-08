# PyTorch YOLO
A minimal PyTorch implementation of YOLOv3, with support for training, inference, and evaluation.

YOLOv4 and YOLOv7 weights are also compatible with this implementation.

## about

This is an improvement based on the code of another dalao Looking at the [original repository address](https://github.com/eriklindernoren/PyTorch-YOLOv3), here I have made some improvements that I think may be effective in improving performance, but there is no guarantee of correctness

## Installation
### Installing from source

For normal training and evaluation we recommend installing the package from source using a poetry virtual environment.

```bash
git clone https://github.com/yuqiu2004/pytorch-yolo-v3-altered.git
cd pytorch-yolo-v3-altered/
pip3 install poetry --user
poetry install
```

You need to join the virtual environment by running `poetry shell` in this directory before running any of the following commands without the `poetry run` prefix.

#### Download pretrained weights

```bash
cd weights
./download_weights.sh
cd ..
```

#### Download COCO

```bash
cd data
./get_coco_dataset.sh
cd ..
```

### Install via pip

```bash
pip3 install pytorchyolo --user
```

## Test
Evaluates the model on COCO test dataset.

```bash
poetry run yolo-test --weights weights/yolov3.weights
```

## Inference
Uses pretrained weights to make predictions on images.

```bash
poetry run yolo-detect --images data/samples/
```

## Train
For argument descriptions have a look at `poetry run yolo-train --help`

#### Example (COCO)
To train on COCO using a Darknet-53 backend pretrained on ImageNet run:

```bash
poetry run yolo-train --data config/coco.data  --pretrained_weights weights/darknet53.conv.74
```

#### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below
* Go to http://localhost:6006/

```bash
poetry run tensorboard --logdir='logs' --port=6006
```

## Train on Custom Dataset

#### Custom model
Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your dataset.

```bash
cd config 
./create_custom_model.sh <num-classes>  # Will create custom model 'yolov3-custom.cfg'
```

#### Classes
Add class names to `data/custom/classes.names`. This file should have one row per class name.

#### Image Folder
Move the images of your dataset to `data/custom/images/`.

#### Annotation Folder
Move your annotations to `data/custom/labels/`.

#### Define Train and Validation Sets
In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation data respectively.

#### Train
To train on the custom dataset run:

```bash
poetry run yolo-train --model config/yolov3-custom.cfg --data config/custom.data
```

