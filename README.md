# Detection of Spinal Lesions in X-ray Images Using a Combined Classification and Detection Approach


## Authors

- **Pham Ngoc Hao** – Student ID: 23110146  
- **Nguyen Dang Van Canh** – Student ID: 23110135  

Faculty of Mathematics and Computer Science  
University of Science, Vietnam National University Ho Chi Minh City (VNU-HCM)


## Problem
Given a spinal X-ray image, the objective of this project is to automatically detect and classify
multiple types of spinal lesions.

- **Input:** A single spinal X-ray image.
- **Output:** A set of detected lesions, each represented by a bounding box, a lesion label,
  and a confidence score.

## Installation
To install via Pip, run
```bash
pip install -r ./requirements/requirements.txt
python -m pip install git+https://github.com/facebookresearch/detectron2.git

```
## Data Preparation
This project implemented using dataset VinDR-SpineXR. For more detail, see [Data Preparation](./data/README.md)

## Training
Both of two models after training will be saved their in ```text ./ouputs/```. In this project, we use two model namely: DenseNet-210 and Sparse R-CNN. For the former training task, run the script
```bash
cls_train_net.bat
```
Before training the latter. Please see [Pretrained Instruction](./pretrained/README.md).

Run the script
```bash
det_train.bat
```

## Evaluation
Run the script
```bash
eval_all.bat
```

## Saving predicted results to file JSON file   
Run the script
```bash
infer
```

## Visualization results
Run the script
```bash
visualize.bat
```