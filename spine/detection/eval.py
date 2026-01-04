import argparse
from ultralytics import YOLO
import os
import torch

def evaluation(weights_path, data_yaml, batch= 16, split = 'test'):
    if not os.path.exists(weights_path):
        return
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(weights_path)

    metrics = model.val(
        data = data_yaml,
        split = split,
        imgsz = 640,
        batch = batch,
        conf = 0.001,
        iou = 0.6,
        plots = True,
        device = device
    )

    names = model.names  # dict {id: class_name}

    print("\nPer-class AP@50–95:")
    for i, ap in enumerate(metrics.box.maps):
        print(f"{names[i]:<25}: {ap:.4f}")


    print(f'mAP@50 | {metrics.box.map50:.4f}')
    print(f'mAP@50-95 | {metrics.box.map:.4f}')
    print(f'precision | {metrics.box.mp:.4f}')
    print(f'recall | {metrics.box.mr:.4f}')

    print(f'save plots at: {metrics.save_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights-path', required=True, type = str)
    parser.add_argument('--data-yaml', required = True, type = str)
    
    args = parser.parse_args()

    evaluation(weights_path=args.weights_path, data_yaml=args.data_yaml, batch = 4, split = 'test')

