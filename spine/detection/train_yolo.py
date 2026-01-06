from ultralytics import YOLO
import torch
import argparse

def train_yolo(data_yaml, epochs = 100, batch =4, imgsz = 640, device='0'):
    model = YOLO('yolov8m.pt')
    print(f'start training on deivce: {device}')

    results = model.train(
        data = data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        scale=0.2,
        batch=batch,
        optimizer='SGD',
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        device=device,
        workers=4,
        mosaic=0.0,
        save=True,
        val=True,
        plots=True
    )
    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml-file', required = True, type = str)
    parser.add_argument('--epochs', type = int, default=100)
    parser.add_argument('--batch-size', type = int, default=4)
    parser.add_argument('--imgsz', type = int, default=640)
    parser.add_argument('--device', type = str, default= '0')

    args = parser.parse_args()

    if args.device == '0' and not torch.cuda.is_available():
        args.device = 'cpu'

    train_yolo(args.yaml_file, epochs=args.epochs, batch = args.batch_size, imgsz = args.imgsz, device=args.device)

if __name__ == '__main__':
    main()