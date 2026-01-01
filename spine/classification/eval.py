import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import densenet201
from spine.classification.cls_dataset import SpinexrDataset
from spine.classification.data_loader import get_transform
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score

def get_model():
    model = densenet201(weights = None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        'accuracy': acc,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def evaluate(csv_file, img_dir, model_path, batch_size=4, num_workers = 2, img_size = 224):

    # Prepare data 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'evaluating on: {device}')

    print(f'load test csv data: {csv_file}')

    test_tf = get_transform(img_size, augment = False)
    test_ds = SpinexrDataset(csv_file, img_dir, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle = False, num_workers=num_workers)
    print(f'test set size: {len(test_ds)} images')

    # Load model
    model = get_model().to(device)

    print(f'loading weights from: {model_path}')
    checkpoint = torch.load(model_path, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    c_star = checkpoint['threshold']
    model.eval()

    all_probs = []
    all_labels = []
    print('Start Inference')

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:,1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auroc = roc_auc_score(all_labels, all_probs)
    print(f'\nAUROC: {auroc:.4f}')

    print(f'uising optimal threshold from validation {c_star:.4f}')
    preds = (all_probs >= c_star).astype(int)

    metrics = compute_metrics(all_labels, preds)
    print("\nFinal metrics (fixed threshold from validation):")
    print(f"Accuracy    : {metrics['accuracy']:.4f}")
    print(f"F1-score    : {metrics['f1']:.4f}")
    print(f"Sensitivity : {metrics['sensitivity']:.4f}")
    print(f"Specificity : {metrics['specificity']:.4f}")
    print('\nClassification report:')
    print(classification_report(all_labels, preds, digits=4))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv-file', required=True, type=str)
    parser.add_argument('--image-dir', required=True, type=str)
    parser.add_argument('--model-path', required=True, type=str)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=224)

    args = parser.parse_args()

    evaluate(
        csv_file=args.csv_file,
        img_dir=args.image_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.image_size
    )


if __name__ == '__main__':
    main()
