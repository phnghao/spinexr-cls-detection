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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

def get_model():
    model = densenet201(weights = None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = f1_score(y_true, y_pred)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return sensitivity, specificity, f1

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
    print(f'Evaluating model DenseNet201 on test set')

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            logits = model(images).squeeze(1)

            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auroc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs >= c_star).astype(int)
    sens, spec, f1 = compute_metrics(all_labels, preds)

    print('results')
    print(f'AUROC        : {auroc:.4f}')
    print(f'Sensitivity  : {sens:.4f}')
    print(f'Specificity  : {spec:.4f}')
    print(f'F1-score     : {f1:.4f}')
    print(f'Used threshold (from val): {c_star:.4f}')

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
