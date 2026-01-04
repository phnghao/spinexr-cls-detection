import torch
import numpy as np
from torch import nn
from torchvision.models import densenet201, DenseNet201_Weights
from spine.classification.data_loader import prepare_loader
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
import argparse

def getmodel():
    model = densenet201(weights = DenseNet201_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(
        model.classifier.in_features,1
    )
    return model

def get_trainer(model):
    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    return loss, optimizer

def youden_index_thr(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    ix = np.argmax(J)

    best_thr = thresholds[ix]
    best_J = J[ix]

    return best_thr, best_J

def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1 = f1_score(y_true, y_pred)

    return sensitivity, specificity, f1
        

def train_epochs(epoch, model, loader, loss_func, optimizer, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Train]')
    for i, (image, label) in pbar:
        images, labels= image.to(device), label.float().to(device)

        output = model(images).squeeze(1)
        loss = loss_func(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pbar.set_postfix({'loss': loss.item()})

    return running_loss/len(loader)


def val_epoch(epoch, model, loader, loss_func, device):
    model.eval()
    running_loss = 0.0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Val]')
        for i, (image, label) in pbar:
            images, labels= image.to(device), label.float().to(device)

            output = model(images).squeeze(1)
            loss = loss_func(output, labels)

            running_loss += loss.item()

            probs = torch.sigmoid(output)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    avg_loss = running_loss / len(loader)
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return avg_loss, all_probs, all_labels


def train(csv_file, img_dir, save_file, n_epochs = 500, batch_size = 4, num_workers =2, img_size=224):
    save_dir = os.path.dirname(save_file)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader = prepare_loader(
        csv_file=csv_file,
        img_dir=img_dir,
        val_split=0.15,
        batch_size =batch_size,
        num_workers=num_workers,
        img_size=img_size
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on : {device}')
    model = getmodel()
    model = model.to(device)

    loss_func, optimizer = get_trainer(model)

    best_auroc = 0.0

    for epoch in range(1, n_epochs+1):
        train_loss = train_epochs(
            epoch, model, train_loader, loss_func, optimizer, device
        )

        val_loss, val_probs, val_labels = val_epoch(
            epoch, model, val_loader, loss_func, device
        )

        # AUROC
        val_auroc = roc_auc_score(val_labels, val_probs)

        # Youden's Index
        best_c, best_J = youden_index_thr(val_labels, val_probs)

        # metrics
        val_sens, val_spec, val_f1 = compute_metrics(val_labels, val_probs, best_c)

        print(
            f'Epoch {epoch}/{n_epochs} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'AUROC: {val_auroc:.4f} | '
            f'Sens: {val_sens:.4f} | '
            f'Spec: {val_spec:.4f} | '
            f'F1: {val_f1:.4f} | '
            f'c*: {best_c:.4f} | '
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': best_c,
                'val_auroc': val_auroc,
                'sensitivity': val_sens,
                'specificity': val_spec,
                'f1_score': val_f1,
                'youden_J': best_J
            }, save_file)
            print(f'saved AUROC: {val_auroc:.4f}')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv-file', required=True, type = str)
    parser.add_argument('--image-dir', required=True, type = str)
    parser.add_argument('--save-file', required=True, type = str)
    parser.add_argument('--nepochs', type = int, default=500)
    parser.add_argument('--batch-size', type = int, default=4)
    parser.add_argument('--num-workers', type =int, default=4)
    parser.add_argument('--image-size', type = int, default=224)

    args = parser.parse_args()

    train(
        csv_file = args.csv_file,
        img_dir=args.image_dir,
        save_file = args.save_file,
        n_epochs=args.nepochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.image_size
    )

if __name__ =='__main__':
    main()