import torch
import numpy as np
from torch import nn
from torchvision.models import densenet201, DenseNet201_Weights
from spine.classification.data_loader import prepare_loader
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
import argparse

def getmodel():
    model = densenet201(weights = DenseNet201_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(
        model.classifier.in_features,2
    )
    return model

def get_trainer(model):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    return loss, optimizer

def youden_index_thr(y_true, y_prob, n_thresholds = 1001):
    thresholds = np.linspace(0, 1, n_thresholds)

    best_J = -1
    best_thr = 0.5

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        J = sensitivity + specificity -1
        if J > best_J:
            best_J = J
            best_thr = thr
    return best_thr, best_J
        

def train_epochs(epoch, model, loader, loss_func, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0 
    total = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Train]')
    for i, (image, label) in pbar:
        image, label = image.to(device), label.to(device)

        output = model(image)
        loss = loss_func(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(loader)
    acc = 100*correct / total
    return avg_loss, acc


def val_epoch(epoch, model, loader, loss_func, device):
    model.eval()
    running_loss = 0.0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Val]')
        for i, (image, label) in pbar:
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = loss_func(output, label)

            running_loss += loss.item()

            probs = torch.softmax(output, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(label.cpu())

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

    best_val_J = -1
    for epoch in range(1, n_epochs+1):
        train_loss, train_acc = train_epochs(
            epoch, model, train_loader, loss_func, optimizer, device
        )

        val_loss, val_probs, val_labels = val_epoch(
            epoch, model, val_loader, loss_func, device
        )

        best_c, best_J = youden_index_thr(val_labels, val_probs)
        
        print(
            f'Epoch {epoch}/{n_epochs} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'Best c*: {best_c:.4f} | '
            f'Youden J: {best_J:.4f}'
        )
        if best_J > best_val_J:
            best_val_J = best_J
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': best_c
            }, save_file)

            print(f'saved best model | J = {best_val_J:.4f}, c* = {best_c:.4f}')



        
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