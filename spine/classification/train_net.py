import torch
from torch import nn
from torchvision.models import densenet201, DenseNet201_Weights
from spine.classification.data_loader import prepare_loader
from tqdm import tqdm
import os
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
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Val]')
        for i, (image, label) in pbar:
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = loss_func(output, label)

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100*correct / total
    return avg_loss, acc


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on : {device}')
    model = getmodel()
    model = model.to(device)

    loss_func, optimizer = get_trainer(model)
    best_val_acc = 0.0
    for epoch in range(1, n_epochs+1):
        train_loss, train_acc = train_epochs(
            epoch, model, train_loader, loss_func, optimizer, device
        )

        val_loss, val_acc = val_epoch(
            epoch, model, val_loader, loss_func, device
        )

        print(
            f"Epoch {epoch}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_file)
            print(f"Saved best model (Val Acc = {best_val_acc:.2f}%)")

        
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