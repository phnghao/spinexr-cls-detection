import torch
import numpy as np
import pandas as pd
from .cls_dataset import SpinexrDataset
from spine.augments.autoaugment_utils import distort_image_with_autoaugment
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class Auto_augments:
    def __call__(self, img):
        img = np.array(img)
        augmented_img, _ = distort_image_with_autoaugment(img, np.zeros((0,4)), 'v0')

        if not isinstance(augmented_img, Image.Image):
            augmented_img = Image.fromarray(augmented_img)
        return augmented_img

def get_transform(img_size = 224, augment = True):
    ops = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
    ]
    if augment:
        ops.append(Auto_augments())
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
    ])
    return transforms.Compose(ops)

def prepare_loader(csv_file, img_dir, val_split = 0,  batch_size = 16, num_workers = 2, img_size = 224):
    """
    val_split : ratio that spliting loader into 2 part include train and val
    va_split = 0.0 i.e use all train 
    """

    if val_split > 0:
        df = pd.read_csv(csv_file)
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df, test_size=val_split, stratify=df['label'],
            random_state=42
        )

        train_tf = get_transform(img_size, augment=True)
        val_tf = get_transform(img_size, augment=False)

        train_ds = SpinexrDataset(train_df, img_dir, transform=train_tf)
        val_ds = SpinexrDataset(val_df, img_dir, transform=val_tf)
            
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(f'train set: {len(train_ds)}')
        print(f'val set: {len(val_ds)}')

        return train_loader, val_loader
    else:
        tf = get_transform(img_size, augment=False)
        dataset = SpinexrDataset(
            csv = csv_file,
            image_dir=img_dir,
            transform=tf
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle = False, num_workers=num_workers)
        return loader


    

