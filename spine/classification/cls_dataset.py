import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class SpinexrDataset(Dataset):
    def __init__(self, csv, image_dir, transform = None):
        if isinstance(csv, pd.DataFrame):
            self.df = csv.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv)
            
        self.image_dir = Path(image_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = self.image_dir/f'{row.image_id}.png'

        image = Image.open(img_path).convert('RGB')
        label = int(row.label)

        if self.transform:
            image = self.transform(image)

        return image, label