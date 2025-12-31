import os
import pandas as pd
import shutil
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse
from pathlib import Path

def cpy_img(img_id, root_dir, out_dir):
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)

    src = root_dir / f"{img_id}.png"
    dst = out_dir / f"{img_id}.png"

    if src.exists():
        shutil.copy2(src, dst)
        return True
    else:
        return False


def process(csv_file, root_dir, out_dir, n_jobs=4):
    df = pd.read_csv(csv_file)
    image_ids = df['image_id'].unique()

    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(cpy_img)(img_id, root_dir, out_dir)
        for img_id in tqdm(image_ids)
    )

    missing = len(results) - sum(results)
    print(f"[{out_dir}] Missing images: {missing}/{len(results)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root-dir', required=True, type = str)
    parser.add_argument('--traincsv-file',required=True, type = str)
    parser.add_argument('--valcsv-file', required=True, type = str)
    parser.add_argument('--train-dir', required=True, type = str)
    parser.add_argument('--val-dir', required=True, type = str)
    parser.add_argument('--cpus', default=4, type = int)

    args = parser.parse_args()

    root_dir = args.root_dir
    train_dir = args.train_dir
    val_dir = args.val_dir

    traincsv_file = args.traincsv_file
    valcsv_file = args.valcsv_file

    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(val_dir, exist_ok= True)

    process(traincsv_file, root_dir, train_dir, n_jobs = args.cpus)
    process(valcsv_file, root_dir, val_dir, n_jobs = args.cpus)


