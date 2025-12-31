import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def split_csv_image(csv_file, train_output, val_output, val_size=0.2, seed=42):
    print(f"Reading {csv_file}")
    df = pd.read_csv(csv_file)

    # mỗi ảnh lấy lesion_type đầu tiên để stratify
    img_labels = (
        df.groupby("image_id")["lesion_type"]
        .first()
        .reset_index()
    )

    print(f"Total unique images: {len(img_labels)}")

    train_imgs, val_imgs = train_test_split(
        img_labels["image_id"],
        test_size=val_size,
        random_state=seed,
        stratify=img_labels["lesion_type"]
    )

    train_df = df[df["image_id"].isin(train_imgs)]
    val_df   = df[df["image_id"].isin(val_imgs)]

    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(val_output), exist_ok=True)

    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)

    print(f"Train set: {len(train_imgs)} images | {len(train_df)} annotations")
    print(f"Val set  : {len(val_imgs)} images | {len(val_df)} annotations")

    print("\nTrain lesion distribution:")
    print(train_df["lesion_type"].value_counts())

    print("\nVal lesion distribution:")
    print(val_df["lesion_type"].value_counts())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True, type=str)
    parser.add_argument("--train-output", required=True, type=str)
    parser.add_argument("--val-output", required=True, type=str)
    parser.add_argument("--val-size", type=float, default=0.2)

    args = parser.parse_args()

    split_csv_image(
        csv_file=args.csv_file,
        train_output=args.train_output,
        val_output=args.val_output,
        val_size=args.val_size
    )

if __name__ == "__main__":
    main()
