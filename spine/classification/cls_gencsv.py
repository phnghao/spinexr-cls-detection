import pandas as pd
import os
import argparse

def cls_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    df['has_bbox'] = df[['xmin', 'ymin', 'xmax', 'ymax']].notna().all(axis=1)

    cls_df = (
        df.groupby('image_id')['has_bbox'].any().reset_index()
    )

    cls_df['label'] = cls_df['has_bbox'].astype(int)
    cls_df = cls_df[['image_id','label']]
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cls_df.to_csv(output_csv, index=False)
    print(cls_df['label'].value_counts())
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-csv', required=True, type = str)
    parser.add_argument('--output-csv', required = True, type = str)

    args = parser.parse_args()

    cls_csv(args.input_csv, args.output_csv)

