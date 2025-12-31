import pandas as pd
import os
import argparse

def merge_data(anno_file, meta_file, output_file):
    df_anno = pd.read_csv(anno_file)

    df_meta = pd.read_csv(meta_file)

    if 'image_id' not in df_anno.columns or 'image_id' not in df_meta.columns:
        raise ValueError('Both CSV files must have image_id columns ')
    
    df_merged = pd.merge(df_anno, df_meta, on = 'image_id', how = 'left')
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok = True)
    df_merged.to_csv(output_file, index = False)

    print(f"--- SUCCESS ---")
    print(f"Input records: {len(df_anno)}")
    print(f"Merged records: {len(df_merged)}")
    print(f"Saved to: {output_file}")
    
    print("\nPreview: 3 first rows")
    print(df_merged[['image_id', 'class_name', 'xmin', 'ymin', 'image_width', 'image_height']].head(3) 
          if 'class_name' in df_merged.columns else df_merged.head(3))
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--annotation-file', required = True, type = str)
    parser.add_argument('--meta-file', required = True, type = str)
    parser.add_argument('--output-file', required = True, type = str)

    args = parser.parse_args()

    merge_data(args.annotation_file, args.meta_file, args.output_file)

if __name__ == '__main__':
    main()