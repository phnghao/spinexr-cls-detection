import pandas as pd
import os
from joblib import Parallel, delayed
import argparse
from tqdm import tqdm

def convert_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return cx, cy, bw, bh

def get_classes():
    return {
        'Osteophytes': 0, 
        'Spondylolysthesis': 1, 
        'Disc space narrowing': 2, 
        'Vertebral collapse': 3, 
        'Foraminal stenosis': 4,
        'Surgical implant': 5, 
        'Other lesions': 6
    }

def process_img_csv(image_id, rows, meta, out_dir, classes_map):
    if image_id not in meta:
        return

    W = meta[image_id]['image_width']
    H = meta[image_id]['image_height']

    if W <= 0 or H <= 0:
        return

    yolo_lines = []

    for _, r in rows.iterrows():
        if pd.isna(r.xmin):
            continue

        cls = classes_map.get(r.lesion_type, None)
        if cls is None:
            continue

        xmin = max(0, min(r.xmin, W - 1))
        ymin = max(0, min(r.ymin, H - 1))
        xmax = max(0, min(r.xmax, W - 1))
        ymax = max(0, min(r.ymax, H - 1))

        if xmax <= xmin or ymax <= ymin:
            continue

        cx, cy, w, h = convert_bbox(
            xmin, ymin, xmax, ymax, W, H
        )

        if w <= 0 or h <= 0:
            continue

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w  = min(max(w, 0.0), 1.0)
        h  = min(max(h, 0.0), 1.0)

        yolo_lines.append(
            f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}'
        )

    out_path = os.path.join(out_dir, f'{image_id}.txt')
    with open(out_path, 'w') as f:
        if yolo_lines:
            unique_lines = list(set(yolo_lines)) 
            f.write('\n'.join(unique_lines))
        else:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno-file', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--cpus', default=2, type=int)
    
    args = parser.parse_args()

    ann_file = args.anno_file
    out_dir = args.output_dir
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f'Reading CSV: {ann_file}')
    df = pd.read_csv(ann_file)
    
    meta = df.groupby('image_id').first()[['image_width', 'image_height']].to_dict('index')

    groups = list(df.groupby('image_id'))
    cls_map = get_classes()

    print(f'Processing {len(groups)} images')
    
    Parallel(n_jobs=args.cpus, backend='multiprocessing')(
        delayed(process_img_csv)(image_id, rows, meta, out_dir, cls_map)
        for image_id, rows in tqdm(groups)
    )


if __name__ == '__main__':
    main()