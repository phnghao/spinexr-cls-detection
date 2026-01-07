import pandas as pd
import json
import os
import cv2 as cv
import argparse
from tqdm import tqdm

CLASS_NAMES = [
    'Osteophytes',
    'Spondylolysthesis',
    'Disc space narrowing',
    'Vertebral collapse',
    'Foraminal stenosis',
    'Surgical implant',
    'Other lesions'
]

CATEGORY_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

def convert_to_coco(csv_path, img_dir, output_path):
    print(f'Loading CSV: {csv_path}')
    df = pd.read_csv(csv_path)

    coco = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    # categories
    for name, cid in CATEGORY_MAP.items():
        coco['categories'].append({
            'id': cid,
            'name': name,
            'supercategory': 'lesion'
        })

    ann_id = 1
    img_id_counter = 1

    unique_images = df['image_id'].unique()
    print(f'Processing {len(unique_images)} images')

    for img_name in tqdm(unique_images):
        filename = f'{img_name}.png'
        img_path = os.path.join(img_dir, filename)

        if not os.path.exists(img_path):
            print(f'[WARNING] Missing image: {img_path}')
            continue

        img = cv.imread(img_path)
        if img is None:
            print(f'[WARNING] Cannot read image: {img_path}')
            continue

        h, w = img.shape[:2]

        coco['images'].append({
            'id': img_id_counter,
            'file_name': filename,
            'height': h,
            'width': w
        })

        rows = df[df['image_id'] == img_name]
        has_ann = False

        for _, row in rows.iterrows():
            label = row['lesion_type']

            if label not in CATEGORY_MAP:
                continue
            if pd.isna(row['xmin']):
                continue

            x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']]
            w_box = x2 - x1
            h_box = y2 - y1

            if w_box <= 0 or h_box <= 0:
                continue

            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id_counter,
                'category_id': CATEGORY_MAP[label],
                'bbox': [float(x1), float(y1), float(w_box), float(h_box)],
                'area': float(w_box * h_box),
                'iscrowd': 0,
                'segmentation': []
            })

            ann_id += 1
            has_ann = True

        img_id_counter += 1

    print(f'Saving COCO json to {output_path}')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(coco, f)

    print(f'Done: {len(coco["images"])} images, {len(coco["annotations"])} annotations')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()

    convert_to_coco(args.csv_path, args.img_dir, args.output_path)
