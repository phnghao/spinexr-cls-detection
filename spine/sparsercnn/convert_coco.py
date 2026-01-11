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
    df = pd.read_csv(csv_path)

    coco = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    for name, cid in CATEGORY_MAP.items():
        coco['categories'].append({
            'id': cid,
            'name': name,
            'supercategory': 'lesion'
        })

    ann_id = 1
    img_id = 1
    num_positive = 0
    num_negative = 0

    for image_name in tqdm(df['image_id'].unique()):
        filename = f'{image_name}.png'
        img_path = os.path.join(img_dir, filename)

        if not os.path.exists(img_path):
            continue

        img = cv.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        coco['images'].append({
            'id': img_id,
            'file_name': filename,
            'height': h,
            'width': w
        })

        rows = df[df['image_id'] == image_name]
        has_ann = False

        for _, r in rows.iterrows():
            label = r['lesion_type']
            if label not in CATEGORY_MAP:
                continue
            if pd.isna(r['xmin']):
                continue

            x1, y1, x2, y2 = r[['xmin', 'ymin', 'xmax', 'ymax']]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': CATEGORY_MAP[label],
                'bbox': [float(x1), float(y1), float(bw), float(bh)],
                'area': float(bw * bh),
                'iscrowd': 0,
                'segmentation': [[]]
            })

            ann_id += 1
            has_ann = True

        if has_ann:
            num_positive += 1
        else:
            num_negative += 1

        img_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f)

    print(f'Images: {len(coco["images"])}')
    print(f'Annotations: {len(coco["annotations"])}')
    print(f'Positive images: {num_positive}')
    print(f'Negative images: {num_negative}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()

    convert_to_coco(args.csv_path, args.img_dir, args.output_path)
