import torch
import pandas as pd
import torch.nn.functional as F
from torchvision.models import densenet201
from spine.classification.data_loader import get_transform
from PIL import Image
import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from ultralytics import YOLO
import argparse

CLASS_NAMES = {
    0: 'Osteophytes',
    1: 'Spondylolysthesis',
    2: 'Disc space narrowing',
    3: 'Vertebral collapse',
    4: 'Foraminal stenosis',
    5: 'Surgical implant',
    6: 'Other lesions'
}

CLASS_COLORS = {
    'Osteophytes': (0, 255, 0),
    'Spondylolysthesis': (255, 0, 255),
    'Disc space narrowing': (0, 0, 255),
    'Vertebral collapse': (255, 165, 0),
    'Foraminal stenosis': (0, 255, 255),
    'Surgical implant': (255, 0, 0),
    'Other lesions': (128, 0, 128)
}

class Classifier:
    def __init__(self, weight_path, img_size = 224, device = None):
        self.model = densenet201(weights=None)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.classifier = torch.nn.Linear(
            self.model.classifier.in_features,1
        )
        self.model.to(self.device)
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        
        self.model.eval()
        self.tf = get_transform(img_size=img_size, augment=False)

    def infer_one(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.tf(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(image).squeeze(1)
            prob = torch.sigmoid(logit).item()
        return prob

class Detector:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def infer_one(self, img_path, conf = 0.25, iou = 0.6):
        results = self.model.predict(
            source=img_path,
            conf = conf,
            iou = iou,
            verbose = False
        )[0]

        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                detections.append({
                    'bbox':box.xyxy.cpu().numpy()[0].tolist(),
                    'conf':float(box.conf),
                    'cls':int(box.cls)
                })
        return detections
    
class GTAnnotator:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.groups = self.df.groupby('image_id')

    def get_boxes(self, image_id):
        if image_id not in self.groups.groups:
            return []

        rows = self.groups.get_group(image_id)
        gt_boxes = []

        for _, r in rows.iterrows():
            if str(r.lesion_type).lower() == 'no finding':
                continue

            if pd.isna(r.xmin):
                continue

            gt_boxes.append({
                'bbox': [
                    int(r.xmin),
                    int(r.ymin),
                    int(r.xmax),
                    int(r.ymax)
                ],
                'cls_name': r.lesion_type
            })

        return gt_boxes

def draw_gt_boxes(image, gt_boxes):
    img = image.copy()

    for g in gt_boxes:
        x1, y1, x2, y2 = map(int, g['bbox'])
        label = g['cls_name']
        color = CLASS_COLORS.get(label, (255, 255, 255))

        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f'{label}'
        (w, h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(img, (x1, y1 - h - 8), (x1 + w + 6, y1), (0, 0, 0), -1)
        cv.putText(
            img, text,
            (x1 + 3, y1 - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )
    return img
    
def draw_boxes(image, detections):
    img = image.copy()

    for d in detections:
        x1, y1, x2, y2 = map(int ,d['bbox'])
        conf = d['conf']
        cls_id = d['cls']

        label = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
        color = CLASS_COLORS.get(label, (255, 255 ,255))

        # bbox

        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # text
        text = f'{label} {conf:.2f}'
        (w,h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(img, (x1, y1 - h - 8), (x1 + w + 6, y1), (0, 0, 0), -1)
        cv.putText(
            img, text,
            (x1 + 3, y1 -4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )
    return img

def infer_one_image(img_path, classifier:Classifier, detector:Detector, gt_annotator: GTAnnotator):
    
    # load image
    image = cv.imread(img_path)
    image_id = os.path.splitext(os.path.basename(img_path))[0]

    # gt
    gt_boxes = gt_annotator.get_boxes(image_id)
    gt_img = draw_gt_boxes(image.copy(), gt_boxes) if len(gt_boxes) > 0 else image

    # classifier
    prob_abnormal = classifier.infer_one(img_path)

    # detector
    detections = detector.infer_one(img_path, conf = 0.001, iou = 0.6)

    # decision rule
    """
    Given an input image x, we denote p_hat(abnormal|x) as the classifier's output that reflects the probability of the image being abnomal.
    For any x with prediction p_hat(abnormal|x) >= c*, all all lesion detection results are retained.
    For the case p_hat(abnormal|x) < c*, only predicted bounding boxes with confidence higher than 0.5 are kept.
    """
    if prob_abnormal >= classifier.threshold:
        final_dets = detections
    else:
        final_dets = [d for d in detections if d['conf'] >= 0.5]

    # visualize
    pred_img = draw_boxes(image.copy(), final_dets)
    cv.putText(
        pred_img,
        f'p_abnormal={prob_abnormal:.3f}',
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # save

    return gt_img, pred_img, prob_abnormal, final_dets

def make_compare(ori, pred):
    h1, w1, _ = ori.shape
    h2, w2, _ = pred.shape

    h = max(h1, h2)

    canvas = np.zeros((h, w1 + w2, 3), dtype = np.uint8)

    canvas[:h1, :w1] = ori
    canvas[:h2, w1:w1 + w2] = pred

    return canvas

def infer_all(input_dir, output_dir, classifier, detector, gt_annotator):
    pred_dir = os.path.join(output_dir, 'predict')
    cmp_dir = os.path.join(output_dir, 'compare')

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(cmp_dir, exist_ok=True)

    pbar = tqdm(os.listdir(input_dir), unit='img')
    for fname in pbar:
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_dir, fname)

        gt_img, pred, p_abn, dets = infer_one_image(
            img_path,
            classifier,
            detector,
            gt_annotator
        )

        compare_img = make_compare(gt_img, pred)

        cv.imwrite(os.path.join(pred_dir, fname), pred)
        cv.imwrite(os.path.join(cmp_dir, fname), compare_img)

        pbar.set_description(
            f'{fname} | p_abnormal={p_abn:.3f} | boxes={len(dets)}'
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cls-path', required = True, type = str)
    parser.add_argument('--det-path', required = True, type =str)
    parser.add_argument('--input-dir', required = True, type = str)
    parser.add_argument('--output-dir', required = True, type = str)
    parser.add_argument('--gt-csv', required=True, type=str)

    args = parser.parse_args()

    clsifer= Classifier(args.cls_path)
    dter = Detector(args.det_path)
    gt = GTAnnotator(args.gt_csv)

    infer_all(args.input_dir, args.output_dir, clsifer, dter, gt)

