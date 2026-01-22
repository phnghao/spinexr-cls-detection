import os
import cv2 as cv
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
from torchvision.models import densenet201
import albumentations as A
from albumentations.pytorch import ToTensorV2

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

from spine.sparsercnn.config import add_sparsercnn_config
from spine.sparsercnn.register_dataset import register_spine_datasets


CLASS_NAME_VI = {
    'Osteophytes': 'Gai xương',
    'Spondylolysthesis': 'Trượt đốt sống',
    'Disc space narrowing': 'Hẹp khe đĩa đệm',
    'Vertebral collapse': 'Sụp thân đốt sống',
    'Foraminal stenosis': 'Hẹp lỗ liên hợp',
    'Surgical implant': 'Cấy ghép vật liệu phẫu thuật',
    'Other lesions': 'Tổn thương khác'
}


def instances2json(instances, metadata):
    results = []
    if len(instances) == 0:
        return results
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()
    class_names = metadata.thing_classes
    for b, s, c in zip(boxes, scores, classes):
        class_en = class_names[c]
        results.append({
            'bbox': [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
            'score': float(s),
            'category_id': int(c),
            'category_name_en': class_en,
            'category_name_vi': CLASS_NAME_VI.get(class_en, class_en)
        })
    return results


def gt2json(d, metadata):
    gts = []
    class_names = metadata.thing_classes
    for ann in d.get('annotations', []):
        x, y, w, h = ann['bbox']
        class_en = class_names[ann['category_id']]
        gts.append({
            'bbox': [x, y, x + w, y + h],
            'category_id': ann['category_id'],
            'category_name_en': class_en,
            'category_name_vi': CLASS_NAME_VI.get(class_en, class_en)
        })
    return gts


class Predictor:
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cls_model = densenet201(weights=None)
        self.cls_model.classifier = nn.Linear(self.cls_model.classifier.in_features, 1)
        ckpt = torch.load(args.cls_model, map_location=self.device)
        self.cls_model.load_state_dict(ckpt['model_state_dict'])
        self.c_star = ckpt.get('threshold', 0.5)
        self.cls_model.to(self.device).eval()
        self.cls_tf = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
        self.cfg = get_cfg()
        add_sparsercnn_config(self.cfg)
        self.cfg.merge_from_file(args.config_file)
        self.cfg.MODEL.WEIGHTS = args.det_model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        if self.device == 'cpu':
            self.cfg.MODEL.DEVICE = 'cpu'
        self.det_model = build_model(self.cfg)
        self.det_model.eval()
        DetectionCheckpointer(self.det_model).load(args.det_model)

    def predict(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x = self.cls_tf(image=img_rgb)['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            p_abn = torch.sigmoid(self.cls_model(x)).item()
        with torch.no_grad():
            inp = torch.as_tensor(img.astype('float32').transpose(2, 0, 1))
            if self.device != 'cpu':
                inp = inp.cuda()
            outputs = self.det_model([{
                'image': inp,
                'height': img.shape[0],
                'width': img.shape[1]
            }])[0]
            instances = outputs['instances']
        if p_abn < self.c_star:
            instances = instances[instances.scores > 0.5]
        return instances, p_abn


def main(args):
    register_spine_datasets()
    dataset = DatasetCatalog.get('spine_test')
    metadata = MetadataCatalog.get('spine_test')
    predictor = Predictor(args)
    visualize_results = []
    predicted_results = []
    os.makedirs(args.out_dir, exist_ok=True)
    for d in tqdm(dataset):
        img = cv.imread(d['file_name'])
        if img is None:
            continue
        instances, p_abn = predictor.predict(img)
        preds = instances2json(instances, metadata)
        visualize_results.append({
            'image_path': d['file_name'],
            'height': img.shape[0],
            'width': img.shape[1],
            'classification_score': p_abn,
            'predictions': preds
        })
        predicted_results.append({
            'image_path': d['file_name'],
            'height': img.shape[0],
            'width': img.shape[1],
            'classification_score': p_abn,
            'ground_truth': gt2json(d, metadata),
            'predictions': preds
        })
    with open(os.path.join(args.out_dir, 'visualize_results.json'), 'w', encoding='utf-8') as f:
        json.dump(visualize_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, 'predicted_results.json'), 'w', encoding='utf-8') as f:
        json.dump(predicted_results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--det-model', required=True)
    parser.add_argument('--cls-model', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    setup_logger()
    main(args)
