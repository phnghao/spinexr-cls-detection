import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
from torchvision.models import densenet201
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Detectron2
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger

# Project
from spine.sparsercnn.config import add_sparsercnn_config
from spine.sparsercnn.register_dataset import register_spine_datasets


class Predictor:
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"predictor on:{self.device}")

        #  classification
        self.cls_model = densenet201(weights=None)
        self.cls_model.classifier = nn.Linear(self.cls_model.classifier.in_features, 1)

        ckpt = torch.load(args.cls_model, map_location=self.device)
        self.cls_model.load_state_dict(ckpt['model_state_dict'])
        self.c_star = ckpt.get('threshold', 0.5)

        self.cls_model.to(self.device).eval()

        self.cls_tf = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        print(f'Classification threshold c*: {self.c_star:.4f}')

        # detection
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
        # classification
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.cls_tf(image=img_rgb)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            p_abn = torch.sigmoid(self.cls_model(x)).item()

        # detection
        with torch.no_grad():
            inp = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            if self.device != "cpu":
                inp = inp.cuda()

            outputs = self.det_model([{
                "image": inp,
                "height": img.shape[0],
                "width": img.shape[1]
            }])[0]

            instances = outputs["instances"]

        # fusion rule
        if p_abn >= self.c_star:
            final_instances = instances
            info = f'high risk (P={p_abn:.2f} >= {self.c_star:.2f})'
            color = (0, 0, 255)
        else:
            final_instances = instances[instances.scores > 0.5]
            info = f'low risk (P={p_abn:.2f} < {self.c_star:.2f})'
            color = (0, 255, 0)

        return final_instances, info, color


def main(args):
    register_spine_datasets()
    dataset = DatasetCatalog.get('spine_test')
    metadata = MetadataCatalog.get('spine_test')
    metadata.thing_colors = [
        (220, 20, 60),    # Osteophytes
        (119, 11, 32),    # Spondylolysthesis
        (0, 0, 142),      # Disc space narrowing
        (0, 0, 230),      # Vertebral collapse
        (106, 0, 228),    # Foraminal stenosis
        (0, 60, 100),     # Surgical implant
        (0, 80, 100),     # Other lesions
    ]

    out_dir = os.path.join(os.path.dirname(args.det_model), 'predict_viz')
    os.makedirs(out_dir, exist_ok=True)

    predictor = Predictor(args)

    if args.num_samples > 0:
        dataset = dataset[:args.num_samples]

    for d in tqdm(dataset, desc='inference'):
        img = cv2.imread(d['file_name'])
        if img is None:
            continue

        instances, info, color = predictor.predict(img)

        v_gt = Visualizer(img[:, :, ::-1], metadata)
        gt_img = v_gt.draw_dataset_dict(d).get_image()[:, :, ::-1]
        gt_img = np.ascontiguousarray(gt_img) 

        v_pr = Visualizer(img[:, :, ::-1], metadata, instance_mode=ColorMode.SEGMENTATION)
        pr_img = v_pr.draw_instance_predictions(instances.to('cpu')).get_image()[:, :, ::-1]
        pr_img = np.ascontiguousarray(pr_img)   


        cv2.putText(gt_img, 'GROUND TRUTH', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(pr_img, 'FUSION PREDICTION', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(pr_img, info, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        merged = cv2.hconcat([gt_img, pr_img])
        cv2.imwrite(os.path.join(out_dir, os.path.basename(d['file_name'])), merged)

    print('prediced completely')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True, type = str)
    parser.add_argument('--det-model', required=True, type = str)
    parser.add_argument('--cls-model', required=True, type = str)
    parser.add_argument('--num-samples', type=int, default=20)

    args, _ = parser.parse_known_args()
    setup_logger()
    main(args)
