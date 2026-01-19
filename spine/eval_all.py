import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import densenet201
from spine.classification.cls_dataset import SpinexrDataset
from spine.classification.data_loader import get_transform
import pandas as pd
import numpy as np
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import argparse
import pprint
import os
from PIL import Image

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

from spine.sparsercnn.config import add_sparsercnn_config
from spine.sparsercnn.detector import SparseRCNN
from spine.sparsercnn.register_dataset import register_spine_datasets

class ClassificationEvaluator:
    def __init__(self, csv_file, img_dir, model_path, output_dir, batch_size=4, num_workers=4, img_size=224):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.model_path = model_path
        self.output_dir = output_dir
        self.batch = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_model(self):
        model = densenet201(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model

    def compute_metrics(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        f1 = f1_score(y_true, y_pred)
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        return sensitivity, specificity, f1
    
    def get_ci(self, values):
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        mean = np.mean(values)
        return mean, lower, upper
    
    def boostrap_metric(self, y_true, y_prob, threshold, n_booststraps=1000):
        print(f'running bootstrap {n_booststraps} times with threshold = {threshold:.4f}')
        sens_list, spec_list, f1_list, auc_list = [], [], [], []
        rng = np.random.RandomState(42)
        n_samples = len(y_true)

        for _ in range(n_booststraps):
            indices = rng.randint(0, n_samples, n_samples)
            if len(np.unique(y_true[indices])) < 2: continue

            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]
            
            y_pred_boot = (y_prob_boot >= threshold).astype(int)

            s, sp, f = self.compute_metrics(y_true_boot, y_pred_boot)
            auc = roc_auc_score(y_true_boot, y_prob_boot)

            sens_list.append(s)
            spec_list.append(sp)
            f1_list.append(f)
            auc_list.append(auc)

        return {
            'AUROC': self.get_ci(auc_list),
            'Sensitivity': self.get_ci(sens_list),
            'Specificity': self.get_ci(spec_list),
            'F1': self.get_ci(f1_list)
        }

    def save_results(self, metrics_dict, filename="cls_metrics.csv"):
        save_path = os.path.join(self.output_dir, filename)
        
        rows = []
        print(f"\n{'Metric':<15} | {'Mean':<10} | {'95% CI (Lower - Upper)':<25}")
        for k, v in metrics_dict.items():
            mean, low, high = v
            print(f"{k:<15} | {mean*100:.2f}%     | ({low*100:.2f} - {high*100:.2f})")
            
            rows.append({
                "Metric": k,
                "Mean": mean,
                "Lower_CI": low,
                "Upper_CI": high,
                "Format_Mean_Percentage": f"{mean*100:.2f}%",
                "Format_CI_Percentage": f"({low*100:.2f} - {high*100:.2f})"
            })
            
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"saved cls metrics to: {save_path}")

    def evaluate(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'evaluating on: {device}')
        print(f'loaded test csv data: {self.csv_file}')

        test_tf = get_transform(img_size=self.img_size, augment=False)
        test_ds = SpinexrDataset(self.csv_file, self.img_dir, transform=test_tf)
        test_loader = DataLoader(test_ds, batch_size=self.batch, shuffle=False, num_workers=self.num_workers)
        print(f'test set size: {len(test_ds)} images')

        model = self.get_model().to(device)
        print(f'loaded weight from: {self.model_path}')
        checkpoint = torch.load(self.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        op_thr = checkpoint['threshold']
        print(f'optimal threshold: {op_thr}')
        model.eval()

        all_probs = []
        all_labels = []
        print(f'evaluating model DenseNet201 on test set')

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='classification eval'):
                images = images.to(device)
                logits = model(images).squeeze(1)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_labels.append(labels)

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        result = self.boostrap_metric(all_labels, all_probs, op_thr)
        self.save_results(result)

class DetectionEvaluator:
    def __init__(self, args):
        self.args = args

    def setup(self, args):
        cfg = get_cfg()
        add_sparsercnn_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)

        if not args.opts or not any("MODEL.WEIGHTS" in s for s in args.opts):
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

        cfg.freeze()
        return cfg
    
    def save_summary(self, results, output_dir):
        save_path = os.path.join(output_dir, "detection_summary.txt")
        with open(save_path, "w") as f:
            f.write("evaluated results\n")
            f.write("="*50 + "\n")
            f.write(pprint.pformat(results))
        print(f"saved to: {save_path}")
    
    def evaluate(self):
        args = self.args
        register_spine_datasets()
        cfg = self.setup(args)

        print(f'loaded model from: {cfg.MODEL.WEIGHTS}')
        model = build_model(cfg)
        model.eval()

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        output_folder = os.path.join(cfg.OUTPUT_DIR, 'detection_eval_results')
        os.makedirs(output_folder, exist_ok=True)
        evaluator = COCOEvaluator("spine_test", cfg, False, output_dir=output_folder)

        val_loader = build_detection_test_loader(cfg, "spine_test")
        print(f'evaluating detection')
        results = inference_on_dataset(model, val_loader, evaluator)

        print("\n" + "="*50)
        print("detection result:")
        pprint.pprint(results)
        print("="*50)
        self.save_summary(results, output_folder)
        return results

class FusionEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cls_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_cls_model(self):
        model = densenet201(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        model.to(self.device)

        print(f'loading classifier from: {self.args.cls_model}')
        checkpoint = torch.load(self.args.cls_model, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        c_star = checkpoint['threshold']
        model.eval()
        return model, c_star
    
    def setup_detector(self):
        cfg = get_cfg()
        add_sparsercnn_config(cfg)
        cfg.merge_from_file(self.args.config_file)
        cfg.merge_from_list(self.args.opts)
        if not self.args.opts or not any("MODEL.WEIGHTS" in s for s in self.args.opts):
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
        print(f'loading detector from: {cfg.MODEL.WEIGHTS}')
        cfg.freeze()
        return cfg
    
    def evaluate(self):
        register_spine_datasets()

        # Load Classifier
        cls_model, c_star = self.load_cls_model()
        print(f'optimal classification threshold: {c_star:.4f}')

        # Load Detector
        det_cfg = self.setup_detector()
        det_model = build_model(det_cfg)
        det_model.eval()
        det_model.to(self.device)

        checkpointer = DetectionCheckpointer(det_model)
        checkpointer.load(det_cfg.MODEL.WEIGHTS)

        # Build DataLoader và Evaluator
        val_loader = build_detection_test_loader(det_cfg, "spine_test")
        output_folder = os.path.join(det_cfg.OUTPUT_DIR, "fusion_eval_results")
        os.makedirs(output_folder, exist_ok=True)

        evaluator = COCOEvaluator(
            "spine_test", det_cfg, False, output_dir=output_folder
        )
        evaluator.reset()

        print("running fusion framework evaluation")

        with torch.no_grad():
            for inputs in tqdm(val_loader, desc="fusion eval"):
                det_outputs = det_model(inputs)
                processed_results = []

                for i, input_data in enumerate(inputs):
                    img = Image.open(input_data["file_name"]).convert("RGB")
                    cls_input = self.cls_transform(img).unsqueeze(0).to(self.device)

                    logits = cls_model(cls_input).squeeze(1)
                    p_abnormal = torch.sigmoid(logits).item()

                    instances = det_outputs[i]["instances"]

                    if p_abnormal < c_star:
                        keep = instances.scores >= 0.5
                        instances = instances[keep]

                    processed_results.append({"instances": instances})

                evaluator.process(inputs, processed_results)

        results = evaluator.evaluate()
        pprint.pprint(results)

        save_path = os.path.join(output_folder, "fusion_summary.txt")
        with open(save_path, "w") as f:
            f.write(f"Fusion evaluation (c* = {c_star:.4f})\n")
            f.write(pprint.pformat(results))

        print("fusion evaluation completed")

def main(args):
    # Task: Classification
    if args.eval_task in ['cls', 'all']:
        print('\n--- Running Classification Evaluation ---')
        if not args.cls_model or not args.cls_csv or not args.img_dir:
            print('Warning: Missing parameters for classification eval. Skipping...')
        else:
            cls_evaluator = ClassificationEvaluator(
                csv_file=args.cls_csv,
                img_dir=args.img_dir,
                model_path=args.cls_model,
                output_dir=args.cls_output,
                batch_size=8,
                img_size=224
            )
            cls_evaluator.evaluate()
    
    # Task: Detection
    if args.eval_task in ["det", "all"]:
        print('\n--- Running Detection Evaluation ---')
        det_evaluator = DetectionEvaluator(args)
        det_evaluator.evaluate()

    # Task: Fusion Framework
    if args.eval_task in ["fusion", "all"]:
        print('\n--- Running Fusion Framework Evaluation ---')
        if not args.cls_model:
            print("Error: --cls-model is required for fusion evaluation.")
            return
            
        fusion_evaluator = FusionEvaluator(args)
        fusion_evaluator.evaluate()

if __name__ == "__main__":
    parser = default_argument_parser()

    parser.add_argument("--eval-task", default="all", choices=["cls", "det", "fusion", "all"], help="Choose evaluation task")
    
    parser.add_argument("--cls-csv", default="./data/classification/test.csv", help="Path to classification test csv")
    parser.add_argument("--img-dir", default="./data/test_images", help="Path to test images folder")
    parser.add_argument("--cls-model", default=None, help="Path to trained classification model (.pth)")
    parser.add_argument("--cls-output", default="./outputs/cls_eval", help="Folder to save classification metrics")

    args = parser.parse_args()

    setup_logger()
    
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )