import os
import json
import cv2
import random
import argparse
import numpy as np
import torch

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes

CLASS_NAMES = {
    0: "Gai xuong",
    1: "Truot dot song",
    2: "Hep khe dia dem",
    3: "Sup than dot song",
    4: "Hep lo lien hop",
    5: "Cay ghep vat lieu",
    6: "Ton thuong khac"
}

CLASS_COLORS_BGR = {
    0: (0, 255, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (255, 255, 0),
    4: (0, 200, 255),
    5: (255, 0, 255),
    6: (200, 200, 200)
}

# IoU Threshold chuẩn: 0.5
# Nếu 2 box chồng nhau > 50% diện tích thì xóa box điểm thấp hơn
NMS_THRESH = 0.5

def setup_metadata():
    name = "spine_fixed_color"
    if name in MetadataCatalog:
        MetadataCatalog.remove(name)
    meta = MetadataCatalog.get(name)
    meta.thing_classes = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    meta.thing_colors = [
        (CLASS_COLORS_BGR[i][2], CLASS_COLORS_BGR[i][1], CLASS_COLORS_BGR[i][0])
        for i in range(len(CLASS_COLORS_BGR))
    ]
    return meta

def apply_nms(boxes):
    if not boxes:
        return []
    bxywh = []
    scores = []
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["bbox"])
        bxywh.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(b.get("score", 1.0)))
    
    # score_threshold=0.05: Chỉ lọc bỏ những box rác điểm quá thấp (< 5%)
    # nms_threshold=0.5: Mức tiêu chuẩn
    idxs = cv2.dnn.NMSBoxes(bxywh, scores, 0.05, NMS_THRESH)
    
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten()
    return [boxes[i] for i in idxs]

def to_instances(boxes, h, w):
    inst = Instances((h, w))
    if not boxes:
        inst.pred_boxes = Boxes(torch.empty((0, 4)))
        inst.pred_classes = torch.empty((0,), dtype=torch.int64)
        inst.scores = torch.empty((0,))
        return inst
    inst.pred_boxes = Boxes(torch.tensor([b["bbox"] for b in boxes], dtype=torch.float32))
    inst.pred_classes = torch.tensor([b["category_id"] for b in boxes], dtype=torch.int64)
    inst.scores = torch.tensor([b.get("score", 1.0) for b in boxes], dtype=torch.float32)
    return inst

def draw(img, instances, metadata, title):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    v = Visualizer(
        img_rgb,
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.SEGMENTATION
    )
    out = v.draw_instance_predictions(instances)
    res = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)
    cv2.putText(res, title, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return res

def main(args):
    with open(args.vis_json, "r", encoding="utf-8") as f:
        vis_data = json.load(f)
    with open(args.pred_json, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    out_pred = os.path.join(args.out_dir, "pred_comparison")
    out_vis = os.path.join(args.out_dir, "vis_comparison")
    os.makedirs(out_pred, exist_ok=True)
    os.makedirs(out_vis, exist_ok=True)

    metadata = setup_metadata()

    idxs = list(range(len(vis_data)))
    random.shuffle(idxs)
    idxs = idxs[:args.num_samples]

    for i in idxs:
        v = vis_data[i]
        p = pred_data[i]

        img_path = v.get("file_name") or v.get("image_path")
        if img_path is None or not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        img_id = v.get("image_id", i)
        fname = f"{img_id}.png"

        gt_boxes = p.get("ground_truth", [])
        pr_boxes = apply_nms(p.get("predictions", []))

        gt_inst = to_instances(gt_boxes, h, w)
        pr_inst = to_instances(pr_boxes, h, w)

        left = draw(img.copy(), gt_inst, metadata, "Ground Truth")
        right = draw(img.copy(), pr_inst, metadata, "Prediction")
        cv2.imwrite(os.path.join(out_pred, fname), cv2.hconcat([left, right]))

        inp = img.copy()
        cv2.putText(inp, "Input", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        outp = draw(img.copy(), pr_inst, metadata, "Output")
        cv2.imwrite(os.path.join(out_vis, fname), cv2.hconcat([inp, outp]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()
    main(args)