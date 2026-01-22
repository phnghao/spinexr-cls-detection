import os
import json
import cv2
import random
import argparse
import numpy as np

CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (255, 255, 0),
    4: (0, 200, 255),
    5: (255, 0, 255),
    6: (200, 200, 200)
}

CLASS_NAMES = {
    0: "Gai xuong",
    1: "Truot dot song",
    2: "Hep khe dia dem",
    3: "Sup than dot song",
    4: "Hep lo lien hop",
    5: "Cay ghep vat lieu",
    6: "Ton thuong khac"
}

NMS_THRESH = 0.1

def apply_nms(boxes):
    if not boxes:
        return []
    bxywh = []
    scores = []
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["bbox"])
        bxywh.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(b.get("score", 1.0)))
    idxs = cv2.dnn.NMSBoxes(bxywh, scores, 0.0, NMS_THRESH)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten()
    return [boxes[i] for i in idxs]

def draw_boxes(img, boxes):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["bbox"])
        cid = b["category_id"]
        score = b.get("score", None)
        name = CLASS_NAMES.get(cid, "Unknown")
        color = CLASS_COLORS.get(cid, (255, 255, 255))
        label = name if score is None else f"{name} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

def main(args):
    with open(args.vis_json, "r", encoding="utf-8") as f:
        vis_data = json.load(f)
    with open(args.pred_json, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    out_pred = os.path.join(args.out_dir, "pred_comparison")
    out_vis = os.path.join(args.out_dir, "vis_comparison")
    os.makedirs(out_pred, exist_ok=True)
    os.makedirs(out_vis, exist_ok=True)

    indices = list(range(len(vis_data)))
    if args.shuffle:
        random.shuffle(indices)
    indices = indices[:args.num_samples]

    for i in indices:
        v = vis_data[i]
        p = pred_data[i]

        img_path = v.get("file_name") or v.get("image_path")
        if img_path is None or not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_id = v.get("image_id", i)
        fname = f"{img_id}.png"

        gt = p.get("ground_truth", [])
        pred = apply_nms(p.get("predictions", []))

        left_gt = draw_boxes(img.copy(), gt)
        cv2.putText(left_gt, "Ground Truth", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        right_pred = draw_boxes(img.copy(), pred)
        cv2.putText(right_pred, "Prediction", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(out_pred, fname),
                    cv2.hconcat([left_gt, right_pred]))

        input_img = img.copy()
        cv2.putText(input_img, "Input", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        output_img = draw_boxes(img.copy(), pred)
        cv2.putText(output_img, "Output", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imwrite(os.path.join(out_vis, fname),
                    cv2.hconcat([input_img, output_img]))

    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    main(args)
