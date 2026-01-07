from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def register_spine_datasets():
    register_coco_instances(
        'spine_train',
        {},
        './data/annotations/train.json',
        './data/train_pngs'
    )