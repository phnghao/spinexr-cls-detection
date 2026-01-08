from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

CLASS_NAMES = [
    'Osteophytes',
    'Spondylolysthesis',
    'Disc space narrowing',
    'Vertebral collapse',
    'Foraminal stenosis',
    'Surgical implant',
    'Other lesions'
]

def register_spine_datasets():
    register_coco_instances(
        name="spine_train",
        metadata={},
        json_file="./data/annotations/train.json",
        image_root="./data/train_pngs"
    )

    MetadataCatalog.get("spine_train").thing_classes = CLASS_NAMES

    register_coco_instances(
        name="spine_test",     
        metadata={},
        json_file="./data/annotations/test.json",
        image_root="./data/test_pngs"
    )

    MetadataCatalog.get("spine_test").thing_classes = CLASS_NAMES


