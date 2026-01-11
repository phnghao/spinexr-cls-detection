import os
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

import spine.sparsercnn.detector
from spine.sparsercnn.config import add_sparsercnn_config
from spine.sparsercnn.register_dataset import register_spine_datasets

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        return COCOEvaluator(
            dataset_name,
            cfg,
            distributed=False,
            output_dir=output_folder
        )
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY

            if 'bias' in key or 'norm' in key:
                weight_decay = 0.0
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR)
        return optimizer
def setup_cfg(args):
    cfg = get_cfg()
    add_sparsercnn_config(cfg)
    # load sparse r-cnn config
    cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)
    # register dataset
    register_spine_datasets()

    cfg.freeze()
    return cfg

def train(args):
    cfg = setup_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    print('start training detection model: Sparse R-CNN')
    return trainer.train()

if __name__ == '__main__':
    setup_logger()

    args = default_argument_parser().parse_args()

    launch(
        train,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )
