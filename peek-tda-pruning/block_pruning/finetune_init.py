"""Cut-and-fine-tune protocol for block replacement.

Builds the cut model for a dataset, copies every pretrained weight whose name and shape still
match (the replacement 1x1 convs start fresh), and hands that model to the trainer. A plain
``YOLO(yaml).load(weights).train()`` would rebuild the model with random weights. The trainer
also seeds the train loader from the run seed (Ultralytics fixes it otherwise).

The from-scratch protocol and the run bookkeeping live in ``_train_one.py``.
"""

from __future__ import annotations

from pathlib import Path


def init_from_pretrained(cfg: Path, weights: str, data: Path):
    """Cut model with the dataset's class count + every matching pretrained tensor.
    Returns (model, tensors copied, tensors in the model)."""
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import intersect_dicts

    net = DetectionModel(str(cfg), nc=check_det_dataset(str(data))["nc"], verbose=False)
    copied = intersect_dicts(YOLO(str(weights)).model.float().state_dict(), net.state_dict())
    net.load_state_dict(copied, strict=False)
    return net, len(copied), len(net.state_dict())


def trainer_for(net, seed: int = 42):
    """DetectionTrainer that trains ``net`` as given, with the run seed in the train loader."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class InitializedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return net

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            if mode == "train" and seed != 42:  # seed 42 keeps Ultralytics' fixed stream
                loader.generator.manual_seed(loader.generator.initial_seed() + seed - 42)
                loader.reset()
            return loader

    return InitializedTrainer
