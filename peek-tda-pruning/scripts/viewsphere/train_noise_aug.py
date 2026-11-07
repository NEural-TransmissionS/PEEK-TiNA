"""YOLO training with pixel-noise augmentation, for rendered images on an exactly black background.

Why (2026-11-06): viewsphere v2 renders are ~96% exactly-black pixels (WSD ~38%). Models trained on them
learn background features that react to any departure from exact 0: 1e-3 input noise on the background
moves class scores by up to ~0.9 (on the satellite itself: ~0.01), so fp16 vs fp32, each epoch's weights,
and BN re-estimation all swing mAP (v2 YOLOv5n: 0.48 fp16 vs 0.06 fp32 on the same weights).
Fix: during training only, add Gaussian noise with a per-image sigma drawn from U(0, NOISE/255) to every
training batch, so "background" never means "exactly 0". Validation and test images stay clean.
The noise itself lives in structural_pruning/peek_rvc_pruning/pixel_noise.py (shared with the pruning
fine-tunes).

Takes the same key=value arguments as `yolo detect train`, after the noise level (in gray levels):
    python train_noise_aug.py 3 model=yolov5nu.pt data=.../v2_wsd4/data.yaml epochs=1000 ...
Used by train_viewsphere_finetune.slurm when NOISE is set.
"""
import sys
from pathlib import Path

import yaml
from ultralytics.models.yolo.detect import DetectionTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "structural_pruning"))
from peek_rvc_pruning import pixel_noise  # noqa: E402

if __name__ == "__main__":
    noise = float(sys.argv[1])
    args = {k: yaml.safe_load(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    print(f"noise augmentation: per-image sigma ~ U(0, {noise}/255)", flush=True)
    pixel_noise.with_noise(DetectionTrainer, noise)(overrides=args).train()
