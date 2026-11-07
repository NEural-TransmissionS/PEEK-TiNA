"""Training-only pixel-noise augmentation for rendered datasets (viewsphere v1 / v2).

The renders are ~96% exactly-black pixels; models trained on them learn background features that react
to any departure from exact 0 (1e-3 noise on the background moves class scores by up to ~0.9), so fp16
vs fp32, each epoch's weights and BN re-estimation all swing mAP. Adding Gaussian noise with a per-image
sigma ~ U(0, max_gray/255) to every training batch fixes that (noise pilot, 2026-11-06).

Used by: scripts/viewsphere/train_noise_aug.py (base models), run_yolo26_tda_pruning.make_trainer
(channel-pruning fine-tunes) and block_pruning/finetune_init.trainer_for (block fine-tunes); the last
two read the level from the PIXEL_NOISE environment variable (unset or 0 = off), as does
run_yolo26_tda_pruning.bn_calibration_images (the same noise on the BN re-estimation images).
"""
import os

import torch


def level() -> float:
    """PIXEL_NOISE (gray levels of 255), 0 if unset."""
    return float(os.environ.get("PIXEL_NOISE") or 0)


def add(img: torch.Tensor, max_gray: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """img: float batch in [0, 1], N x C x H x W. Returns a noisy copy, clamped to [0, 1].
    generator (on img's device) makes the noise reproducible, e.g. for BN calibration images."""
    sigma = torch.rand(img.shape[0], 1, 1, 1, device=img.device, generator=generator) * (max_gray / 255)
    noise = torch.randn(img.shape, device=img.device, dtype=img.dtype, generator=generator)
    return (img + noise * sigma).clamp_(0, 1)


def with_noise(trainer_cls, max_gray: float):
    """trainer_cls with preprocess_batch adding the noise (training batches only; the validator has its
    own preprocessing). max_gray <= 0 returns trainer_cls unchanged."""
    if max_gray <= 0:
        return trainer_cls

    class NoisyTrainer(trainer_cls):
        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            batch["img"] = add(batch["img"], max_gray)
            return batch

    NoisyTrainer.__name__ = f"Noisy{trainer_cls.__name__}"
    return NoisyTrainer
