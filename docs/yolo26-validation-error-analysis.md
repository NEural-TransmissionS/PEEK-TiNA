# YOLO26n pretrained validation error analysis

This analysis uses only the 186-image WSD validation split. It does not inspect the held-out test split. Predictions use the seed-42 pretrained baseline, NMS IoU 0.60, and greedy class-aware matching at IoU 0.50.

At confidence 0.25, 615 of 1,009 annotations match, with 275 false positives and 394 false negatives. Recall varies strongly with normalized ground-truth area:

| Ground-truth size | Recall | False negatives |
|---|---:|---:|
| Small, area <1% | 0.369 | 238/377 |
| Medium, 1–10% | 0.719 | 138/491 |
| Large, ≥10% | 0.872 | 18/141 |

Class recall is 0.602 antenna, 0.799 body, 0.691 solar, and 0.051 thruster. Thruster has 111 false negatives among 117 instances. The two populous robustness slices have similar recall—0.600 for bright/large/many-part and 0.610 for dark/small/many-part—so brightness alone is not the dominant split. The four-image few-part slice is too small for a meaningful comparison.

## Confidence sweep

Inference retained predictions down to confidence 0.001, then applied the same matching offline:

| Confidence | Overall precision | Overall recall | Overall F1 | Thruster precision | Thruster recall | Thruster F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.077 | 0.828 | 0.141 | 0.018 | 0.538 | 0.036 |
| 0.010 | 0.246 | 0.760 | 0.372 | 0.051 | 0.325 | 0.088 |
| 0.025 | 0.362 | 0.727 | 0.484 | 0.084 | 0.256 | 0.126 |
| 0.050 | 0.455 | 0.699 | 0.551 | 0.117 | 0.197 | 0.146 |
| 0.100 | 0.561 | 0.671 | 0.611 | 0.176 | 0.137 | 0.154 |
| 0.200 | 0.660 | 0.633 | 0.646 | 0.270 | 0.085 | 0.130 |
| 0.250 | 0.691 | 0.610 | 0.648 | 0.240 | 0.051 | 0.085 |
| 0.400 | 0.757 | 0.575 | 0.654 | 0.294 | 0.043 | 0.075 |

Low-confidence recall shows that the model proposes many relevant regions but scores them poorly, while the size stratification shows a second small-object localization/representation problem. Therefore the next search changes only four evidence-linked factors: classification-loss weight, box/DFL weights, multiscale training with reduced scale jitter, and a combined multiscale/classification condition. Optimizer and native pretrained initialization remain unchanged. Selection remains validation mAP@0.5:0.95, with per-class AP and size-stratified errors as secondary diagnostics—not alternate winner criteria.

The pinned YOLO26 source also provides an official P2/4 detection-head architecture. Because the dominant failure is below 1% normalized box area, a second architecture-focused set tests YOLO26n-P2 with native settings, mild classification emphasis, reduced scale jitter, and moderate multiscale training. The P2 model adds a higher-resolution prediction scale and therefore increases compute; it must be reported as a separate architecture/accuracy tradeoff rather than a free hyperparameter improvement.
