# Pruning report

`pruning_report.tex` -> `pruning_report.pdf`: the YOLO26 / YOLOv5 pruning results on WSD, Viewsphere (v1, v2)
and COCO. The prose lives in the `.tex`; every number in a table or figure is generated from
`../../results/same_protocol/` (written on the server by `scripts/collect_same_protocol.py`).

| File | Job |
|---|---|
| `make_tables.py` | all LaTeX tables -> `tables/` (YOLO26 no suffix, YOLOv5 `_v5`, Viewsphere v2 `_vs2`) |
| `win_chart.py` | how often each channel criterion wins -> `figures/criterion_wins*.png`, `figures/budget_wins*.png` |
| `gap_chart.py` | persistence scores minus L2 by pruning ratio -> `figures/gap_to_l2.png` |

Rebuild after re-collecting the results:

    python make_tables.py && python win_chart.py && python gap_chart.py
    tectonic pruning_report.tex

Each script takes another `results/same_protocol` folder as its first argument.
