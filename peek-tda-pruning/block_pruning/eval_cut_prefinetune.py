"""mAP50-95 of each ranked block cut right after the cut (pretrained tensors copied, replacement 1x1 convs fresh), before any fine-tune."""
import csv, json, sys
from pathlib import Path
from ultralytics import YOLO
from topo_prune import build_pruned_yaml
from finetune_init import init_from_pretrained
HERE = Path(__file__).parent
DATA = {"wsd": str(HERE / "data/wsd/data.yaml"), "coco": "/shared/projects/tina/PEEK-TiNA/PEEK-rvc-pruning/configs/coco/coco.yaml"}
rows = list(csv.DictReader(open(sys.argv[1])))
cuts = sorted({(r["dataset"], r["model"], r["layers_cut"]) for r in rows if r["random_order"] == ""})
out = []
for ds, m, cut in cuts:
    layers = [int(x) for x in cut.split()]
    cfg = build_pruned_yaml(m, layers, HERE / "runs" / "_cfg")
    net, _, _ = init_from_pretrained(cfg, HERE / f"weights/{ds}_{m}.pt", DATA[ds])
    y = YOLO(str(cfg)); y.model = net
    r = y.val(data=DATA[ds], batch=32, imgsz=640, workers=8, plots=False, verbose=False, project=str(HERE / "runs/_precut"), name=f"{ds}_{m}", exist_ok=True)
    out.append({"dataset": ds, "model": m, "layers_cut": cut, "map50_95_precut": round(r.box.map, 4)})
    print(out[-1], flush=True)
json.dump(out, open(HERE / "results/precut_eval.json", "w"), indent=1)
