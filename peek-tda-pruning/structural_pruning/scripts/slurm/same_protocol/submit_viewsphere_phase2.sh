#!/usr/bin/env bash
# Viewsphere phase 2 (after phase 1's block rankings exist): block cuts and the recovery study.
#   cut + FT : 20-epoch fine-tune of the cut viewsphere fine-tune (weights/viewsphere_{n,s}.pt), AdamW lr 1.25e-3
#              momentum 0.9, warmup 1, matched warmup bias lr. Cut sets: k0, PEEK raw head k1-3, grid/null all
#              k1-3, activation head k1-3, random orders 0-22 (all scope); a cut set already planned is reused.
#   scratch  : the same block cuts retrained from scratch like WSD's (1000 ep, patience 100, batch 16,
#              deterministic, AMP, optimizer auto). Cut sets: k0, PEEK raw head k1-3 for combined/h0/h1,
#              grid/null all k1-3, activation head k1-3, random orders 0-2; reused the same way.
#   recovery : submit_recovery.sh with DATASETS=viewsphere STUDIES="probe budget" WBLR=1.
# Plans: outputs/viewsphere_blocks_ft_map.tsv, outputs/viewsphere_blocks_scratch_map.tsv.
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes
X="--exclude=gpu06"
PW="--partition=gpu1,gpu2,h200,h200_mig,vdi-med,vdi-long $X"
PL="--partition=gpu1,gpu2,h200,h200_mig,vdi-long $X"
LOG=$P/outputs/viewsphere_jobs.txt
cd $S
for m in n s; do ln -sfn viewsphere_${m}_norm.json ranking_peek/viewsphere_$m.json; done  # raw head = top level of _norm

plan() {  # mode (ft|scratch) -> "m what k cut train|reuse args" lines
  ../tina_venv/bin/python - "$1" <<'PY'
import json, sys, numpy as np
sys.path.insert(0, ".")
from topo_prune import SCOPES, cut_order, ranking_path
mode = sys.argv[1]
crits = ("combined", "h0", "h1") if mode == "scratch" else ("combined",)
orders = range(3) if mode == "scratch" else range(23)
for m in "ns":
    have = {frozenset(): "--k 0"}; print(f"{m}\tk0\t0\t[]\ttrain\t--k 0")
    def add(what, k, cut, args):
        cut = frozenset(cut)
        print(f"{m}\t{what}\t{k}\t{sorted(cut)}\t{'reuse' if cut in have else 'train'}\t{have.get(cut, args)}")
        have.setdefault(cut, args)
    for k in (1, 2, 3):
        head = json.loads(ranking_path("viewsphere", m, "peek", "head", "norm").read_text())
        for c in crits:
            add(f"head-{c}", k, cut_order(head, c)[0][:k], f"--k {k} --scope head --criterion {c} --basis peek")
        r = json.loads(ranking_path("viewsphere", m, "peek", "all", "norm").read_text())
        for v in ("grid", "null"):
            add(v, k, cut_order({**r, **r[v]}, "combined")[0][:k], f"--k {k} --scope all --norm {v} --basis peek")
        act = json.loads(ranking_path("viewsphere", m, "activation", "head", "").read_text())
        add("act-head", k, cut_order(act, "combined")[0][:k], f"--k {k} --scope head --basis activation")
    for s in orders:
        order = [int(i) for i in np.random.default_rng(s).permutation(SCOPES["all"])]
        for k in (1, 2, 3):
            add(f"rand{s}", k, order[:k], f"--k {k} --scope all --random-order {s} --basis peek")
PY
}

FT="--epochs 20 --batch 16 --patience 100 --deterministic --amp --optimizer AdamW --lr0 0.00125 --momentum 0.9 --warmup-epochs 1 --warmup-bias-lr 0.00125 --device 0"
SCR="--epochs 1000 --batch 16 --patience 100 --deterministic --amp --device 0"
for mode in ft scratch; do
  MAP=$P/outputs/viewsphere_blocks_${mode}_map.tsv; plan $mode > $MAP
  echo "block $mode plan: $(grep -cP '\ttrain\t' $MAP) runs, $(grep -cP '\treuse\t' $MAP) reused cut sets" | tee -a $LOG
  grep -P "\ttrain\t" $MAP | while IFS=$'\t' read m what k cut _ args; do
    if [ $mode = ft ]; then extra="--init-weights weights/viewsphere_$m.pt $FT"; part=$PW; tl=01:30:00
    else extra="$SCR"; part=$PL; tl=12:00:00; fi
    j=$(sbatch --parsable $part --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=$tl --job-name=vb$mode-$m-$what-k$k \
      --output=slurm/logs/viewsphere_block${mode}_${m}_${what}_k$k.out \
      --wrap=". ../tina_venv/bin/activate; python _train_one.py --dataset viewsphere --size $m $args $extra --skip-existing")
    echo "block $mode $m $what k$k $cut $j" >> $LOG
  done
done

cd $P
DATASETS=viewsphere STUDIES="probe budget" WBLR=1 bash scripts/slurm/submit_recovery.sh
echo "phase 2 submitted; log $LOG, recovery ids in outputs/recovery_jobs.txt"
