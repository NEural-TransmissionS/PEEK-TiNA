#!/usr/bin/env bash
# COCO fine-tunes at lr 1.25e-4 (AdamW, momentum 0.9, 20 epochs, warmup 1, batch 16, 10k subset).
# Why: the WSD lr (1.25e-3) drops the unpruned pretrained yolo26n from 0.400 to 0.308; the lr pilot
# (outputs/yolo26_tda/coco_n_protect6_lrpilot_lr*) kept it at 0.370 with 1.25e-4. Old-lr n PEEK runs are
# kept in coco_n_protect6_peek_ft20_lr1.25e-3. Scores, rankings and one-shot results are lr-independent.
#   channel : n/s PEEK grid 0-40, activation grid 1-30, seeds 43/44 (none/field_h1/l2/random)
#   blocks  : cut + fine-tune from pretrained: k0, raw head k1-3, grid/null all k1-3, random orders 0-22
#             (cut sets already planned are reused; outputs/coco_random_blocks_map.tsv)
set -euo pipefail
T=/shared/projects/tina/PEEK-TiNA; P=$T/PEEK-rvc-pruning; S=$T/topo-prune-shapes; O=$P/outputs/yolo26_tda
cd $P
# old held submissions (lr 1.25e-3 in their environment)
scancel 124824 124825 124826 124827 124828 124829 124830 124837 124838 124841 124842 124843 124844 124845 || true
if [ -d $O/coco_n_protect6_peek_ft20/runs ]; then
  mkdir -p $O/coco_n_protect6_peek_ft20_lr1.25e-3 && mv $O/coco_n_protect6_peek_ft20/runs $O/coco_n_protect6_peek_ft20_lr1.25e-3/
fi
mkdir -p $O/coco_n_protect6_peek_ft20/runs
cp -rn $O/coco_n_protect6_lrpilot_lr0.000125/runs/* $O/coco_n_protect6_peek_ft20/runs/   # same config, reused

export REPO_ROOT=$P PROTECT_LAYERS=6 WORKERS=16 BATCH=16 RATIOS="0.1 0.2 0.3 0.4 0.5"
export OPTIMIZER=AdamW LR0=0.000125 MOMENTUM=0.9 EPOCHS=20 WARMUP_EPOCHS=1
export DATA=$P/configs/coco/coco.yaml FT_DATA=$P/configs/coco/coco_ft10k.yaml
SB=scripts/slurm/yolo26_tda.sbatch; X="--exclude=gpu06"
PN="--partition=gpu1,gpu2,h200,h200_mig,vdi-med,vdi-long $X"
PS="--partition=gpu1,gpu2,h200,h200_mig $X"
TDA="h0 h1 total field_h0 field_h1 field_total"; ALL="h0 h1 total l2 random field_h0 field_h1 field_total"
LOG=outputs/coco_lr_jobs.txt; : > $LOG
prune() {  # part name array seed time
  sbatch --parsable $1 --export=ALL,SEED=$4 --job-name=$2 --array=$3 --time=$5 $SB prune; }
agg() {  # name deps(:a:b)
  sbatch --parsable --partition=med --export=ALL --job-name=$1 --gres=none --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --dependency=afterany$2 $SB aggregate; }

for m in n s; do
  export WEIGHTS=$T/yolo26$m.pt
  if [ $m = n ]; then PP=$PN; tl=03:00:00; else PP=$PS; tl=04:00:00; fi
  export BASIS=peek OUT=$O/coco_${m}_protect6_peek_ft20 CRITERIA="$ALL"
  pk=$(prune "$PP" cl-$m-peek 0-40 42 $tl); echo "coco_$m peek grid: $pk agg=$(agg cl-$m-peek-agg :$pk)" | tee -a $LOG
  export BASIS=activation OUT=$O/coco_${m}_protect6_activation_ft20 CRITERIA="$TDA"
  ac=$(prune "$PP" cl-$m-act 1-30 42 $tl); echo "coco_$m activation grid: $ac agg=$(agg cl-$m-act-agg :$ac)" | tee -a $LOG
  export BASIS=peek OUT=$O/coco_${m}_protect6_seeds_ft20 CRITERIA="field_h1 l2 random"
  sd=""; for s in 43 44; do sd="$sd:$(prune "$PP" cl-$m-s$s 0-15 $s $tl)"; done
  echo "coco_$m seeds: ${sd#:} agg=$(agg cl-$m-seeds-agg $sd)" | tee -a $LOG
done

# block cuts, cut + fine-tune from the pretrained COCO weights
cd $S
for m in n s; do ln -sfn coco_${m}_norm.json ranking_peek/coco_$m.json; done   # raw ranking = top level of the _norm file
FT="--epochs 20 --batch 16 --patience 100 --deterministic --amp --optimizer AdamW --lr0 0.000125 --momentum 0.9 --warmup-epochs 1 --device 0 --data $P/configs/coco/coco_ft10k.yaml"
MAP=$P/outputs/coco_random_blocks_map.tsv
../tina_venv/bin/python - > $MAP <<'PY'
import json, sys, numpy as np
sys.path.insert(0, ".")
from topo_prune import SCOPES, cut_order, ranking_path
for m in "ns":
    have = {frozenset(): "k0"}; print(f"{m}\tk0\t0\t[]\ttrain\t--k 0")
    for k in (1, 2, 3):
        r = json.loads(ranking_path("coco", m, "peek", "head", "").read_text())
        cut = frozenset(cut_order(r, "combined")[0][:k]); args = f"--k {k} --scope head"
        print(f"{m}\thead\t{k}\t{sorted(cut)}\t{'reuse' if cut in have else 'train'}\t{have.get(cut, args)}"); have.setdefault(cut, args)
        for v in ("grid", "null"):
            r = json.loads(ranking_path("coco", m, "peek", "all", "norm").read_text()); r = {**r, **r[v]}
            cut = frozenset(cut_order(r, "combined")[0][:k]); args = f"--k {k} --scope all --norm {v}"
            print(f"{m}\t{v}\t{k}\t{sorted(cut)}\t{'reuse' if cut in have else 'train'}\t{have.get(cut, args)}"); have.setdefault(cut, args)
    for s in range(23):
        order = [int(i) for i in np.random.default_rng(s).permutation(SCOPES["all"])]
        for k in (1, 2, 3):
            cut = frozenset(order[:k]); args = f"--k {k} --scope all --random-order {s}"
            print(f"{m}\trand{s}\t{k}\t{sorted(cut)}\t{'reuse' if cut in have else 'train'}\t{have.get(cut, args)}"); have.setdefault(cut, args)
PY
echo "block plan: $(grep -cP '\ttrain\t' $MAP) runs, $(grep -cP '\treuse\t' $MAP) reused cut sets" | tee -a $P/$LOG
grep -P "\ttrain\t" $MAP | while IFS=$'\t' read m what k cut _ args; do
  if [ $m = n ]; then PP=$PN; tl=03:00:00; else PP=$PS; tl=04:00:00; fi
  j=$(sbatch --parsable $PP --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=$tl --job-name=cb-$m-$what-k$k \
    --output=slurm/logs/coco_blockft_${m}_${what}_k$k.out \
    --wrap=". ../tina_venv/bin/activate; python _train_one.py --dataset coco --size $m $args --basis peek --init-weights weights/coco_$m.pt $FT")
  echo "block $m $what k$k $cut $j" >> $P/$LOG
done
echo "submitted: $(grep -c . $P/$LOG) lines in $LOG"
