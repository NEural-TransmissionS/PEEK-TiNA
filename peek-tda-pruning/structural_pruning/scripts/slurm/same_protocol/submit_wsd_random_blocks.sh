#!/usr/bin/env bash
# WSD random block-cut baseline under cut + fine-tune: random orders 3-22 (0-2 already run),
# k = 1..3, n and s. Cut sets already trained under cut + fine-tune (rankings, earlier orders,
# or an earlier order in this batch) are skipped; outputs/wsd_random_blocks_map.tsv records which
# trained run stands for each (model, order, k).
set -euo pipefail
S=/shared/projects/tina/PEEK-TiNA/topo-prune-shapes; cd $S
P=/shared/projects/tina/PEEK-TiNA/PEEK-rvc-pruning
PART="--partition=gpu1,gpu2,h200,h200_mig,vdi-long,vdi-med,vdi-short --exclude=gpu06"
FT="--epochs 20 --batch 16 --patience 100 --deterministic --amp --optimizer AdamW --lr0 0.00125 --momentum 0.9 --warmup-epochs 1 --device 0"
MAP=$P/outputs/wsd_random_blocks_map.tsv; LOG=$P/outputs/wsd_random_blocks_jobs.txt; : > $LOG
../tina_venv/bin/python - > /tmp/rb_plan.tsv <<'PY'
import glob, json, os, sys, numpy as np
sys.path.insert(0, ".")
from topo_prune import SCOPES
for m in "ns":
    have = {}
    for f in glob.glob(f"results/wsd_{m}_k*_ft.json"):
        j = json.load(open(f)); have[frozenset(j["layers_cut"])] = os.path.basename(f)[:-5]
    for s in range(3, 23):
        order = [int(i) for i in np.random.default_rng(s).permutation(SCOPES["all"])]
        for k in (1, 2, 3):
            key = frozenset(order[:k]); name = f"wsd_{m}_k{k}_all_rand{s}_ft"
            if key in have:
                print(f"{m}\t{s}\t{k}\t{sorted(key)}\treuse\t{have[key]}")
            else:
                have[key] = name; print(f"{m}\t{s}\t{k}\t{sorted(key)}\ttrain\t{name}")
PY
cp /tmp/rb_plan.tsv $MAP
echo "planned: $(grep -c train $MAP) new runs, $(grep -c reuse $MAP) reused cut sets"
grep -P "\ttrain\t" $MAP | while IFS=$'\t' read m s k cut what name; do
  j=$(sbatch --parsable $PART --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --job-name=wr-$m-r$s-k$k \
    --output=slurm/logs/wsd_randft_${m}_r${s}_k$k.out \
    --wrap=". ../tina_venv/bin/activate; python _train_one.py --dataset wsd --size $m --k $k --scope all --random-order $s --basis peek --init-weights weights/wsd_$m.pt $FT")
  echo "$m r$s k$k $cut $j" >> $LOG
done
echo "submitted $(wc -l < $LOG) jobs"; rm -f /tmp/rb_plan.tsv
