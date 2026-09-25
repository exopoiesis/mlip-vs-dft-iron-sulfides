#!/bin/bash
# RUNS INSIDE the container. Paths go through environment variables so that two ladders
# (different holdouts) do not overwrite each other. Same arguments as train3.sh.
#   R23_OUT  - directory with train_*.xyz / valid_*.xyz (default /work/out)
#   R23_RUNS - where to place the runs (default /work/runs)
#   R23_WORK - working directory (default /work)
#   R23_FOUNDATION - foundation checkpoint (default historical MACE cache path)
#   R23_PYTHON - Python executable (default python3)
# Arguments: <rung> <seed> <epochs> <weight_E> <weight_F> <lr> <dtype> <tag>
set -uo pipefail

if [ "$#" -ne 8 ]; then
  echo "Usage: train4.sh <rung> <seed> <epochs> <weight_E> <weight_F> <lr> <dtype> <tag>" >&2
  exit 2
fi

RUNG=$1; SEED=$2; EPOCHS=$3; EW=$4; FW=$5; LR=$6; DT=$7; TAG=$8
OUTD=${R23_OUT:-/work/out}
RUNS=${R23_RUNS:-/work/runs}

OUT="${RUNS}/${TAG}"
mkdir -p "$OUT"
E0S='{1:-3.6673775936325184, 16:-4.673811566998673, 26:-7.054671703876374}'
FOUND=${R23_FOUNDATION:-/root/.cache/mace/MACE_MPtrj_20229model}
PYTHON=${R23_PYTHON:-python3}

cd "${R23_WORK:-/work}" || exit 1
"$PYTHON" -m mace.cli.run_train \
  --name "$TAG" \
  --train_file "${OUTD}/train_${RUNG}_s${SEED}.xyz" \
  --valid_file "${OUTD}/valid_${RUNG}_s${SEED}.xyz" \
  --energy_key REF_energy --forces_key REF_forces \
  --E0s "$E0S" \
  --foundation_model "$FOUND" \
  --multiheads_finetuning False \
  --model_dir "$OUT" --checkpoints_dir "$OUT" --results_dir "$OUT" --log_dir "$OUT" \
  --device cuda --default_dtype "$DT" \
  --batch_size 4 --valid_batch_size 4 \
  --lr "$LR" --energy_weight "$EW" --forces_weight "$FW" \
  --max_num_epochs "$EPOCHS" --patience 250 --seed "$SEED" --save_cpu \
  > "$OUT/train.log" 2>&1
rc=$?

"$PYTHON" - "$OUT/train.log" <<'PY'
import re, sys, datetime
ts=[]
for line in open(sys.argv[1], errors="replace"):
    m=re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.\d+ INFO: Epoch (\d+):", line)
    if m: ts.append((datetime.datetime.strptime(m.group(1),"%Y-%m-%d %H:%M:%S"), int(m.group(2))))
if len(ts)>2:
    dt=(ts[-1][0]-ts[0][0]).total_seconds(); de=ts[-1][1]-ts[0][1]
    print(f"    {dt/de:6.2f} s/epoch")
PY

echo "rc=$rc tag=$TAG"
[ "$rc" -ne 0 ] && tail -8 "$OUT/train.log"
exit "$rc"
