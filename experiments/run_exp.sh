#!/usr/bin/env bash
# 干净实验轮：训练指定配置 -> holdout PPL 评估 -> 追加 research/experiments.tsv
# 用法: ./experiments/run_exp.sh <round_name> <notes> [extra train args...]
set -u
ROUND="$1"; NOTES="$2"; shift 2
cd "$(dirname "$0")/.."
mkdir -p experiments/logs research_runs research
LOG="experiments/logs/${ROUND}.log"
OUT_DIR="research_runs/${ROUND}"
DATA='/Volumes/pan/text/pretrain_train_mini.jsonl'
HOLD='/Volumes/pan/text/pretrain_holdout_mini.jsonl'
TSV='research/experiments.tsv'

START=$(date +%s)
export PYTHONUNBUFFERED=1
uv run trainer/train_pretrain.py \
    --data_path "$DATA" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --learning_rate 0.01 \
    --epochs 1 \
    --pack_sequences \
    --doc_mask \
    --use_value_res \
    --use_attn_gate \
    --batch_size 48 \
    --accumulation_steps 1 \
    --max_seq_len 640 \
    --log_interval 100 \
    --out_dir "$OUT_DIR" \
    "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
END=$(date +%s)

if [ $RC -ne 0 ]; then
  echo "=== train failed rc=$RC, see $LOG ==="
  printf '%s\t%s\t%s\tFAILED\t%s\t%s\t%s\n' "$ROUND" "$(date +%Y-%m-%d\ %H:%M)" "$((END-START))" "0" "0" "$NOTES" >> "$TSV"
  exit $RC
fi

# holdout PPL（干净尾部 2000 篇，训练从未见过）
CKPT=$(ls "$OUT_DIR"/pretrain_*.safetensors 2>/dev/null | grep -v optimizer | head -1)
PPLLINE=$(uv run experiments/eval_ppl.py --ckpt "$CKPT" --data_path "$HOLD" --tag "$ROUND" --notes "$NOTES" 2>&1 | grep -E '^\[ppl\] ' | tail -1)
PPL=$(echo "$PPLLINE" | sed -E 's/.*overall_ppl=([0-9.]+).*/\1/')
CE=$(echo "$PPLLINE" | sed -E 's/.*mean_ce=([0-9.]+).*/\1/')
LAST=$(grep -E '^Epoch:' "$LOG" | tail -1)
LOSS=$(echo "$LAST" | sed -E 's/.*loss:([0-9.]+).*/\1/')
TOKENS=$(echo "$LAST" | sed -E 's/.*tokens\/s:([0-9]+).*/\1/')
STEP=$(echo "$LAST" | sed -E 's/^Epoch:\[[0-9]+\/[0-9]+\]\(([0-9]+)\/.*/\1/')
PARAMS=$(grep -E '总参数量' "$LOG" | tail -1 | sed -E 's/.*：([0-9.]+)M.*/\1/')

if [ ! -f "$TSV" ]; then
  printf 'round\tdate\tduration_s\tparams_m\ttrain_loss\ttokens_per_s\tlast_step\tholdout_ce\tholdout_ppl\tnotes\n' > "$TSV"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ROUND" "$(date +%Y-%m-%d\ %H:%M)" "$((END-START))" "$PARAMS" "$LOSS" "$TOKENS" "$STEP" "$CE" "$PPL" "$NOTES" >> "$TSV"
echo "=== recorded: $ROUND loss=$LOSS holdout_ppl=$PPL ==="
