#!/usr/bin/env bash
# 跑一轮实验并追加记录到 results.tsv
# 用法: ./experiments/run_round.sh <round_name> <notes> [extra train args...]
# 训练参数固定为当前基线配方（MLA + 序列打包 + doc_mask + value_res +
# attn_gate + mtp1），extra args 追加在后面（可覆盖默认值）。
set -u

ROUND="$1"; NOTES="$2"; shift 2

cd "$(dirname "$0")/.."
mkdir -p experiments/logs out_exp

LOG="experiments/logs/${ROUND}.log"
OUT_DIR="out_exp/${ROUND}"

START=$(date +%s)
uv run trainer/train_pretrain.py \
    --data_path '/Volumes/pan/text/pretrain_t2t_mini.jsonl' \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --learning_rate 0.01 \
    --epochs 1 \
    --pack_sequences \
    --doc_mask \
    --use_value_res \
    --use_attn_gate \
    --batch_size 32 \
    --accumulation_steps 1 \
    --max_seq_len 640 \
    --lr_decay_steps 1700 \
    --max_train_minutes 30 \
    --log_interval 20 \
    --out_dir "$OUT_DIR" \
    "$@" 2>&1 | tee "$LOG"
END=$(date +%s)

# 从日志提取指标：最后一个 loss、最后一个 tokens/s、最后一个 step
LAST=$(grep -E '^Epoch:' "$LOG" | tail -1)
LOSS=$(echo "$LAST" | sed -E 's/.*loss:([0-9.]+).*/\1/')
TOKENS=$(echo "$LAST" | sed -E 's/.*tokens\/s:([0-9]+).*/\1/')
STEP=$(echo "$LAST" | sed -E 's/^Epoch:\[[0-9]+\/[0-9]+\]\(([0-9]+)\/.*/\1/')
PARAMS=$(grep -E '总参数量' "$LOG" | tail -1 | sed -E 's/.*：([0-9.]+)M.*/\1/')

# results.tsv: round date duration_s params_m final_loss tokens_per_s last_step notes
if [ ! -f results.tsv ]; then
    printf 'round\tdate\tduration_s\tparams_m\tfinal_loss\ttokens_per_s\tlast_step\tnotes\n' > results.tsv
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ROUND" "$(date +%Y-%m-%d\ %H:%M)" "$((END-START))" "$PARAMS" "$LOSS" "$TOKENS" "$STEP" "$NOTES" >> results.tsv

echo "=== recorded: round=$ROUND loss=$LOSS tokens/s=$TOKENS step=$STEP ==="
