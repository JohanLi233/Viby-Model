#!/usr/bin/env bash
# ΔW-Loop 筛查队列（等无训练进程后串行执行）：
# 三臂同族 512x6 MLA（N≈28.5M）、同 0.1B 切片 1 epoch（cosine 随数据量自动
# 对齐），唯一变量是循环调制类型。判据：holdout CE 差 > 0.05 nats 才采信。
#   r030 dense 512x6    @0.1B   (loop_k=1，同 N 同 D 锚点)
#   r031 loop 512x6 x2  @0.1B   (FiLM-only 对照)
#   r032 loop 512x6 x2  @0.1B   (ΔW-Loop，dw_rank=8；+0.55M 参数 ≈ +1.9%)
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_dw.log"
SLICE='/Volumes/pan/text/pretrain_train_0.1B.jsonl'

echo "[$(date +%H:%M)] ΔW queue waiting for running trainings..." >> "$LOG"
while pgrep -f 'train_pretrain.py' > /dev/null; do
  sleep 30
done

echo "[$(date +%H:%M)] r030: dense 512x6 @0.1B" >> "$LOG"
./experiments/run_exp.sh r030_dense28M_0.1B \
  "dense 512x6 @0.1B slice; ΔW screening anchor (iso-N iso-D)" \
  --data_path "$SLICE" \
  --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --loop_k 1 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r031: loop 512x6 x2 FiLM @0.1B" >> "$LOG"
./experiments/run_exp.sh r031_loop28M_film_0.1B \
  "loop 512x6 x2 FiLM-only @0.1B; ΔW control (identical except dw)" \
  --data_path "$SLICE" \
  --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --loop_k 2 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r032: loop 512x6 x2 ΔW r8 @0.1B" >> "$LOG"
./experiments/run_exp.sh r032_loop28M_dw8_0.1B \
  "loop 512x6 x2 + dw_rank=8 @0.1B; weight-space per-step diversity" \
  --data_path "$SLICE" \
  --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --loop_k 2 --dw_rank 8 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] ΔW queue done" >> "$LOG"
