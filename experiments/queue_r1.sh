#!/usr/bin/env bash
# R1 队列（等待夜间队列完全结束后执行）：
# 循环引理实测——同有效深度、同总 FLOPs 的 dense vs loop 对照。
#   r010 dense 576x12 @0.2B          (N=61.3M, 对照)
#   r011 loop 576x6 x2  @0.2B        (N≈35M, 同 FLOPs 同有效深度 12)
#   r012 loop 576x12x2  @0.1B        (N=61.3M, 同总 FLOPs，有效深度 24)
#   r013 dense 576x12 @0.1B          (N=61.3M, 干净 D 点)
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_r1.log"

echo "[$(date +%H:%M)] R1 queue waiting for overnight queue..." >> "$LOG"
while pgrep -f 'queue_overnight.sh' > /dev/null; do
  sleep 60
done
while pgrep -f 'train_pretrain.py' > /dev/null; do
  sleep 30
done

echo "[$(date +%H:%M)] R1 start: r010 dense576x12 @0.2B" >> "$LOG"
./experiments/run_exp.sh r010_dense61M_0.2B \
  "dense 576x12, D=0.2B dedicated cosine; loop-lemma control" \
  --data_path /Volumes/pan/text/pretrain_train_0.2B.jsonl \
  --hidden_size 576 --num_hidden_layers 12 --num_attention_heads 8 \
  --kv_lora_rank 192 --loop_k 1 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r011: loop 576x6 x2 @0.2B" >> "$LOG"
./experiments/run_exp.sh r011_loop35M_0.2B \
  "loop 576x6 x2 (N~35M, same FLOPs & effective depth 12 as r010)" \
  --data_path /Volumes/pan/text/pretrain_train_0.2B.jsonl \
  --hidden_size 576 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --loop_k 2 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r012: loop 576x12 x2 @0.1B" >> "$LOG"
./experiments/run_exp.sh r012_loop61M_0.1B \
  "loop 576x12 x2 (N=61.3M, same total FLOPs as r010, effective depth 24)" \
  --data_path /Volumes/pan/text/pretrain_train_0.1B.jsonl \
  --hidden_size 576 --num_hidden_layers 12 --num_attention_heads 8 \
  --kv_lora_rank 192 --loop_k 2 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r013: dense 576x12 @0.1B" >> "$LOG"
./experiments/run_exp.sh r013_dense61M_0.1B \
  "dense 576x12, D=0.1B dedicated cosine; clean D point" \
  --data_path /Volumes/pan/text/pretrain_train_0.1B.jsonl \
  --hidden_size 576 --num_hidden_layers 12 --num_attention_heads 8 \
  --kv_lora_rank 192 --loop_k 1 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] R1 queue done" >> "$LOG"
