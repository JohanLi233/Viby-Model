#!/usr/bin/env bash
# 夜间顺序队列：r000 结束后依次跑 r001(61M)/r002(30M)/r003(15M)。
# 所有模型同族 MLA+value_res+attn_gate+mtp1+pack+doc_mask；
# save_interval=3255 -> 保存 0.1B/0.2B/0.3B token 检查点（bs48*seq640=30720/步）。
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_overnight.log"

echo "[$(date +%H:%M)] queue waiting for r000..." >> "$LOG"
while pgrep -f 'train_pretrain.py.*r000_60M_full' > /dev/null; do
  sleep 60
done
echo "[$(date +%H:%M)] r000 done, start r001 (61M: 576x12 MLA)" >> "$LOG"
./experiments/run_exp.sh r001_60M_full \
  "N=61.3M (576x12 MLA) full epoch; save@0.1/0.2/0.3B" \
  --hidden_size 576 --num_hidden_layers 12 --num_attention_heads 8 \
  --kv_lora_rank 192 --save_interval 3255 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r001 done, start r002 (30M: 512x6 MLA)" >> "$LOG"
./experiments/run_exp.sh r002_30M_full \
  "N=28.5M (512x6 MLA) full epoch; save@0.1/0.2/0.3B" \
  --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --save_interval 3255 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r002 done, start r003 (15M: 320x8 MLA)" >> "$LOG"
./experiments/run_exp.sh r003_15M_full \
  "N=15.2M (320x8 MLA) full epoch; save@0.1/0.2/0.3B" \
  --hidden_size 320 --num_hidden_layers 8 --num_attention_heads 8 \
  --kv_lora_rank 160 --save_interval 3255 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] queue done" >> "$LOG"
