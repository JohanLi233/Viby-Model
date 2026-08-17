# Seed-2 reproducibility queue for the r053 result
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_repeat.log"
S10='/Volumes/pan/text/pretrain_train_10M.jsonl'
echo "[$(date +%H:%M)] waiting for 0.1B verification queue..." >> "$LOG"
while pgrep -f 'queue_hadamard_01B.sh' > /dev/null; do sleep 60; done
while pgrep -f 'train_pretrain.py' > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] r055 r053-repeat seed2 @10M" >> "$LOG"
./experiments/run_exp.sh r055_dense896x8_had_sand_10M_s2 \
  "r053 repeat seed=2: dense-Hadamard h896 L8 + sandwich @10M" \
  --data_path "$S10" --hidden_size 896 --num_hidden_layers 8 --num_attention_heads 8 \
  --kv_lora_rank 192 --ffn_type hadamard --sandwich_norm --mtp_depth 1 --seed 2 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] r057 dense baseline seed2 @10M" >> "$LOG"
./experiments/run_exp.sh r057_dense512x6_10M_s2 \
  "dense 512x6 @10M seed=2; seed-noise floor for r053 comparison" \
  --data_path "$S10" --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --mtp_depth 1 --seed 2 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] repeat queue done" >> "$LOG"
