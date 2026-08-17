# Lean-Engram arm only (user scope cut): 384x6 denseHad+sandwich @10M, control vs Engram.
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_engram_lean.log"
S10='/Volumes/pan/text/pretrain_train_10M.jsonl'

echo "[$(date +%H:%M)] r063 denseHad384x6 sandwich @10M (engram control)" >> "$LOG"
./experiments/run_exp.sh r063_dense384x6_had_sand_10M \
  "lean-Engram screen control: denseHad 384x6 +sandwich @10M" \
  --data_path "$S10" --hidden_size 384 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 160 --ffn_type hadamard --sandwich_norm --mtp_depth 1 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r064 denseHad384x6 sandwich + lean Engram @10M" >> "$LOG"
./experiments/run_exp.sh r064_dense384x6_had_sand_eng_10M \
  "lean Engram on denseHad 384x6 +sandwich @10M: site(2) orders(2,3) slots4096 sub128, +1.2M table params; B1 knowledge arm" \
  --data_path "$S10" --hidden_size 384 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 160 --ffn_type hadamard --sandwich_norm --mtp_depth 1 \
  --engram_layers 2 --engram_orders 2,3 --engram_slots 4096 --engram_sub_dim 128 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] engram lean queue done" >> "$LOG"
