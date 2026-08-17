# Hadamard 0.1B verification queue
# r054: denseHad h=896 L=8 + sandwich_norm @0.1B (primary verification)
# r056: denseHad h=512 L=6 + sandwich_norm @26M (small-N + sandwich ablation)
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_hadamard_01B.log"
SLICE01='/Volumes/pan/text/pretrain_train_0.1B.jsonl'
SLICE26='/Volumes/pan/text/pretrain_train_26M.jsonl'

echo "[$(date +%H:%M)] r054 denseHad896x8 sandwich @0.1B" >> "$LOG"
./experiments/run_exp.sh r054_dense896x8_had_sand_0.1B \
  "dense-Hadamard h=896 L=8 + sandwich_norm @0.1B; verification of r053 CE 5.2969 vs dense 6.0482" \
  --data_path "$SLICE01" --hidden_size 896 --num_hidden_layers 8 --num_attention_heads 8 \
  --kv_lora_rank 192 --ffn_type hadamard --sandwich_norm --mtp_depth 1 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r056 denseHad512x6 sandwich @26M" >> "$LOG"
./experiments/run_exp.sh r056_dense512x6_had_sand_26M \
  "dense-Hadamard 512x6 + sandwich_norm @26M; small-N iso-FLOPs sandwich ablation vs r052" \
  --data_path "$SLICE26" --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --ffn_type hadamard --sandwich_norm --mtp_depth 1 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] 0.1B verification queue done" >> "$LOG"
