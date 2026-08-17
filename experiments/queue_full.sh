# r058: small-N + long-D verification for Hadamard route
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/r058_full.log"
DATA='/Volumes/pan/text/pretrain_train_mini.jsonl'
./experiments/run_exp.sh r058_dense512x6_had_sand_full \
  "dense-Hadamard 512x6 + sandwich_norm @343M full epoch; small-N long-D scaling probe" \
  --data_path "$DATA" --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --ffn_type hadamard --sandwich_norm --mtp_depth 1 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] r058 done" >> "$LOG"
