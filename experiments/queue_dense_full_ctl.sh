# r059: dense 512x6 full-epoch control for r058 (Hadamard+sandwich)
# Same train_mini 343M full epoch, same seed 1337, same hyperparams;
# ONLY difference: default SwiGLU FFN, no sandwich_norm.
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_dense_full_ctl.log"
echo "[$(date +%H:%M)] r059 dense 512x6 full-epoch control start" >> "$LOG"
./experiments/run_exp.sh r059_dense512x6_full \
  "dense 512x6 full-epoch control @train_mini; vs r058 (Hadamard+sandwich) iso-data iso-seed iso-hyper; closed-book holdout" \
  --data_path /Volumes/pan/text/pretrain_train_mini.jsonl \
  --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --mtp_depth 1 --seed 1337 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] r059 done" >> "$LOG"
