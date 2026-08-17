# r053: sandwich-norm ablation on best Hadamard candidate
# Wait for iso1 (r051) and iso2 (r052) queues, then:
#   r053 dense-Hadamard h=896 L=8 @10M + --sandwich_norm
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_sandwich.log"
SLICE='/Volumes/pan/text/pretrain_train_10M.jsonl'
echo "[$(date +%H:%M)] waiting for iso queues..." >> "$LOG"
while pgrep -f 'queue_hadamard_iso.sh' > /dev/null; do sleep 30; done
while pgrep -f 'queue_hadamard_iso2.sh' > /dev/null; do sleep 30; done
while pgrep -f 'train_pretrain.py' > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] r053 dense-Hadamard h896 L8 sandwich @10M" >> "$LOG"
./experiments/run_exp.sh r053_dense896x8_had_sand_10M \
  "dense-Hadamard h=896 L=8 + sandwich_norm @10M; paper ablation free -0.009 nats" \
  --data_path "$SLICE" --hidden_size 896 --num_hidden_layers 8 --num_attention_heads 8 \
  --kv_lora_rank 192 --ffn_type hadamard --sandwich_norm --mtp_depth 1 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] sandwich queue done" >> "$LOG"
