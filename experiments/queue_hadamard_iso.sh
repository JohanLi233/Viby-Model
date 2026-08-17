# Hadamard iso-N near-iso-FLOPs arm
# Wait for queue_hadamard (r049/r050), then:
#   r051: dense-Hadamard h=896 L=8  N=28.87M @10M
#         iso-N vs r045 dense 512x6; theoretical per-token FLOPs ~1.11x
#         (FWHT replaces SwiGLU FFN; freed params reallocated to width/depth)
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_hadamard_iso.log"
SLICE='/Volumes/pan/text/pretrain_train_10M.jsonl'
echo "[$(date +%H:%M)] waiting for hadamard queue..." >> "$LOG"
while pgrep -f 'queue_hadamard.sh' > /dev/null; do sleep 30; done
while pgrep -f 'train_pretrain.py' > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] r051 dense-Hadamard h896 L8 @10M" >> "$LOG"
./experiments/run_exp.sh r051_dense896x8_had_10M \
  "dense-Hadamard h=896 L=8 N=28.87M @10M; iso-N near-iso-FLOPs vs dense512x6 (FWHT mixer)" \
  --data_path "$SLICE" --hidden_size 896 --num_hidden_layers 8 --num_attention_heads 8 \
  --kv_lora_rank 192 --ffn_type hadamard --mtp_depth 1 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] hadamard iso queue done" >> "$LOG"
