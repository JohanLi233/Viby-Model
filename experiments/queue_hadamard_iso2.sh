# r052: strict iso-total-FLOPs dense-Hadamard arm
# dense512x6-Hadamard N=10.62M, proxy 22.6 FLOPs/tok.
# Baseline r045 dense512x6: 10M tokens * 58.3 = 583M token-FLOP units.
# iso D = 583/22.6 = 25.8M -> use 26M slice.
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_hadamard_iso2.log"
SLICE='/Volumes/pan/text/pretrain_train_26M.jsonl'
echo "[$(date +%H:%M)] waiting for iso1 queue..." >> "$LOG"
while pgrep -f 'queue_hadamard_iso.sh' > /dev/null; do sleep 30; done
while pgrep -f 'train_pretrain.py' > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] r052 dense-Hadamard 512x6 @26M (iso-total-FLOPs)" >> "$LOG"
./experiments/run_exp.sh r052_dense512x6_had_26M \
  "dense-Hadamard 512x6 N=10.62M @26M; strict iso-total-FLOPs vs r045 dense@10M" \
  --data_path "$SLICE" --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --ffn_type hadamard --mtp_depth 1 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] iso2 queue done" >> "$LOG"
