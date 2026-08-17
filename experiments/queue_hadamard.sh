# Hadamard channel-mixer compute-efficiency queue
# Wait for engram queue (r048) to finish, then:
#   r049: dense-Hadamard 512x6 @34M tokens  (iso-total-matmul-FLOPs vs r045 dense@10M)
#   r050: dense-Hadamard 512x6 @10M tokens  (iso-D ablation for N-projection)
# Both use FWHT HadamardMLP (O(n log n), 3n params/layer), standard dense
# attention recipe (value_res/attn_gate/MTP1), ffn_type=hadamard.
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_hadamard.log"
H10='/Volumes/pan/text/pretrain_train_10M.jsonl'
H34='/Volumes/pan/text/pretrain_train_34M.jsonl'

echo "[$(date +%H:%M)] waiting for engram queue..." >> "$LOG"
while pgrep -f 'queue_engram_10M.sh' > /dev/null; do sleep 30; done
while pgrep -f 'train_pretrain.py' > /dev/null; do sleep 30; done

echo "[$(date +%H:%M)] r049 dense-Hadamard 512x6 @34M" >> "$LOG"
./experiments/run_exp.sh r049_dense512x6_had_34M \
  "dense-Hadamard 512x6 N=10.6M @34M; iso-matmul-FLOPs vs r045 dense@10M; FWHT mixer" \
  --data_path "$H34" --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --ffn_type hadamard --mtp_depth 1 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r050 dense-Hadamard 512x6 @10M" >> "$LOG"
./experiments/run_exp.sh r050_dense512x6_had_10M \
  "dense-Hadamard 512x6 N=10.6M @10M; iso-D ablation vs r045 dense@10M" \
  --data_path "$H10" --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --ffn_type hadamard --mtp_depth 1 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] hadamard queue done" >> "$LOG"
