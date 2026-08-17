# Small-N grid for the Hadamard long-D route (34M slice, cheap)
# r059: h=384 L=6  N=7.18M  FLOPs/tok=0.26x dense
# r060: h=448 L=8  N=10.43M FLOPs/tok=0.42x dense
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_ngrid.log"
S='/Volumes/pan/text/pretrain_train_34M.jsonl'
echo "[$(date +%H:%M)] r059 denseHad384x6 sandwich @34M" >> "$LOG"
./experiments/run_exp.sh r059_dense384x6_had_sand_34M \
  "small-N grid: denseHad 384x6 N=7.18M @34M" \
  --data_path "$S" --hidden_size 384 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 160 --ffn_type hadamard --sandwich_norm --mtp_depth 1 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] r060 denseHad448x8 sandwich @34M" >> "$LOG"
./experiments/run_exp.sh r060_dense448x8_had_sand_34M \
  "small-N grid: denseHad 448x8 N=10.43M @34M" \
  --data_path "$S" --hidden_size 448 --num_hidden_layers 8 --num_attention_heads 8 \
  --kv_lora_rank 192 --ffn_type hadamard --sandwich_norm --mtp_depth 1 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] ngrid done" >> "$LOG"
