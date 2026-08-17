# r065: lean Engram @34M — longer-D arm. Control r059 (denseHad 384x6 sand @34M, CE 4.2174) already exists.
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_engram_34M.log"
S34='/Volumes/pan/text/pretrain_train_34M.jsonl'

echo "[$(date +%H:%M)] r065 denseHad384x6 sandwich + lean Engram @34M" >> "$LOG"
./experiments/run_exp.sh r065_dense384x6_had_sand_eng_34M \
  "lean Engram @34M: site(2) orders(2,3) slots4096 sub128 on denseHad 384x6 +sandwich; control=r059 CE 4.2174; tests if knowledge-memory gain needs longer D" \
  --data_path "$S34" --hidden_size 384 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 160 --ffn_type hadamard --sandwich_norm --mtp_depth 1 \
  --engram_layers 2 --engram_orders 2,3 --engram_slots 4096 --engram_sub_dim 128 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] engram 34M done" >> "$LOG"
