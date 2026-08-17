# Knowledge arm v2 + ceiling control:
#   r066: engram-v2 on denseHad 384x6 sand @34M — value_proj zero-init (noise-injection fix),
#         capacity fix: orders 1,2,3 / slots 8192 / heads 2 / sub_dim 64 (+3.4M, N≈10.4M).
#         Control: r059 CE 4.2174. v1 (r064/r065) was negative at both 10M and 34M.
#   then: restart r059_dense512x6_full (dense 512x6 @343M control, was killed at step 1400) —
#         required for the iso-D ceiling comparison vs r058.
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_knowledge_v2.log"
S34='/Volumes/pan/text/pretrain_train_34M.jsonl'

echo "[$(date +%H:%M)] r066 engram-v2 denseHad384x6 sandwich @34M" >> "$LOG"
./experiments/run_exp.sh r066_dense384x6_had_sand_eng2_34M \
  "engram-v2: value_proj zero-init + orders(1,2,3) slots8192 heads2 sub64 (+3.4M, N=10.36M) on denseHad 384x6 sand @34M; control=r059 CE 4.2174" \
  --data_path "$S34" --hidden_size 384 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 160 --ffn_type hadamard --sandwich_norm --mtp_depth 1 \
  --engram_layers 2 --engram_orders 1,2,3 --engram_slots 8192 --engram_sub_dim 64 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r059 dense 512x6 full-epoch control restart @343M" >> "$LOG"
./experiments/run_exp.sh r059_dense512x6_full \
  "dense 512x6 full-epoch control @train_mini; vs r058 (Hadamard+sandwich) iso-data iso-seed iso-hyper; closed-book holdout (restart: first run killed at step1400)" \
  --data_path /Volumes/pan/text/pretrain_train_mini.jsonl \
  --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8 \
  --kv_lora_rank 192 --mtp_depth 1 --seed 1337 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] knowledge_v2 queue done" >> "$LOG"
