# Engram-SAN 10M screening arm (B1 knowledge-memory test)
# Wait for queue_san_10M (r046 SAN-pure, r047 SAN+Hadamard) to finish, then:
#   r048: SAN + Engram   h=384 L=40, engram sites (2,20), slots=4096
#         N~28.58M (iso-N vs dense 28.5M and SAN-pure 28.45M)
# Compare against r045 dense 512x6 @10M CE and r046 SAN-pure.
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_engram_10M.log"
SLICE='/Volumes/pan/text/pretrain_train_10M.jsonl'

echo "[$(date +%H:%M)] waiting for SAN queue..." >> "$LOG"
while pgrep -f 'queue_san_10M.sh' > /dev/null; do sleep 30; done
while pgrep -f 'train_pretrain.py' > /dev/null; do sleep 30; done

echo "[$(date +%H:%M)] r048 SAN+Engram @10M" >> "$LOG"
./experiments/run_exp.sh r048_san384x40_engram_10M \
  "SAN h384 L40 + Engram sites(2,20) slots4096 orders(2,3) N=28.58M @10M; B1 knowledge-memory probe" \
  --data_path "$SLICE" --hidden_size 384 --num_hidden_layers 40 --num_attention_heads 8 \
  --kv_lora_rank 160 --qk_rope_head_dim 32 --mtp_depth 0 --ffn_type none \
  --zero_centered_norm --use_res_gate --san_res_init --emb_scale 7.68 \
  --learning_rate 0.02 --no_value_res --no_attn_gate \
  --engram_layers 2,20 --engram_orders 2,3 --engram_slots 4096 --engram_sub_dim 128 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] engram queue done" >> "$LOG"
