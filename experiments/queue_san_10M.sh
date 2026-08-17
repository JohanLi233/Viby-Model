# SAN 10M screening queue (Cactus Needle / arXiv:2607.18363)
# Waits for dense baseline r045, then runs:
#   r046: SAN pure attention-only  h=384 L=44  @10M
#   r047: SAN + HadamardMLP      h=384 L=44  @10M
# Both: N~28.5M, ffn_type controls mixer, ZCN + scalar res gate + o_proj
# depth init, emb_scale=0.02*hidden (Needle init equivalence), lr=0.02 (paper Muon).
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_san_10M.log"
SLICE='/Volumes/pan/text/pretrain_train_10M.jsonl'

echo "[$(date +%H:%M)] waiting for dense baseline..." >> "$LOG"
while pgrep -f 'train_pretrain.py.*r045_dense512x6_10M' > /dev/null; do sleep 30; done

echo "[$(date +%H:%M)] r046 SAN none @10M" >> "$LOG"
./experiments/run_exp.sh r046_san384x44_none_10M \
  "SAN ffn=none h=384 L=44 N=28.45M @10M full; ZCN+resgate+san_init emb_scale=7.68 lr0.02" \
  --data_path "$SLICE" --hidden_size 384 --num_hidden_layers 44 --num_attention_heads 8 \
  --kv_lora_rank 160 --qk_rope_head_dim 32 --mtp_depth 0 --ffn_type none \
  --zero_centered_norm --use_res_gate --san_res_init --emb_scale 7.68 \
  --learning_rate 0.02 --no_value_res --no_attn_gate >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r047 SAN hadamard @10M" >> "$LOG"
./experiments/run_exp.sh r047_san384x44_had_10M \
  "SAN ffn=hadamard h=384 L=44 N=28.52M @10M full; same SAN recipe + fixed Hadamard channel mixer" \
  --data_path "$SLICE" --hidden_size 384 --num_hidden_layers 44 --num_attention_heads 8 \
  --kv_lora_rank 160 --qk_rope_head_dim 32 --mtp_depth 0 --ffn_type hadamard \
  --zero_centered_norm --use_res_gate --san_res_init --emb_scale 7.68 \
  --learning_rate 0.02 --no_value_res --no_attn_gate >> "$LOG" 2>&1

echo "[$(date +%H:%M)] SAN 10M queue done" >> "$LOG"
