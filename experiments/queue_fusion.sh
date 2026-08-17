# r067: HRM x Engram x (small-N denseHad + sandwich) fusion, full 343M epoch.
# Config: h=384 P=2 per stack, H=2 L=2 -> 12 layer-evals/token (~2x serial vs 384x6 dense);
# Hadamard FFN absorbs per-eval FFN cost; engram-v2 (zero-init value_proj) into L-stack layer 1;
# emb_scale=sqrt(384)=19.6 (HRM operating point, cf. HRM_CANDIDATE.md).
# Gate: holdout CE vs r058 denseHad512x6+sand @343M = 2.2761 (N=10.6M, 0.39x FLOPs/tok;
# fusion N~9.6M, ~0.5-0.6x). Kill if it stalls or OOMs early (r048 precedent).
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_fusion.log"

echo "[$(date +%H:%M)] r067 HRM+Engram+Hadamard+sandwich fusion @343M" >> "$LOG"
./experiments/run_exp.sh r067_hrm384x2H2L2_had_sand_eng_full \
  "HRM(384,P2,H2,L2,emb_scale19.6)+hadamard+sandwich+engram-v2(site1,orders123,slots8192,sub64) @343M full; fusion probe vs r058 CE 2.2761" \
  --hidden_size 384 --num_hidden_layers 2 --num_attention_heads 8 \
  --kv_lora_rank 160 --hrm_H_cycles 2 --hrm_L_cycles 2 --hrm_emb_scale 19.6 \
  --ffn_type hadamard --sandwich_norm --mtp_depth 1 \
  --engram_layers 1 --engram_orders 1,2,3 --engram_slots 8192 --engram_sub_dim 64 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] fusion queue done" >> "$LOG"
