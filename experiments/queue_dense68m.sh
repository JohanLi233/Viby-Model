# r080: 基础 dense 对照基线（无 MoE、带 engram），与 r070 HRM-Text-MoE
# 全面对齐：768x8、MLA(kv_lora 192, rope 32)、engram(layer0, orders 2,3,
# slots 8192, sub_dim 128)、value_res、attn_gate、MTP1、packed+docmask、
# bs6x2048 accum2、cache20。总参 67.52M（n_routed_experts=0 全 dense 含 MTP，
# dense FFN I=1792）。
# 用途：给 r060(MoE)/r070(HRM-MoE) 提供同数据同调度下的非 MoE 锚点——
# "MoE/HRM 到底值不值"的最终裁决基线。
# 依赖 muon.py 的空组剔除修复（纯 dense 无 router 参数，否则 MultiOptimizer
# 首个 step 抛 IndexError）。
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/r080_dense68m.log"
./experiments/run_exp.sh r080_dense68m \
  "dense 768x8 I=1792 67.5M + MLA + engram0 + vres + gate + MTP1，无MoE @bs6x2048 accum2 packed+docmask full epoch（r070的dense锚点）" \
  --hidden_size 768 --num_hidden_layers 8 --num_attention_heads 8 \
  --kv_lora_rank 192 --qk_rope_head_dim 32 \
  --n_routed_experts 0 --intermediate_size 1792 \
  --engram_layers 0 --engram_orders 2,3 --engram_slots 8192 --engram_sub_dim 128 \
  --use_swanlab \
  --pack_sequences --doc_mask \
  --batch_size 6 --accumulation_steps 2 --max_seq_len 2048 \
  --cache_limit_gb 20 --save_interval 500 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] r080 done" >> "$LOG"
