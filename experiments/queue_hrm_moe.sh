# r070: HRM-Text-MoE 全量 pretrain（/Volumes/pan/text 全量数据）
# 设计见 research/HRM_MOE.md：HRM 双状态循环（H=2, L=3, P=1 → 8 次 stack
# 求值/token，≈r060 的 8 层）+ L/H/MTP 全 MoE（112 路由 top-6 + 1 共享）
# + CycleDeltaRouter（per-cycle 低秩路由增量，破循环引理包含上界）
# + CycleFiLM（per-cycle scale/shift）+ engram(L 栈 layer0) + value_res
# + attn_gate + MTP(1)。总参数 100.78M，激活 ≈33M/token —— 与 r060
# （99.7M / 33.5M）iso-N、iso-激活、iso-每token层求值，干净对照。
# hrm_emb_scale=√768=27.7128（r042 的忠实 HF 操作点）；bp_cycles=2（尾部 2 个
# L cycle 回传，反向驻留减半）。
# 优化器/调度/数据口径与 r060 完全一致（bs6 accum2 packed+docmask
# seq2048 cache20 save500；router 自动走 muon.py 的小 lr AdamW 组，
# CycleDelta U/V_c 含 ".router." 走独立 AdamW 组；hrm_film 归 AdamW 标量组）。
# ⚠ 路由均衡必须加强（2026-08-17 探针实锤，/tmp/diag_hrm{A..D}）：
# E=112 下三个 router（L/H/MTP）的 top-1 桶容量 C 在 bias=0.001 +
# router_lr=0.05×（r060@E=32 的稳定点）下全部单调塌缩（6.7K→22K+/160步），
# (E,C,D) 桶缓冲把单步峰值顶到 52G>48G → swap、吞吐 14.8K→3K。与
# CycleRouter 无关（关闭对照臂同样塌）。修复：bias 0.005 +
# router_lr 0.01× —— 峰值钉死 38.7G、C 稳态 3-4K（尖峰 10K 会被拉回）、
# 瞬时吞吐 12.1K 持平 r060、swap 不增。
set -u
cd "$(dirname "$0")/.."
export MOE_ROUTER_LR_MULT=0.01
LOG="experiments/logs/r070_hrm_moe.log"
./experiments/run_exp.sh r070_hrm_moe \
  "HRM-Text-MoE(H2L3P1=8evals) + 112x104 top6+1sh 全栈MoE + CycleRouter+CycleFiLM + engram0 + vres + gate + MTP1 + bias0.005/routerlr0.01x @bs6x2048 accum2 packed+docmask full epoch" \
  --hidden_size 768 --num_hidden_layers 1 --num_attention_heads 8 \
  --kv_lora_rank 192 --qk_rope_head_dim 32 \
  --n_routed_experts 112 --num_experts_per_tok 6 --n_shared_experts 1 \
  --moe_intermediate_size 104 --routed_scaling_factor 2.5 \
  --hrm_H_cycles 2 --hrm_L_cycles 3 --hrm_bp_cycles 2 \
  --hrm_emb_scale 27.7128 --hrm_cycle_router 1 --hrm_cycle_router_rank 8 --hrm_cycle_film 1 \
  --moe_bias_update_rate 0.005 \
  --use_swanlab \
  --engram_layers 0 --engram_orders 2,3 --engram_slots 8192 --engram_sub_dim 128 \
  --pack_sequences --doc_mask \
  --batch_size 6 --accumulation_steps 2 --max_seq_len 2048 \
  --cache_limit_gb 20 --save_interval 500 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] r070 done" >> "$LOG"
