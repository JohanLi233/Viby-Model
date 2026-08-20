# r073: HRM-Text-MoE v2 —— CycleDeltaRouter + engram 初始 z_H 注入
#
# 路由稳定性默认配置：router noise=0.05 + soft aux loss=0.001 +
# per-slot bias rate=0.02；noise=0.1 更稳但早期 loss 下降更慢。
# cycle_delta_max 默认 0（探针中 clamp 反而加重 MTP 集中，保留实验开关）。
# tied embedding 默认不缩放 logits（SCALE_LOGITS=1 可开启初始化诊断，
# 但会显著放慢早期 loss 下降）。
# engram 固定使用 DeepSeek 论文融合口径：先 sigmoid 门控，再
# RMSNorm→短因果卷积→SiLU，无硬截断。
#
# 用法：
#   ./experiments/queue_hrm_moe_cycledelta.sh              # 全量 1 epoch
#   CYCLE_RANK=16 ./experiments/queue_hrm_moe_cycledelta.sh
#   ENGRAM_EVERY_CYCLE=1 ./experiments/queue_hrm_moe_cycledelta.sh   # engram 旧行为消融
#   MAX_TRAIN_MINUTES=30 ./experiments/queue_hrm_moe_cycledelta.sh   # 30 分钟探针
#   USE_SWANLAB=0 MAX_STEPS=400 ./experiments/queue_hrm_moe_cycledelta.sh  # 本地探针（不上报）
set -u
cd "$(dirname "$0")/.."

ROUND="${ROUND:-r073_hrm_moe_cycledelta}"
CYCLE_RANK="${CYCLE_RANK:-8}"
LR="${LR:-0.01}"
BIAS_RATE="${BIAS_RATE:-0.02}"
ROUTER_NOISE="${ROUTER_NOISE:-0.05}"
AUX_WEIGHT="${AUX_WEIGHT:-0.001}"
CYCLE_DELTA_MAX="${CYCLE_DELTA_MAX:-0}"
HRM_STATE_NORM="${HRM_STATE_NORM:-1}"
HRM_INPUT_SKIP="${HRM_INPUT_SKIP:-0.0}"
HRM_TOKEN_GATE="${HRM_TOKEN_GATE:-0.1}"
MOE_LOGIT_NORM="${MOE_LOGIT_NORM:-1}"
MOE_LOGIT_TEMP="${MOE_LOGIT_TEMP:-1.0}"
DIVERSITY_LOSS="${DIVERSITY_LOSS:-0.01}"
ENGRAM_SCALE="${ENGRAM_SCALE:-1.0}"
ENGRAM_LR_MULT="${ENGRAM_LR_MULT:-1.0}"
SCALE_LOGITS="${SCALE_LOGITS:-0}"
BS="${BS:-6}"
ACCUM="${ACCUM:-2}"
SEQ="${SEQ:-2048}"
CACHE_GB="${CACHE_GB:-0}"

# E=112 细粒度路由的稳定性配置（r070 探针 C/D 标定）。
export MOE_ROUTER_LR_MULT="${MOE_ROUTER_LR_MULT:-0.01}"
export MOE_CYCLE_ROUTER_LR_MULT="${MOE_CYCLE_ROUTER_LR_MULT:-0.1}"
# 排查吞吐/负载漂移时打开：
# export VIBY_DEBUG_MEM=1

EXTRA=()
if [[ "${ENGRAM_EVERY_CYCLE:-0}" == "1" ]]; then
    EXTRA+=(--engram_inject_every_cycle)
fi
if [[ "${SCALE_LOGITS}" == "0" ]]; then
    EXTRA+=(--no-scale_logits_by_emb_scale)
fi
if [[ "${HRM_STATE_NORM}" == "0" ]]; then
    EXTRA+=(--no-hrm_state_norm)
fi
if [[ "${MOE_LOGIT_NORM}" == "0" ]]; then
    EXTRA+=(--no-moe_router_logit_norm)
fi
if [[ "${USE_SWANLAB:-1}" == "1" ]]; then
    EXTRA+=(--use_swanlab)
else
    EXTRA+=(--no_swanlab)
fi
if [[ -n "${MAX_TRAIN_MINUTES:-}" ]]; then
    EXTRA+=(--max_train_minutes "$MAX_TRAIN_MINUTES")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
    EXTRA+=(--max_steps "$MAX_STEPS")
fi

./experiments/run_exp.sh "$ROUND" \
  "HRM-MoE v2(H2L3P1=8evals) CycleDeltaRouter(rank${CYCLE_RANK}) + engram initial-zH + 112x104 top6+1sh + FiLM + vres + gate + MTP1 + bias${BIAS_RATE}/routerlr${MOE_ROUTER_LR_MULT}/cyclerlr${MOE_CYCLE_ROUTER_LR_MULT} @bs${BS}xseq${SEQ} accum${ACCUM} packed+docmask" \
  --hidden_size 768 --num_hidden_layers 1 --num_attention_heads 8 \
  --kv_lora_rank 192 --qk_rope_head_dim 32 \
  --learning_rate "$LR" \
  --n_routed_experts 112 --num_experts_per_tok 6 --n_shared_experts 1 \
  --moe_intermediate_size 104 --routed_scaling_factor 2.5 \
  --hrm_H_cycles 2 --hrm_L_cycles 3 --hrm_bp_cycles 2 \
  --hrm_emb_scale 27.7128 --hrm_input_skip "$HRM_INPUT_SKIP" \
  --hrm_token_gate_scale "$HRM_TOKEN_GATE" \
  --hrm_cycle_router 1 --hrm_cycle_router_rank "$CYCLE_RANK" \
  --hrm_cycle_film 1 --cycle_router_lr_mult "$MOE_CYCLE_ROUTER_LR_MULT" \
  --moe_bias_update_rate "$BIAS_RATE" \
  --moe_router_noise "$ROUTER_NOISE" \
  --moe_aux_loss_weight "$AUX_WEIGHT" \
  --moe_router_logit_temp "$MOE_LOGIT_TEMP" \
  --moe_diversity_loss_weight "$DIVERSITY_LOSS" \
  --cycle_delta_max "$CYCLE_DELTA_MAX" \
  --engram_layers 0 --engram_orders 2,3 --engram_slots 8192 --engram_sub_dim 128 \
  --engram_scale "$ENGRAM_SCALE" \
  --engram_lr_mult "$ENGRAM_LR_MULT" \
  --pack_sequences --doc_mask \
  --batch_size "$BS" --accumulation_steps "$ACCUM" --max_seq_len "$SEQ" \
  --cache_limit_gb "$CACHE_GB" --save_interval 500 \
  ${EXTRA[@]+"${EXTRA[@]}"}
