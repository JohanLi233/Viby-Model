#!/usr/bin/env bash
# 论文 gap 补全 probe 队列（P31–P39）。同一时刻只跑一个训练任务。
# 每个 run：500 微步（--max_steps 500，微批口径）≈14 分钟，seed/系数见表。
# 口径：全部 post-fix 新代码帧；与 P19+ 数字可比，不与 P13–P17 跨减。
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DATA=/Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl
STATUS=research_runs/gap_queue_status.tsv
[ -f "$STATUS" ] || printf 'probe\trc\tloss500\n' > "$STATUS"

run_probe () {  # name coeff ns_steps seed dense(0/1)
  NAME="$1"; COEFF="$2"; NSTEPS="$3"; SEED="$4"; DENSE="$5"
  OUT="research_runs/$NAME"
  mkdir -p "$OUT"
  MODEL_ARGS="--hidden_size 768 --vocab_size 6400"
  if [ "$DENSE" = 1 ]; then
    MODEL_ARGS="$MODEL_ARGS --num_hidden_layers 12 --num_attention_heads 12 \
      --mtp_depth 0 --n_routed_experts 1 --num_experts_per_tok 1 \
      --n_shared_experts 0 --moe_intermediate_size 3072 --moe_latent_dim 0"
  else
    MODEL_ARGS="$MODEL_ARGS --num_hidden_layers 8 --num_attention_heads 8 \
      --mtp_depth 1 --mtp_loss_weight 0.3 --n_routed_experts 256 \
      --num_experts_per_tok 8 --n_shared_experts 2 --moe_intermediate_size 384 \
      --routed_scaling_factor 2.5 --moe_router_logit_norm"
  fi
  echo "=== $NAME coeff=$COEFF ns=$NSTEPS seed=$SEED dense=$DENSE $(date +%H:%M) ==="
  VIBY_MUONH_NS_COEFF="$COEFF" PYTHONUNBUFFERED=1 \
  $PY trainer/train_pretrain.py \
    --out_dir "$OUT" \
    --epochs 1 --batch_size 12 --accumulation_steps 2 \
    --learning_rate 0.0015450949123618698 \
    --device mlx --dtype bfloat16 --compile_model --no_swanlab \
    $MODEL_ARGS \
    --max_seq_len 1024 --pack_sequences --doc_mask --use_attn_gate \
    --z_loss_weight 0.0001 --optimizer muon --muonh --muon_ns_steps "$NSTEPS" \
    --max_steps 550 --max_train_minutes 30 --log_interval 50 \
    --save_interval 100000 --min_lr_ratio 0.05 --seed "$SEED" \
    --data_path "$DATA" \
    > "$OUT/console.log" 2>&1
  RC=$?
  L500=$(grep -E '^Epoch:\[1/1\]\(500/' "$OUT/console.log" | tail -1 | sed -E 's/.*loss:([0-9.]+).*/\1/')
  printf '%s\t%s\t%s\n' "$NAME" "$RC" "$L500" >> "$STATUS"
  echo "=== $NAME rc=$RC loss@500=$L500 ==="
}

# Gap 1: post-fix U 曲线（MoE，seed 1337）
run_probe probe_p31_classic     classic    5 1337 0
run_probe probe_p32_cubic5      cubic5     5 1337 0
run_probe probe_p33_cubic5b10   cubic5b10  5 1337 0
# Gap 3: 稠密台膝盖网格加蜜（dense，seed 1337）
run_probe probe_p34_dense_b01   cubic5b01  5 1337 1
run_probe probe_p35_dense_b005  cubic5b005 6 1337 1
# Gap 2: 双臂多种子（b05 的 1338/1339 已有 P25a/b）
run_probe probe_p36_b002_s1338  cubic5b002 7 1338 0
run_probe probe_p37_b002_s1339  cubic5b002 7 1339 0
run_probe probe_p38_b10_s1338   cubic5b10  5 1338 0
run_probe probe_p39_b10_s1339   cubic5b10  5 1339 0
echo "=== queue done $(date +%H:%M) ==="
