#!/usr/bin/env bash
# 稠密台 2000 微步确认（gap 3 收尾）：dense 500 步最优档 vs b05 对照，配对同 seed。
# 用法: bash experiments/run_dense_2k.sh <winner_coeff> <winner_ns_steps>
# 例: bash experiments/run_dense_2k.sh cubic5b005 6
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DATA=/Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl
WIN_COEFF="$1"; WIN_STEPS="$2"
STATUS=research_runs/dense_2k_status.tsv
[ -f "$STATUS" ] || printf 'probe\trc\tloss1900\n' > "$STATUS"

run_2k () {  # name coeff ns_steps
  NAME="$1"; COEFF="$2"; NSTEPS="$3"
  OUT="research_runs/$NAME"
  mkdir -p "$OUT"
  echo "=== $NAME coeff=$COEFF ns=$NSTEPS $(date +%H:%M) ==="
  VIBY_MUONH_NS_COEFF="$COEFF" PYTHONUNBUFFERED=1 \
  $PY trainer/train_pretrain.py \
    --out_dir "$OUT" \
    --epochs 1 --batch_size 12 --accumulation_steps 2 \
    --learning_rate 0.0015450949123618698 \
    --device mlx --dtype bfloat16 --compile_model --no_swanlab \
    --hidden_size 768 --vocab_size 6400 \
    --num_hidden_layers 12 --num_attention_heads 12 \
    --mtp_depth 0 --n_routed_experts 1 --num_experts_per_tok 1 \
    --n_shared_experts 0 --moe_intermediate_size 3072 --moe_latent_dim 0 \
    --max_seq_len 1024 --pack_sequences --doc_mask --use_attn_gate \
    --z_loss_weight 0.0001 --optimizer muon --muonh --muon_ns_steps "$NSTEPS" \
    --max_steps 2000 --max_train_minutes 55 --log_interval 100 \
    --save_interval 100000 --min_lr_ratio 0.05 --seed 1337 \
    --data_path "$DATA" \
    > "$OUT/console.log" 2>&1
  RC=$?
  L=$(grep -E '^Epoch:\[1/1\]\(1900/' "$OUT/console.log" | tail -1 | sed -E 's/.*loss:([0-9.]+).*/\1/')
  printf '%s\t%s\t%s\n' "$NAME" "$RC" "$L" >> "$STATUS"
  echo "=== $NAME rc=$RC loss@1900=$L ==="
}

run_2k probe_p40_dense_win_2k "$WIN_COEFF" "$WIN_STEPS"
run_2k probe_p41_dense_b05_2k cubic5b05 5
echo "=== dense 2k done $(date +%H:%M) ==="
