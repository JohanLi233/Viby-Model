# Loop mechanism search queue (2026-08-16 evening)
# Waits for r032 to finish, then:
#   1) H4 inference-unroll scan: r031(FiLM) k=1/2/3/4/6, r032(dW) k=1/2/4
#   2) r033 W-Scale-Loop  512x6 loop_k=2 --ws_loop 1  at 0.1B
#   3) r034 dW+W-Scale    512x6 loop_k=2 --dw_rank 8 --ws_loop 1 at 0.1B
# Decision rule: holdout CE diff > 0.05 nats vs r030/r031/r032 is credible;
# iso-FLOPs comparison: r030 at step 3200 equals loop series at step 1600.
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_loop_search.log"
SLICE='/Volumes/pan/text/pretrain_train_0.1B.jsonl'
HOLD='/Volumes/pan/text/pretrain_holdout_mini.jsonl'

echo "[$(date +%H:%M)] loop_search queue waiting for current training..." >> "$LOG"
while pgrep -f 'train_pretrain.py' > /dev/null; do
  sleep 30
done

run_h4() {
  local run="$1"; shift
  local ckpt="research_runs/${run}/pretrain_512.safetensors"
  for k in "$@"; do
    echo "[$(date +%H:%M)] H4 ${run} k=${k}" >> "$LOG"
    uv run experiments/eval_ppl.py       --ckpt "$ckpt" --data_path "$HOLD"       --tag "${run}_k${k}" --loop_k_override "$k"       --notes "H4 train_k=2 deploy_k=${k}; loop_k_override prefix-slice load"       --batch_size 16 2>&1 | grep -E '^\[ppl\] ' | tee -a "$LOG"
  done
}

run_h4 r031_loop28M_film_0.1B 1 2 3 4 6
run_h4 r032_loop28M_dw8_0.1B 1 2 4

echo "[$(date +%H:%M)] r033: W-Scale-Loop 512x6 x2 at 0.1B" >> "$LOG"
./experiments/run_exp.sh r033_loop28M_ws_0.1B   "W-Scale-Loop 512x6 x2 (ws_loop=1) at 0.1B; per-step diagonal weight scaling"   --data_path "$SLICE"   --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8   --kv_lora_rank 192 --loop_k 2 --ws_loop 1 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r034: dW r8 + W-Scale at 0.1B" >> "$LOG"
./experiments/run_exp.sh r034_loop28M_dw8_ws_0.1B   "loop 512x6 x2 dw_rank=8 + ws_loop=1 at 0.1B; dW and W-Scale combined"   --data_path "$SLICE"   --hidden_size 512 --num_hidden_layers 6 --num_attention_heads 8   --kv_lora_rank 192 --loop_k 2 --dw_rank 8 --ws_loop 1 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] loop_search queue done" >> "$LOG"
