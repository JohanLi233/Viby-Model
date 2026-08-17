# HRM candidate screening queue (new miracle-architecture search)
# Waits for the loop-search queue to fully finish, then runs two HRM arms on 0.1B.
#
# HRM core (aligned with HF HrmText): two interacting recurrent states
#   z_L <- L_stack(z_L + z_H), repeated L_cycles times
#   z_H <- H_stack(z_H + z_L), once per H cycle
# Training forward == inference forward; no inference-time extra unroll.
#
# r040: hidden=576, P=2 real layers/stack, H=1, L=2
#        -> real 4 layers, 6 layer-evals/token, N~26.2M
#        iso-FLOPs / iso-N-ish vs r030 dense 512x6 (6 evals, N=28.5M)
# r041: hidden=512, P=3, H=1, L=2
#        -> real 6 layers, 9 layer-evals/token, N=28.5M
#        exact iso-N vs r030; 1.5x compute per token, train=infer
#
# Both inherit run_exp recipe: pack+docmask+value_res+attn_gate+MTP1+Muon.
# Decision rule: holdout CE vs r030=3.5819; delta > 0.05 nats credible.
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_hrm.log"
SLICE='/Volumes/pan/text/pretrain_train_0.1B.jsonl'

echo "[$(date +%H:%M)] HRM queue waiting for loop_search queue..." >> "$LOG"
while pgrep -f 'queue_loop_search.sh' > /dev/null; do
  sleep 60
done
while pgrep -f 'train_pretrain.py' > /dev/null; do
  sleep 30
done

echo "[$(date +%H:%M)] r040: HRM 576x2x2 H=1 L=2 at 0.1B" >> "$LOG"
./experiments/run_exp.sh r040_hrm576_P2_L2_0.1B   "HRM 576 P=2 H=1 L=2 at 0.1B; 6 evals/token, N~26M; iso-FLOPs vs dense512x6"   --data_path "$SLICE"   --hidden_size 576 --num_hidden_layers 2 --num_attention_heads 8   --kv_lora_rank 192 --hrm_H_cycles 1 --hrm_L_cycles 2   --hrm_bp_cycles 2 --hrm_emb_scale 1.0 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r041: HRM 512x3x2 H=1 L=2 at 0.1B" >> "$LOG"
./experiments/run_exp.sh r041_hrm512_P3_L2_0.1B   "HRM 512 P=3 H=1 L=2 at 0.1B; 9 evals/token, N=28.5M; iso-N vs dense512x6"   --data_path "$SLICE"   --hidden_size 512 --num_hidden_layers 3 --num_attention_heads 8   --kv_lora_rank 192 --hrm_H_cycles 1 --hrm_L_cycles 2   --hrm_bp_cycles 2 --hrm_emb_scale 1.0 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] HRM queue done" >> "$LOG"
