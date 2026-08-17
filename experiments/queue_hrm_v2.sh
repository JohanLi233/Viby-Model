# HRM v2 queue: corrected operating points after review
# Waits for queue_hrm (r040/r041 degenerate controls) to finish.
#
# Review corrections:
#   1) embedding_scale: HF 50 assumes embedding init std 0.02; MLX embedding
#      init std = 1/sqrt(hidden). Equivalent scale = sqrt(hidden).
#   2) hierarchy: H=2 with z_L inherited across H cycles (code already does
#      this, matching HF); H=1 cannot express the slow/fast interaction.
#
# r042: faithful HF operating point at iso-N vs r030 dense 512x6
#       hidden=768, P=1, H=2, L=3, bp=2, emb_scale=sqrt(768)=27.7128
#       -> N=28.46M, 8 layer-evals/token, ~3.0x r030 per-token FLOPs
# r043: minimal hierarchy / fast-inference compromise
#       hidden=592, P=2, H=2, L=1, bp=1, emb_scale=sqrt(592)=24.3311
#       -> N=27.63M, 8 layer-evals/token, ~1.78x r030 per-token FLOPs
#
# Both train=infer same forward. Decision vs r030 holdout CE 3.5819.
set -u
cd "$(dirname "$0")/.."
LOG="experiments/logs/queue_hrm_v2.log"
SLICE='/Volumes/pan/text/pretrain_train_0.1B.jsonl'

echo "[$(date +%H:%M)] HRM v2 waiting for queue_hrm..." >> "$LOG"
while pgrep -f 'queue_hrm.sh' > /dev/null; do
  sleep 60
done
while pgrep -f 'train_pretrain.py' > /dev/null; do
  sleep 30
done

echo "[$(date +%H:%M)] r042: faithful HRM H=2 L=3 P=1 hidden=768 at 0.1B" >> "$LOG"
./experiments/run_exp.sh r042_hrm768_P1_H2L3_0.1B   "faithful HRM H=2 L=3 P=1 h=768 emb_scale=27.71 bp=2 at 0.1B; iso-N vs dense512x6"   --data_path "$SLICE"   --hidden_size 768 --num_hidden_layers 1 --num_attention_heads 8   --kv_lora_rank 192 --hrm_H_cycles 2 --hrm_L_cycles 3   --hrm_bp_cycles 2 --hrm_emb_scale 27.712812921102035 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] r043: minimal-hierarchy HRM H=2 L=1 P=2 hidden=592 at 0.1B" >> "$LOG"
./experiments/run_exp.sh r043_hrm592_P2_H2L1_0.1B   "HRM H=2 L=1 P=2 h=592 emb_scale=24.33 bp=1 at 0.1B; fast hierarchy probe"   --data_path "$SLICE"   --hidden_size 592 --num_hidden_layers 2 --num_attention_heads 8   --kv_lora_rank 192 --hrm_H_cycles 2 --hrm_L_cycles 1   --hrm_bp_cycles 1 --hrm_emb_scale 24.331050121192877 >> "$LOG" 2>&1

echo "[$(date +%H:%M)] HRM v2 queue done" >> "$LOG"
