"""Candidate architecture per-token FLOPs proxy and compute-normalized comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math import log2
from model.model import VibyConfig, VibyForCausalLM
from mlx.utils import tree_flatten

CTX = 640

def params(cfg):
    m = VibyForCausalLM(cfg)
    return sum(v.size for _, v in tree_flatten(m.trainable_parameters())) / 1e6

def flops_proxy(cfg):
    n = params(cfg)
    emb = cfg.vocab_size * cfg.hidden_size / 1e6
    nonemb = n - emb
    hrm = getattr(cfg, 'hrm_H_cycles', 0) > 0
    if hrm:
        L_eff = cfg.hrm_H_cycles * (cfg.hrm_L_cycles + 1) * cfg.num_hidden_layers
    else:
        L_eff = max(1, getattr(cfg, 'loop_k', 1) or 1) * cfg.num_hidden_layers
    mat = 2 * nonemb * 1e6
    att = 4 * L_eff * CTX * cfg.hidden_size
    had = 0.0
    if getattr(cfg, 'ffn_type', 'swiglu') == 'hadamard':
        n2 = 1 << (cfg.hidden_size - 1).bit_length()
        had = 2 * n2 * log2(n2) * L_eff
    total = mat + att + had
    return dict(N=n, nonemb=nonemb, L_eff=L_eff, matmul=mat/1e6, attn=att/1e6,
                hadamard=had/1e6, total=total/1e6)

def base():
    return VibyConfig(hidden_size=512, num_hidden_layers=6, num_attention_heads=8,
                      kv_lora_rank=192, qk_rope_head_dim=32, vocab_size=6400,
                      max_position_embeddings=640, mtp_depth=1,
                      use_value_res=True, use_attn_gate=True)

def san(L=44, ffn='none', engram=False):
    return VibyConfig(hidden_size=384, num_hidden_layers=L, num_attention_heads=8,
                      kv_lora_rank=160, qk_rope_head_dim=32, vocab_size=6400,
                      max_position_embeddings=640, mtp_depth=0, ffn_type=ffn,
                      zero_centered_norm=1, use_res_gate=1, san_res_init=1,
                      engram_layers=((2,20) if engram else ()), engram_orders=(2,3),
                      engram_slots=4096, engram_sub_dim=128)

def dense_had(h=512, L=6):
    return VibyConfig(hidden_size=h, num_hidden_layers=L, num_attention_heads=8,
                      kv_lora_rank=192, qk_rope_head_dim=32, vocab_size=6400,
                      max_position_embeddings=640, mtp_depth=1,
                      use_value_res=True, use_attn_gate=True, ffn_type='hadamard')

if __name__ == '__main__':
    b = flops_proxy(base())
    rows = [
        ('dense512x6', b, 6.0482, 10),
        ('SAN pure', flops_proxy(san(44)), 6.1485, 10),
        ('SAN+Had', flops_proxy(san(44, 'hadamard')), 6.0613, 10),
        ('SAN+Engram(susp)', flops_proxy(san(40, 'none', True)), None, 10),
        ('denseHad512x6 r049', flops_proxy(dense_had(512, 6)), None, 34),
        ('denseHad896x8 r051', flops_proxy(dense_had(896, 8)), None, 10),
    ]
    print(f"{'candidate':22s} {'N_M':>6s} {'L_eff':>5s} {'FLOPs/tok(M)':>12s} {'ratio':>6s}")
    for name, r, ce, D in rows:
        print(f"{name:22s} {r['N']:6.2f} {r['L_eff']:5d} {r['total']:12.1f} {r['total']/b['total']:6.2f}")
    print(f"\nbaseline proxy: matmul={b['matmul']:.1f}M attn={b['attn']:.1f}M total={b['total']:.1f}M")
    print('CE deltas vs dense@10M: SAN pure +0.100, SAN+Had +0.013')
