"""V4.1 的 Block：mHC 残差流 + 混合注意力 + MoE。

与参考实现一致的两个细节：
1. 子层系数（pre/post/comb）由 hc_mixes 现算，且"本子层算出的系数，
   下一个子层用"——attention 用上一层 FFN 产出的 pre_mix 收敛输入，
   FFN 用本层 attention 产出的 pre_mix。
2. Engram 在层入口写残差流（由模型循环调用），不参与残差混合系数。
"""

from mlx import nn

from .attention import Attention
from .hc import HyperConnection, hc_post, hc_pre
from .moe import MoEFeedForward
from .norms import RMSNorm


class Block(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = config.dim
        self.attn = Attention(config, layer_idx)
        self.ffn = MoEFeedForward(config, layer_idx)
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn_hc = HyperConnection(
            config.dim, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps, config.norm_eps
        )
        self.ffn_hc = HyperConnection(
            config.dim, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps, config.norm_eps
        )

    def __call__(self, x, start_pos: int, pre_mix, shared, cache=None,
                 segment_ids=None, pad_mask=None, decode: bool = False):
        """x: [B,T,hc,d] → (x, 下一个子层要用的 pre_mix)。"""
        residual = x
        attn_pre, attn_post, attn_comb = self.attn_hc.mixes(x)
        h = hc_pre(x, pre_mix)
        h = self.attn_norm(h)
        if decode:
            h = self.attn.decode(h, start_pos, shared, cache)
        else:
            h = self.attn(h, start_pos, shared, cache, segment_ids, pad_mask)
        x = hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.mixes(x)
        h = hc_pre(x, attn_pre)
        h = self.ffn_norm(h)
        h = self.ffn(h)
        x = hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre
