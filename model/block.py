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

try:
    from .kernels.hc_pre_norm import enabled_for as _hc_pre_norm_ok
    from .kernels.hc_pre_norm import hc_pre_norm as _hc_pre_norm
except Exception:  # noqa: BLE001
    _hc_pre_norm = None

    def _hc_pre_norm_ok(_x):
        return False


def apply_hc_pre_norm(x, pre_mix, norm):
    """hc_pre + RMSNorm。hc_mult=4 且 GPU 时走融合核，否则回退两步实现。"""
    if _hc_pre_norm is not None and _hc_pre_norm_ok(x):
        return _hc_pre_norm(x, pre_mix, norm.weight, norm.eps)
    return norm(hc_pre(x, pre_mix))


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
            config.dim,
            config.hc_mult,
            config.hc_sinkhorn_iters,
            config.hc_eps,
            config.norm_eps,
        )
        self.ffn_hc = HyperConnection(
            config.dim,
            config.hc_mult,
            config.hc_sinkhorn_iters,
            config.hc_eps,
            config.norm_eps,
        )

    def __call__(
        self,
        x,
        start_pos: int,
        pre_mix,
        shared,
        cache=None,
        segment_ids=None,
        pad_mask=None,
        decode: bool = False,
        attention_injection=None,
    ):
        """x: [B,T,hc,d] → (x, 下一个子层要用的 pre_mix)。"""
        residual = x
        attn_pre, attn_post, attn_comb = self.attn_hc.mixes(x)
        h = apply_hc_pre_norm(x, pre_mix, self.attn_norm)
        if decode:
            h = self.attn.decode(h, start_pos, shared, cache)
        else:
            h = self.attn(h, start_pos, shared, cache, segment_ids, pad_mask)
        if attention_injection is not None:
            h = (h.astype(attention_injection.dtype) + attention_injection).astype(h.dtype)
        x = hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.mixes(x)
        h = apply_hc_pre_norm(x, attn_pre, self.ffn_norm)
        h = self.ffn(h)
        x = hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def recurrent(
        self,
        x,
        pre_mix,
        shared,
        *,
        query_positions,
        memory_positions,
        query_segment_ids=None,
        memory_segment_ids=None,
        query_pad_mask=None,
        memory_pad_mask=None,
        cache=None,
    ):
        """Run one physical recurrent CED stage, preserving mHC sublayer order.

        The caller owns a separate self-KV ``cache`` for every (round, layer).
        Parameters and boundary evidence are shared, while ``pre_mix`` and the
        sparse selection in ``shared`` travel with the updated latent hidden.
        """
        residual = x
        attn_pre, attn_post, attn_comb = self.attn_hc.mixes(x)
        h = apply_hc_pre_norm(x, pre_mix, self.attn_norm)
        h = self.attn.recurrent(
            h,
            shared,
            query_positions=query_positions,
            memory_positions=memory_positions,
            query_segment_ids=query_segment_ids,
            memory_segment_ids=memory_segment_ids,
            query_pad_mask=query_pad_mask,
            memory_pad_mask=memory_pad_mask,
            cache=cache,
        )
        x = hc_post(h, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.mixes(x)
        h = apply_hc_pre_norm(x, attn_pre, self.ffn_norm)
        h = self.ffn(h, pad_mask=query_pad_mask)
        x = hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre
