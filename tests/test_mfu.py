"""训练 MFU：6N_active（排除 embedding 查找）+ softmax 注意力二次项 + MTP 展开。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))


import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from mlx.utils import tree_flatten

from model.config import VibyConfig
from model.flops import (
    DEFAULT_PEAK_TFLOPS,
    attn_fwdbwd_flops_per_token,
    gemm_active_params,
    model_flops_utilization,
    training_flops_per_token,
)
from model.model import VibyForCausalLM
from trainer.utils import log_training_progress


def _tiny(**kw):
    base = dict(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        vocab_size=256,
        max_position_embeddings=256,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        mtp_depth=0,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=48,
        moe_latent_dim=0,
        ngram_table_size=0,
        n_shared_experts=1,
    )
    base.update(kw)
    return VibyConfig(**base)


class MfuArithmeticTest(unittest.TestCase):
    def test_perfect_peak_is_100_percent(self):
        flops_per_token = 6e8
        peak = 13.5
        tokens = peak * 1e12 / flops_per_token
        self.assertAlmostEqual(
            model_flops_utilization(tokens, flops_per_token, peak), 1.0
        )

    def test_zero_throughput_is_zero_mfu(self):
        self.assertEqual(model_flops_utilization(0.0, 1e9, 13.5), 0.0)


class TrainingFlopsTest(unittest.TestCase):
    def test_six_n_excludes_embedding_lookup(self):
        model = VibyForCausalLM(_tiny())
        gemm = gemm_active_params(model)
        embed = int(model.model.embed_tokens.weight.size)
        buffers = 0
        for p, v in tree_flatten(model.parameters()):
            if p.rsplit(".", 1)[-1] in {
                "expert_bias",
                "freqs_cos",
                "freqs_sin",
                "rope_freqs",
            }:
                buffers += v.size
        active = model.num_active_parameters()
        self.assertEqual(gemm, active - embed - buffers)
        self.assertLess(gemm, active)

    def test_moe_topk_scales_routed_expert_flops(self):
        low = VibyForCausalLM(_tiny(num_experts_per_tok=2))
        high = VibyForCausalLM(_tiny(num_experts_per_tok=8))
        self.assertGreater(
            gemm_active_params(high) - gemm_active_params(low),
            0,
        )
        routed_low = 0
        for p, v in tree_flatten(low.parameters()):
            if ".experts." in p and v.ndim >= 3:
                routed_low += v.size // v.shape[0] * 2
        routed_high = routed_low // 2 * 8
        self.assertEqual(
            gemm_active_params(high) - gemm_active_params(low),
            routed_high - routed_low,
        )

    def test_mtp_steps_multiply_extra_block(self):
        once = VibyForCausalLM(_tiny(mtp_depth=1, mtp_steps=1))
        twice = VibyForCausalLM(_tiny(mtp_depth=1, mtp_steps=2))
        self.assertGreater(gemm_active_params(twice), gemm_active_params(once))
        mtp_once = 0
        lm = 0
        for p, v in tree_flatten(once.parameters()):
            if "mtp_modules" in p:
                if ".experts." in p and v.ndim >= 3:
                    e = int(v.shape[0])
                    k = once.config.num_experts_per_tok
                    mtp_once += v.size // e * min(k, e)
                elif p.rsplit(".", 1)[-1] not in {
                    "expert_bias",
                    "freqs_cos",
                    "freqs_sin",
                    "rope_freqs",
                }:
                    mtp_once += v.size
            elif p.startswith("lm_head"):
                lm += v.size
        self.assertEqual(
            gemm_active_params(twice) - gemm_active_params(once),
            mtp_once + lm,
        )

    def test_attention_term_grows_linearly_with_seq_len(self):
        model = VibyForCausalLM(_tiny())
        f32 = training_flops_per_token(model, 32)
        f64 = training_flops_per_token(model, 64)
        delta = attn_fwdbwd_flops_per_token(
            model.config, 64
        ) - attn_fwdbwd_flops_per_token(model.config, 32)
        self.assertGreater(delta, 0)
        self.assertEqual(f64 - f32, delta)

    def test_mla_attention_uses_rope_qk_and_v_dims(self):
        cfg = _tiny()
        # 2 层 MLA：fwd+bwd = 6 * H * (qk + v) * T * L
        qk = cfg.head_dim + cfg.qk_rope_head_dim
        v = cfg.head_dim
        expect = 2 * 6 * cfg.num_attention_heads * (qk + v) * 64
        self.assertEqual(attn_fwdbwd_flops_per_token(cfg, 64), expect)

    def test_linear_attn_only_counts_gqa_quadratic(self):
        mla = _tiny(num_hidden_layers=4)
        lin = _tiny(num_hidden_layers=4, use_linear_attn=True)
        # 4 层线性：global = 第 4 层（最后一层），仅 1 个 GQA
        self.assertGreater(
            attn_fwdbwd_flops_per_token(mla, 64),
            attn_fwdbwd_flops_per_token(lin, 64),
        )
        hd = lin.head_dim
        expect = 1 * 6 * lin.num_attention_heads * (hd + hd) * 64
        self.assertEqual(attn_fwdbwd_flops_per_token(lin, 64), expect)

    def test_training_flops_is_6n_plus_attention(self):
        model = VibyForCausalLM(_tiny(mtp_depth=1, mtp_steps=2))
        T = 48
        self.assertEqual(
            training_flops_per_token(model, T),
            6 * gemm_active_params(model)
            + attn_fwdbwd_flops_per_token(model.config, T),
        )

    def test_ngram_table_not_in_active_or_gemm(self):
        on = VibyForCausalLM(_tiny(ngram_table_size=2048))
        off = VibyForCausalLM(_tiny(ngram_table_size=0))
        table = int(on.model.ngram.table.size)
        self.assertEqual(on.ngram_lookup_parameters(), table)
        # 表已从 6N 剔除；其余 n-gram 投影/门控参数仍计入（与 num_active 一致）
        ngram_total = sum(
            int(v.size) for _, v in tree_flatten(on.model.ngram.parameters())
        )
        expect = ngram_total - table
        self.assertEqual(
            gemm_active_params(on) - gemm_active_params(off),
            expect,
        )
        self.assertEqual(
            on.num_active_parameters() - off.num_active_parameters(),
            expect,
        )


class LogTrainingProgressMfuTest(unittest.TestCase):
    def test_log_line_includes_mfu_percent(self):
        args = SimpleNamespace(
            epochs=1,
            batch_size=4,
            max_seq_len=32,
            mtp_loss_weight=0.3,
            flops_per_token=6e8,
            peak_tflops=DEFAULT_PEAK_TFLOPS,
        )
        opt = SimpleNamespace(learning_rate=0.01)
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_training_progress(
                epoch=0,
                step=0,
                iter_per_epoch=10,
                current_loss=1.0,
                optimizer=opt,
                start_time=0.0,
                args=args,
            )
        text = buf.getvalue()
        self.assertIn("mfu:", text)
        self.assertIn("%", text)


if __name__ == "__main__":
    unittest.main()
