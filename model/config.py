import json
import math
import os


class VibyConfig:
    model_type = "viby"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.head_dim = kwargs.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        # local/global 分层（Kimi Linear 3:1）：global 层（(layer_idx+1)%4==0
        # 或最后一层）用 full-causal NoPE GQA（n_kv_heads_global，None 时取
        # num_attention_heads//4）；其余 local 层为 KDA（逐通道门控 delta
        # 规则线性注意力，见 kda.py）。全模型无 RoPE/滑窗。
        self.n_kv_heads_global = kwargs.get("n_kv_heads_global", None)
        self.hidden_act = kwargs.get("hidden_act", "situ")
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        # 默认解绑：lm_head 是独立 nn.Linear（muonh 下走 AdamH 组）。
        # True 仅保留给显式消融/旧 sidecar 兼容。
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)

        # 注意力输出门（Marin）：gate=2·σ(W_g·x)，W_g 零初始化（初始门 1）。
        self.use_attn_gate = kwargs.get("use_attn_gate", True)
        # MTP（Kimi-K3 / EAGLE-3 口径）：预训练挂 1 个与主干同构的 MTP 层，
        # 输入为主干多层特征（低/中/高层 AttnRes 合并后的 hidden）融合 +
        # 下一 token 嵌入；后训练阶段冻结主干微调为 EAGLE-3 草稿模型。
        self.mtp_depth = kwargs.get("mtp_depth", 1)
        self.mtp_loss_weight = kwargs.get("mtp_loss_weight", 0.3)
        # EAGLE-3 多层特征抽取层（1-indexed 层输出，各层 AttnRes 合并后、
        # final_norm 前）：None → 自动 [1, ceil(L/2), L]（L=8 → [1,4,8]）。
        _fl = kwargs.get("mtp_feature_layers", None)
        if _fl is None:
            _fl = (1, math.ceil(num_hidden_layers / 2), num_hidden_layers)
        self.mtp_feature_layers = tuple(sorted(int(i) for i in _fl))
        for _i in self.mtp_feature_layers:
            if not 1 <= _i <= num_hidden_layers:
                raise ValueError(
                    f"mtp_feature_layers 各项必须在 [1, {num_hidden_layers}] 内: "
                    f"{self.mtp_feature_layers}"
                )
        if len(self.mtp_feature_layers) != 3:
            # MTPModule 硬编码低/中/高三路融合（norm_low/mid/high + fc_l(3d)），
            # 提前在配置期报错，避免运行时 shape mismatch
            raise ValueError(
                f"mtp_feature_layers 必须恰好 3 项（低/中/高）: "
                f"{self.mtp_feature_layers}"
            )
        # logit z-loss（Marin）：loss += z_loss_weight * mean(lse²)，
        # 主 LM 与 MTP 各 head 同权重，融合进 CE kernel。
        self.z_loss_weight = float(kwargs.get("z_loss_weight", 1e-4))
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        # DeepSeekMoE（V3/V4 风格）：FFN = n_shared_experts 个共享专家 +
        # E 个细粒度路由专家（所有层均为 MoE）。
        # 路由：sigmoid 打分，选择分 = sigmoid 分 + expert_bias；另有一个
        # expert_bias 是
        # frozen 非梯度 buffer，由训练循环按 QB（分位数均衡）规则每步
        # 快照覆写（见 VibyForCausalLM.update_moe_biases）。top-k 命中后
        # 用原始 sigmoid 分归一化并乘 routed_scaling_factor。
        self.n_routed_experts = kwargs.get("n_routed_experts", 0)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 6)
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", None)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.routed_scaling_factor = kwargs.get("routed_scaling_factor", 2.5)
        # 路由 logits 逐 token 标准化：让 sigmoid 始终工作在敏感区，
        # 防止 logits 后期被推到 ±5 以上后 bias 失效。
        self.moe_router_logit_norm = bool(kwargs.get("moe_router_logit_norm", True))
        self.moe_router_logit_temp = float(kwargs.get("moe_router_logit_temp", 1.0))
        # router 输入 token 多样性正则：loss = mean(log1p(common²/residual²))。
        # 直接惩罚训练后期所有 token 收敛到同一方向（res/common 塌缩）。
        self.moe_diversity_loss_weight = float(
            kwargs.get("moe_diversity_loss_weight", 0.0)
        )
        # 软负载均衡辅助损失权重。loss = (E/K)·Σ f_i·p_i，f_i 为专家
        # 选择频率（stop_gradient）、p_i 为平均 sigmoid 分。0 关闭。
        # Latent MoE：路由专家在 latent 空间（维度 moe_latent_dim）计算——
        # 共享 lat_down/lat_up 投影包住路由专家，lat_down 后接 learnable
        # latent_norm（RMSNorm）再 dispatch；router 仍在全维 hidden 上
        # 打分，共享专家保持全宽（抗 dropping 主干）。d=hidden/2 时每个
        # 路由专家的 FLOPs/参数减半，等预算可把 moe_intermediate_size 翻倍。
        # 默认 None → hidden_size//2（开启）；显式 0 关闭（消融）。
        # 只用于新训练：无零初始化等价，函数类从 step 0 改变。
        _lat = kwargs.get("moe_latent_dim", None)
        self.moe_latent_dim = self.hidden_size // 2 if _lat is None else int(_lat)

        if self.n_kv_heads_global is None:
            self.n_kv_heads_global = max(1, self.num_attention_heads // 4)

        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if (
            self.n_kv_heads_global <= 0
            or self.num_attention_heads % self.n_kv_heads_global != 0
        ):
            raise ValueError("n_kv_heads_global 必须为正且整除 num_attention_heads")
        if self.n_routed_experts <= 0:
            raise ValueError("仅支持 MoE FFN：n_routed_experts 必须大于 0")
        if self.num_experts_per_tok <= 0:
            raise ValueError("num_experts_per_tok 必须大于 0")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError(
                "num_experts_per_tok 不能大于 n_routed_experts，"
                "否则 top-k 会重复选中同一专家"
            )
        if self.n_shared_experts < 0:
            raise ValueError("n_shared_experts 不能为负数")
        if self.moe_latent_dim < 0:
            raise ValueError("moe_latent_dim 不能为负数")
        if self.moe_latent_dim >= self.hidden_size and self.moe_latent_dim > 0:
            raise ValueError(
                "moe_latent_dim 必须小于 hidden_size（压缩才有意义），0 关闭"
            )
        if self.moe_router_logit_temp <= 0:
            raise ValueError("moe_router_logit_temp 必须大于 0")
        if self.moe_diversity_loss_weight < 0:
            raise ValueError("moe_diversity_loss_weight 不能为负数")
        if self.z_loss_weight < 0:
            raise ValueError("z_loss_weight 不能为负数")
        moe_in = self.moe_intermediate_size or self.intermediate_size
        if moe_in <= 0:
            raise ValueError("moe_intermediate_size/intermediate_size 必须大于 0")

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "VibyConfig":
        data = dict(data)
        data.pop("model_type", None)
        return cls(**data)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "VibyConfig":
        with open(os.path.join(path, "config.json")) as f:
            return cls.from_dict(json.load(f))
