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
        # 注意力后端。默认 MLA（DeepSeek V2/V3 full softmax）：KV 压到
        # kv_lora_rank，解耦 RoPE（qk_rope_head_dim，跨 head 共享），
        # 逐 head QK 维 = head_dim + qk_rope_head_dim、V 维 = head_dim。
        # use_linear_attn=True 才启用 Kimi Linear 3:1（KDA local + NoPE
        # GQA global + ShortConv），全模型无 RoPE。
        self.use_linear_attn = bool(kwargs.get("use_linear_attn", False))
        self.kv_lora_rank = int(kwargs.get("kv_lora_rank", 192))
        self.qk_rope_head_dim = int(kwargs.get("qk_rope_head_dim", 32))
        self.rope_theta = float(kwargs.get("rope_theta", 1e6))
        self.original_max_position_embeddings = int(
            kwargs.get("original_max_position_embeddings", 2048)
        )
        self.rope_scaling = kwargs.get("rope_scaling", None)
        if self.rope_scaling is None and kwargs.get("inference_rope_scaling", False):
            self.rope_scaling = {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": self.original_max_position_embeddings,
                "attention_factor": 1.0,
                "type": "yarn",
            }
        # linear 模式下 global 层（(layer_idx+1)%4==0 或最后一层）用
        # full-causal NoPE GQA（n_kv_heads_global，None 时取
        # num_attention_heads//4）；其余 local 层为 KDA。
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
        # MTP（Qwen3.8-Next 口径）：0=关闭；>0 挂 1 个 full-attn MoE 层
        # （与 Qwen `mtp_num_hidden_layers=1` 对齐，不再堆独立 depth 模块）。
        # 输入 = 主干末层 hidden（final_norm 前）+ 下一 token 嵌入；
        # 预训练把同一层 teacher-forced 展开 mtp_steps 次。
        self.mtp_depth = kwargs.get("mtp_depth", 1)
        self.mtp_loss_weight = kwargs.get("mtp_loss_weight", 0.3)
        self.mtp_steps = int(kwargs.get("mtp_steps", 2))
        # 旧 sidecar 的 EAGLE-3 三路层号只作往返保留，前向不再读取。
        _fl = kwargs.get("mtp_feature_layers", None)
        if _fl is None:
            _fl = (num_hidden_layers,)
        self.mtp_feature_layers = tuple(int(i) for i in _fl)
        # logit z-loss（Marin）：loss += z_loss_weight * mean(lse²)，
        # 主 LM 与 MTP 各 head 同权重，融合进 CE kernel。
        self.z_loss_weight = float(kwargs.get("z_loss_weight", 1e-4))
        # 未使用：真正生效的是 apply_trunc_normal_init 的 0.5/√fan_in。
        # 字段只为旧 sidecar JSON 往返，不要当 GPT-2 式 0.02 初始化读。
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        # DeepSeekMoE（V3/V4 风格）：FFN = n_shared_experts 个共享专家 +
        # E 个细粒度路由专家（所有层均为 MoE）。
        # 路由：sigmoid 打分，选择分 = sigmoid 分 + expert_bias。
        # expert_bias 是 frozen 非梯度 buffer，由训练循环按 QB（分位数均衡）
        # 规则每步快照覆写（见 VibyForCausalLM.update_moe_biases）。top-k 命中后
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
        # Latent MoE：路由专家在 latent 空间（维度 moe_latent_dim）计算——
        # 共享 lat_down/lat_up 投影包住路由专家，lat_down 后接 learnable
        # latent_norm（RMSNorm）再 dispatch；聚合输出在 lat_up 前再过
        # latent_out_norm（K3 Stable LatentMoE §2.3.1：压低路由分支对
        # 专家选择/路由权重尺度漂移的敏感度）。router 仍在全维 hidden 上
        # 打分，共享专家保持全宽（抗 dropping 主干）。d=hidden/2 时每个
        # 路由专家的 FLOPs/参数减半，等预算可把 moe_intermediate_size 翻倍。
        # 默认 None → hidden_size//2（开启）；显式 0 关闭（消融）。
        # 只用于新训练：无零初始化等价，函数类从 step 0 改变。
        _lat = kwargs.get("moe_latent_dim", None)
        self.moe_latent_dim = self.hidden_size // 2 if _lat is None else int(_lat)

        # KDA V 头扩张（Qwen GDN：V 头多于 QK 头，Q/K/门按组重复）。
        # 默认 2：8 QK 头 → 16 V 头。1=旧正方形。只用于新训练。
        self.kda_v_head_ratio = int(kwargs.get("kda_v_head_ratio", 2))
        # 哈希 n-gram 查找表（Qwen3.8-Flash-Next layer-2 口径的小表版）。
        # 0=关闭。默认 2^16；挂在 ngram_layer（1-indexed）入口，门零初始化。
        self.ngram_table_size = int(kwargs.get("ngram_table_size", 65536))
        self.ngram_layer = int(kwargs.get("ngram_layer", 2))
        _ngo = kwargs.get("ngram_orders", (2, 3))
        self.ngram_orders = tuple(int(i) for i in _ngo)

        if self.n_kv_heads_global is None:
            self.n_kv_heads_global = max(1, self.num_attention_heads // 4)

        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.use_linear_attn:
            if (
                self.n_kv_heads_global <= 0
                or self.num_attention_heads % self.n_kv_heads_global != 0
            ):
                raise ValueError("n_kv_heads_global 必须为正且整除 num_attention_heads")
        else:
            if self.kv_lora_rank <= 0:
                raise ValueError("kv_lora_rank 必须大于 0")
            if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2 != 0:
                raise ValueError("qk_rope_head_dim 必须为正偶数")
            if self.rope_theta <= 0:
                raise ValueError("rope_theta 必须大于 0")
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
        if self.mtp_depth > 0 and self.mtp_steps < 1:
            raise ValueError("mtp_steps 必须大于 0")
        if self.kda_v_head_ratio < 1:
            raise ValueError("kda_v_head_ratio 必须 >= 1")
        if self.ngram_table_size < 0:
            raise ValueError("ngram_table_size 不能为负（0 关闭）")
        if self.ngram_table_size > 0:
            if self.ngram_layer < 1:
                raise ValueError("ngram_layer 是 1-indexed，必须 >= 1")
            if not self.ngram_orders:
                raise ValueError("ngram_orders 不能为空")
            for _o in self.ngram_orders:
                if _o not in (2, 3):
                    raise ValueError("ngram_orders 只支持 2（bigram）和 3（trigram）")
        moe_in = self.moe_intermediate_size or self.intermediate_size
        if moe_in <= 0:
            raise ValueError("moe_intermediate_size/intermediate_size 必须大于 0")

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "VibyConfig":
        data = dict(data)
        data.pop("model_type", None)
        # 旧 KDA/GQA sidecar 没有 MLA 字段：保持当时的 linear attn 默认，
        # 否则 strict 加载会把 KDA 权重对到 MLA 上。
        if "use_linear_attn" not in data and "kv_lora_rank" not in data:
            data["use_linear_attn"] = True
        return cls(**data)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "VibyConfig":
        with open(os.path.join(path, "config.json")) as f:
            return cls.from_dict(json.load(f))
