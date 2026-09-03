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
        # FFN 激活：'silu'（默认，SwiGLU/silu(g)·u）或 'situ'（SiTU-GLU，β1/β2
        # tanh 软帽）。旧 sidecar 存了旧值会恢复 situ；新 run 无该键走 silu。
        self.hidden_act = kwargs.get("hidden_act", "silu")
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
        # Gated XSA（Exclusive Self-Attention 的可学习版，arXiv:2603.09078 /
        # modded-nanogpt record #82）：逐 head 学 tanh(α) 扣掉 attention 输出
        # 中与自身 V 平行的分量 z = y − tanh(α)·(yᵀv/‖v‖²)·v；α=0 初始化 ⇒
        # 起步恒等（层可自选剂量，故不强制限制层数也能让浅层趋于 0）。
        # 默认开；xsa_last_n=0 时按配方取最深 ≈1/3 层（参数高尔夫 XSA_LAST_N、
        # 自注意力偏置随层加深而增大，故收益集中在深层）。
        self.use_xsa = bool(kwargs.get("use_xsa", True))
        _xln = int(kwargs.get("xsa_last_n", 0) or 0)
        if _xln <= 0 and self.use_xsa:
            _xln = max(1, num_hidden_layers // 3)
        self.xsa_last_n = max(0, _xln)
        # AttnRes 混合窗口：只对最近 W 个残差做 softmax 加权。
        # 全历史是 O(L²) 读 v + 跨步 dv 累加（8 层 ΣN≈157）；窗口把
        # 每次 merge 的 N 钉死，墙钟从平方降到线性。0=论文全历史。
        self.attn_res_window = int(kwargs.get("attn_res_window", 4) or 0)
        # True：残差流做加法写（h ← h + F），AttnRes 只混合 [h]+近 W 个写入
        # 作为下一子层的读。False（默认）：AttnRes 替换 hidden（论文原式）。
        # 旧 sidecar 缺键 → False，与 r086 等对照 run 一致。
        self.attn_res_register = bool(kwargs.get("attn_res_register", False))
        # True：寄存器模式下子层输入 = h，不再 softmax 混合 [h]+写入。
        # 旧 sidecar 缺键 → False。需要 attn_res_register。
        self.attn_res_read_h = bool(kwargs.get("attn_res_read_h", False))
        # iHC（identity Hyper-Connections，Hy4 / Chimera）：M 条残差流，
        # 读 x̃=Σ h_pre,m R_m，写 R_m ← R_m + h_post,m Δ，H_res=I（无
        # Sinkhorn）。开时接管残差读写，AttnRes merge/read 不走；开关本身
        # 保留。旧 sidecar 缺键 → False。ihc_streams 关时仍存档，默认 4。
        self.ihc = bool(kwargs.get("ihc", False))
        self.ihc_streams = int(kwargs.get("ihc_streams", 4) or 0)
        # True：流 0 恒等、n-gram 只进 ihc_ngram_stream、进栈前注入。
        # 旧 sidecar 缺键 → False。ihc_collapse 缺省：typed 时 identity，否则 mean。
        self.ihc_typed = bool(kwargs.get("ihc_typed", False))
        _col = kwargs.get("ihc_collapse", None)
        if _col is None:
            self.ihc_collapse = "identity" if self.ihc_typed else "mean"
        else:
            self.ihc_collapse = str(_col)
        self.ihc_ngram_stream = int(kwargs.get("ihc_ngram_stream", 1) or 0)
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
        # E 个细粒度路由专家。first_k_dense_replace>0 时前 K 层改为
        # 单路 dense SwiGLU（默认与专家同宽），其后仍是 MoE。
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
        # True：路由专家在共享 lat_up 之外，按专家对角缩放 tile(h_k) 写入
        # 残差补空间（写出基扩展）。s 零初始化 ⇒ 起步与关掉逐位一致。
        # 旧 sidecar 缺键 → False。需要 moe_latent_dim > 0。
        self.moe_write_spread = bool(kwargs.get("moe_write_spread", False))
        # True：路由写出在 latent_out_norm 之后乘未归一化 top-k sigmoid 和。
        # RMSNorm 对正齐次，关掉时 routed_scaling_factor 写不进残差。
        # 旧 sidecar 缺键 → False。需要 moe_latent_dim > 0。
        self.moe_route_scale = bool(kwargs.get("moe_route_scale", False))
        # 浅层 dense stem：layer_idx < K 用 FeedForward，其后 MoE。
        # 0（默认）= 全层 MoE，旧 sidecar 缺键保持对照。
        self.first_k_dense_replace = int(kwargs.get("first_k_dense_replace", 0) or 0)
        # 浅层 dense 中间维。None 且 K>0 → moe_intermediate_size（小 stem）；
        # K=0 且未传 → 0（不占宽）。旧 sidecar 缺键同此。
        _dense_i = kwargs.get("dense_intermediate_size", None)
        if _dense_i is None:
            self.dense_intermediate_size = (
                int(self.moe_intermediate_size or self.intermediate_size)
                if self.first_k_dense_replace > 0
                else 0
            )
        else:
            self.dense_intermediate_size = int(_dense_i)

        # KDA V 头扩张（Qwen GDN：V 头多于 QK 头，Q/K/门按组重复）。
        # 默认 2：8 QK 头 → 16 V 头。1=旧正方形。只用于新训练。
        self.kda_v_head_ratio = int(kwargs.get("kda_v_head_ratio", 2))
        # 哈希 n-gram 查找表（Engram 口径，arXiv 2601.07372：多头素数哈希 +
        # 上下文门控 + 膨胀短卷积）。0=关闭。ngram_table_size 是全部
        # （阶 × 头）子表的总桶数（各头素数取整，实际略小）；记忆维
        # d_mem = ngram_d_mem（0=取 hidden）；每头维 = d_mem/(阶数×头数)。
        # 挂在 ngram_layer（1-indexed）入口，W_V/卷积零初始化 ⇒ 起步恒等。
        self.ngram_table_size = int(kwargs.get("ngram_table_size", 65536))
        self.ngram_layer = int(kwargs.get("ngram_layer", 2))
        _ngo = kwargs.get("ngram_orders", (2, 3))
        self.ngram_orders = tuple(int(i) for i in _ngo)
        self.ngram_heads = int(kwargs.get("ngram_heads", 8))
        self.ngram_d_mem = int(kwargs.get("ngram_d_mem", 0) or 0)
        # True：未过门的 n-gram 向量经 lm_head 加到 logits（标量尺度零初始化）。
        # 旧 sidecar 缺键 → False。
        self.ngram_logit_skip = bool(kwargs.get("ngram_logit_skip", False))
        # True：用 n-gram unembed 的 max-softmax 置信度缩放每层 attn/mlp 写入。
        # g = 1 − s · stopgrad(max p)；s 零初始化 ⇒ step 0 时 g≡1。
        # 旧 sidecar 缺键 → False。需要 ngram_table_size > 0。
        self.ngram_conf_gate = bool(kwargs.get("ngram_conf_gate", False))

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
        if self.moe_write_spread and self.moe_latent_dim <= 0:
            raise ValueError("moe_write_spread 需要 moe_latent_dim > 0")
        if self.moe_route_scale and self.moe_latent_dim <= 0:
            raise ValueError("moe_route_scale 需要 moe_latent_dim > 0")
        if self.attn_res_read_h and not self.attn_res_register:
            raise ValueError("attn_res_read_h 需要 attn_res_register")
        if self.ihc and self.ihc_streams < 2:
            raise ValueError("ihc_streams 必须 >= 2")
        if self.ihc_collapse not in ("mean", "identity"):
            raise ValueError("ihc_collapse 必须是 mean 或 identity")
        if self.ihc_typed and not self.ihc:
            raise ValueError("ihc_typed 需要 ihc")
        if self.ihc:
            if not (0 <= self.ihc_ngram_stream < max(self.ihc_streams, 1)):
                raise ValueError("ihc_ngram_stream 必须在 [0, ihc_streams)")
            if (
                self.ihc_typed
                and self.ihc_collapse == "identity"
                and self.ihc_ngram_stream == 0
            ):
                raise ValueError("ihc_ngram_stream 不能为 0（identity 塌缩读流 0）")
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
            if self.ngram_heads < 1:
                raise ValueError("ngram_heads 必须 >= 1")
            _dmem = self.ngram_d_mem or self.hidden_size
            _ntab = len(self.ngram_orders) * self.ngram_heads
            if _dmem % _ntab != 0:
                raise ValueError(
                    f"d_mem({_dmem}) 必须被 阶数×头数({_ntab}) 整除"
                )
            if (self.ngram_logit_skip or self.ngram_conf_gate) and _dmem != (
                self.hidden_size
            ):
                raise ValueError(
                    "ngram_logit_skip/ngram_conf_gate 需要 d_mem == hidden_size"
                )
        if self.ngram_conf_gate and self.ngram_table_size <= 0:
            raise ValueError("ngram_conf_gate 需要 ngram_table_size > 0")
        moe_in = self.moe_intermediate_size or self.intermediate_size
        if moe_in <= 0:
            raise ValueError("moe_intermediate_size/intermediate_size 必须大于 0")
        if self.first_k_dense_replace < 0:
            raise ValueError("first_k_dense_replace 不能为负")
        if self.first_k_dense_replace > self.num_hidden_layers:
            raise ValueError("first_k_dense_replace 不能大于 num_hidden_layers")
        if self.first_k_dense_replace > 0 and self.dense_intermediate_size <= 0:
            raise ValueError("first_k_dense_replace>0 需要 dense_intermediate_size>0")

    def is_dense_ffn(self, layer_idx: int) -> bool:
        return 0 <= int(layer_idx) < self.first_k_dense_replace

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
