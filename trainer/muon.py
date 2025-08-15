import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(G, steps: int):
    """
    通过 Newton-Schulz 迭代计算 G 的零次幂/正交化。我们选择使用一个
    五次迭代，其系数经过选择以最大化在零点的斜率。为了
    最小化迭代步数，凭经验发现，即使迭代在区间上
    不再完全收敛到 1，继续增加在零点的斜率也是有效的。
    因此，这次迭代不会产生 UV^T，而是产生类似 US'V^T 的东西，
    其中 S' 是对角矩阵，S_{ii}' ~ Uniform(0.5, 1.5)，这实际上
    相对于 UV^T（其中 USV^T = G 是 SVD）并不会损害模型性能。
    """
    assert (
        G.ndim >= 2
    )  # 由 @scottjmaddox 实现的批量 Muon，并由 @YouJiacheng 在实践中应用
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # 确保谱范数最多为 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # 执行 NS 迭代
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # 根据 @jxbz、@leloykun 和 @YouJiacheng 的建议改编的五次计算策略
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # 针对卷积核的情况
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - 通过 Newton-Schulz 正交化的动量优化器

    https://kellerjordan.github.io/posts/muon/

    Muon 内部运行标准的 SGD-momentum，然后执行一个正交化的后处理步骤，
    其中每个二维参数的更新被替换为最近的正交矩阵。为了高效地
    进行正交化，我们使用 Newton-Schulz 迭代，它的优点是可以在 GPU 上
    稳定地以 bfloat16 格式运行。

    Muon 只应用于隐藏权重层。输入嵌入、最终输出层以及
    任何内部的增益或偏置都应使用标准方法（如 AdamW）进行优化。
    隐藏的卷积权重可以通过将它们视为二维然后折叠最后 3 个维度来使用 Muon 进行训练。

    参数:
        lr: 学习率，单位为每次更新的谱范数。
        weight_decay: AdamW 风格的权重衰减。
        momentum: 动量。通常 0.95 就可以了。
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                dist.get_world_size() - len(params) % dist.get_world_size()
            )
            for base_i in range(len(params))[:: dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # 强制同步
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=True,
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + dist.get_world_size()],
                    params_pad[base_i + dist.get_rank()],
                )

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    用于非分布式设置的 Muon 变体。
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)  # 强制同步
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    nesterov=True,
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


class MuonClip(torch.optim.Optimizer):
    """
    MuonClip - Muon + QK-Clip 优化器，结合了 Muon 优化器的高效性与 QK-Clip 机制的稳定性

    基于 Kimi K2 技术报告的实现，通过监控注意力 logits 并在超过阈值时重新缩放
    查询和键的投影权重来防止注意力 logit 爆炸，确保训练稳定性。

    参数:
        param_groups: 参数组列表，包含 use_muon 标志
        tau: QK-Clip 阈值，默认 25.0
    """

    def __init__(self, param_groups, tau=25.0):
        # 首先验证和设置参数组
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                # 默认值
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"]
                )
            else:
                # 默认值
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )

        super().__init__(param_groups, dict())
        self.tau = tau
        self.qk_clip_stats = {
            "current_step_activations": 0,  # 当前步骤的激活次数
            "current_step_checks": 0,  # 当前步骤的检查次数
        }
        self._temp_model_ref = None  # 临时模型引用

    def _get_unwrapped_model(self, model):
        """
        Safely unwrap model from DDP/FSDP wrappers to access the underlying model.
        """
        if model is None:
            return None
        # Handle DDP wrapped models
        if hasattr(model, "module"):
            return model.module
        return model

    def apply_qk_clip(self, model):
        """
        Fully vectorized QK-Clip mechanism for improved efficiency.
        Handles DDP/FSDP wrapping robustly.
        """
        # 1. 获取解包后的模型实例
        unwrapped_model = self._get_unwrapped_model(model)
        if unwrapped_model is None:
            return

        if (
            not hasattr(unwrapped_model, "attention_max_logits")
            or unwrapped_model.attention_max_logits is None
        ):
            return

        # 使用解包后的模型引用访问缓冲区
        max_logits_tensor = (
            unwrapped_model.attention_max_logits
        )  # (num_layers, num_heads)

        # DDP Synchronization: Ensure all ranks use the global maximum logit.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(max_logits_tensor, op=dist.ReduceOp.MAX)

        tau = self.tau
        tau_tensor = torch.tensor(
            tau, device=max_logits_tensor.device, dtype=max_logits_tensor.dtype
        )

        # Vectorized Scaling Calculation (Following Kimi K2 Algorithm 1)
        # Ensure numerical stability. Clamp logits to a small positive value.
        stable_logits = torch.clamp(max_logits_tensor, min=1e-6)

        # Calculate scaling factors using Kimi K2 approach:
        # γ ← τ/S^h_max, then W^h_qc ← W^h_qc · √γ
        gamma = tau_tensor / stable_logits  # γ = τ/S_max
        # [MODIFICATION] Use gamma directly for more aggressive scaling, instead of torch.sqrt(gamma).
        # This provides a stronger corrective force when logits explode, changing the scaling
        # from quadratic (sqrt(gamma)^2) to linear (gamma), which should be more effective.
        all_scaling_factors = gamma  # torch.sqrt(gamma)

        # Cap the scaling factors at 1.0 (We only scale down).
        all_scaling_factors = torch.clamp(all_scaling_factors, max=1.0)

        # Statistics Collection
        # Identify how many heads required scaling (activation)
        activations = (all_scaling_factors < (1.0 - 1e-7)).sum().item()
        total_checks = max_logits_tensor.numel()

        self.qk_clip_stats["current_step_activations"] += activations
        self.qk_clip_stats["current_step_checks"] += total_checks

        if activations == 0:
            return

        # 2. 稳健地识别模型层结构 (兼容 Viby/HuggingFace 结构)
        target_layers = None
        if hasattr(unwrapped_model, "model") and hasattr(
            unwrapped_model.model, "layers"
        ):
            target_layers = unwrapped_model.model.layers
        elif hasattr(unwrapped_model, "layers"):
            target_layers = unwrapped_model.layers

        if target_layers is None:
            # 如果无法识别层结构，则跳过
            print("Warning: QK-Clip could not find model layers structure.")
            return

        # Vectorized Weight Update
        with torch.no_grad():
            # 使用识别出的 target_layers 进行迭代
            for layer_idx, layer in enumerate(target_layers):
                attention = layer.self_attn
                H_D = attention.head_dim
                H_Q = attention.n_local_heads
                H_KV = attention.n_local_kv_heads
                N_rep = attention.n_rep

                # Get the scaling factors for Q heads in this layer
                q_scaling = all_scaling_factors[layer_idx, :H_Q]  # (H_Q,)

                # Optimization: Skip layer if no scaling is needed
                if (q_scaling == 1.0).all():
                    continue

                # Vectorized Q Scaling
                # Expand scaling factors: (H_Q,) -> (H_Q*H_D,)
                # Using expand is memory-efficient as it uses strides instead of copying data.
                # (H_Q, 1) -> expand to (H_Q, H_D) -> reshape to (H_Q*H_D,)
                q_scaling_expanded = q_scaling.unsqueeze(1).expand(-1, H_D).reshape(-1)

                # Apply scaling using broadcasting: weight *= scaling_vector.unsqueeze(1)
                q_weight_dtype = attention.q_proj.weight.dtype
                # Use .data for in-place modification within no_grad context
                attention.q_proj.weight.data *= q_scaling_expanded.unsqueeze(1).to(
                    q_weight_dtype
                )

                # Vectorized K Scaling (GQA aggregation)

                if N_rep > 1:
                    # Reshape Q scaling factors: (H_Q,) -> (H_KV, N_rep)
                    q_scaling_grouped = q_scaling.view(H_KV, N_rep)
                    # Aggregate using min (find the most aggressive scaling needed in the group)
                    # .values extracts the tensor from the named tuple returned by torch.min
                    k_scaling = torch.min(q_scaling_grouped, dim=1).values  # (H_KV,)
                else:
                    # MHA case (N_rep == 1)
                    k_scaling = q_scaling

                # Expand K scaling factors: (H_KV,) -> (H_KV*H_D,)
                k_scaling_expanded = k_scaling.unsqueeze(1).expand(-1, H_D).reshape(-1)

                # Apply scaling
                k_weight_dtype = attention.k_proj.weight.dtype
                attention.k_proj.weight.data *= k_scaling_expanded.unsqueeze(1).to(
                    k_weight_dtype
                )

    @torch.no_grad()
    def step(self, closure=None, model=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 重置当前步骤的统计
        self.qk_clip_stats["current_step_activations"] = 0
        self.qk_clip_stats["current_step_checks"] = 0

        # 使用临时设置的模型引用（如果可用）
        if (
            model is None
            and hasattr(self, "_temp_model_ref")
            and self._temp_model_ref is not None
        ):
            model = self._temp_model_ref

        unwrapped_model = self._get_unwrapped_model(model)

        # 第一步：聚合来自前向传播的统计信息
        if unwrapped_model is not None and hasattr(
            unwrapped_model, "update_attention_stats_from_forward"
        ):
            unwrapped_model.update_attention_stats_from_forward()

        # 第二步：执行标准的 Muon/Adam 更新 (按照 Algorithm 1)
        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                    dist.get_world_size() - len(params) % dist.get_world_size()
                )
                for base_i in range(len(params))[:: dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            nesterov=True,
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(
                        params_pad[base_i : base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        # 第三步：权重更新后应用 QK-Clip (按照 Algorithm 1)
        if model is not None:
            # 传入原始（可能被包装的）model，apply_qk_clip 内部处理解包
            self.apply_qk_clip(model)

        # 第四步：重置统计，为下一次迭代准备
        if (
            unwrapped_model is not None
            and hasattr(unwrapped_model, "attention_max_logits")
            and unwrapped_model.attention_max_logits is not None
        ):
            unwrapped_model.attention_max_logits.zero_()

        return loss


class SingleDeviceMuonClip(torch.optim.Optimizer):
    """
    MuonClip 的单设备版本，用于非分布式训练
    """

    def __init__(self, param_groups, tau=25.0):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # 默认值
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"]
                )
            else:
                # 默认值
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )

        super().__init__(param_groups, dict())
        self.tau = tau
        self.qk_clip_stats = {
            "current_step_activations": 0,  # 当前步骤的激活次数
            "current_step_checks": 0,  # 当前步骤的检查次数
        }
        self._temp_model_ref = None  # 临时模型引用

    def _get_unwrapped_model(self, model):
        """
        Safely unwrap model from DDP/FSDP wrappers to access the underlying model.
        """
        if model is None:
            return None
        # Handle DDP wrapped models (though SingleDevice usually doesn't need this)
        if hasattr(model, "module"):
            return model.module
        return model

    def apply_qk_clip(self, model):
        """
        Fully vectorized QK-Clip mechanism for improved efficiency.
        Handles potential model wrapping robustly.
        """
        # 1. 获取解包后的模型实例
        unwrapped_model = self._get_unwrapped_model(model)
        if unwrapped_model is None:
            return

        if (
            not hasattr(unwrapped_model, "attention_max_logits")
            or unwrapped_model.attention_max_logits is None
        ):
            return

        # 使用解包后的模型引用访问缓冲区
        max_logits_tensor = (
            unwrapped_model.attention_max_logits
        )  # (num_layers, num_heads)

        tau = self.tau
        tau_tensor = torch.tensor(
            tau, device=max_logits_tensor.device, dtype=max_logits_tensor.dtype
        )

        # Vectorized Scaling Calculation (Following Kimi K2 Algorithm 1)
        # Ensure numerical stability. Clamp logits to a small positive value.
        stable_logits = torch.clamp(max_logits_tensor, min=1e-6)

        # Calculate scaling factors using Kimi K2 approach:
        # γ ← τ/S^h_max, then W^h_qc ← W^h_qc · √γ
        gamma = tau_tensor / stable_logits  # γ = τ/S_max
        # [MODIFICATION] Use gamma directly for more aggressive scaling, instead of torch.sqrt(gamma).
        # This provides a stronger corrective force when logits explode, changing the scaling
        # from quadratic (sqrt(gamma)^2) to linear (gamma), which should be more effective.
        all_scaling_factors = gamma  # torch.sqrt(gamma)

        # Cap the scaling factors at 1.0 (We only scale down).
        all_scaling_factors = torch.clamp(all_scaling_factors, max=1.0)

        # Statistics Collection
        # Identify how many heads required scaling (activation)
        activations = (all_scaling_factors < (1.0 - 1e-7)).sum().item()
        total_checks = max_logits_tensor.numel()

        self.qk_clip_stats["current_step_activations"] += activations
        self.qk_clip_stats["current_step_checks"] += total_checks

        if activations == 0:
            return

        # 2. 稳健地识别模型层结构 (兼容 Viby/HuggingFace 结构)
        target_layers = None
        if hasattr(unwrapped_model, "model") and hasattr(
            unwrapped_model.model, "layers"
        ):
            target_layers = unwrapped_model.model.layers
        elif hasattr(unwrapped_model, "layers"):
            target_layers = unwrapped_model.layers

        if target_layers is None:
            # 如果无法识别层结构，则跳过
            print("Warning: QK-Clip could not find model layers structure.")
            return

        # Vectorized Weight Update
        with torch.no_grad():
            # 使用识别出的 target_layers 进行迭代
            for layer_idx, layer in enumerate(target_layers):
                attention = layer.self_attn
                H_D = attention.head_dim
                H_Q = attention.n_local_heads
                H_KV = attention.n_local_kv_heads
                N_rep = attention.n_rep

                # Get the scaling factors for Q heads in this layer
                q_scaling = all_scaling_factors[layer_idx, :H_Q]  # (H_Q,)

                # Optimization: Skip layer if no scaling is needed
                if (q_scaling == 1.0).all():
                    continue

                # Vectorized Q Scaling
                # Expand scaling factors: (H_Q,) -> (H_Q*H_D,)
                # Using expand is memory-efficient as it uses strides instead of copying data.
                # (H_Q, 1) -> expand to (H_Q, H_D) -> reshape to (H_Q*H_D,)
                q_scaling_expanded = q_scaling.unsqueeze(1).expand(-1, H_D).reshape(-1)

                # Apply scaling using broadcasting: weight *= scaling_vector.unsqueeze(1)
                q_weight_dtype = attention.q_proj.weight.dtype
                # Use .data for in-place modification within no_grad context
                attention.q_proj.weight.data *= q_scaling_expanded.unsqueeze(1).to(
                    q_weight_dtype
                )

                # Vectorized K Scaling (GQA aggregation)

                if N_rep > 1:
                    # Reshape Q scaling factors: (H_Q,) -> (H_KV, N_rep)
                    q_scaling_grouped = q_scaling.view(H_KV, N_rep)
                    # Aggregate using min (find the most aggressive scaling needed in the group)
                    # .values extracts the tensor from the named tuple returned by torch.min
                    k_scaling = torch.min(q_scaling_grouped, dim=1).values  # (H_KV,)
                else:
                    # MHA case (N_rep == 1)
                    k_scaling = q_scaling

                # Expand K scaling factors: (H_KV,) -> (H_KV*H_D,)
                k_scaling_expanded = k_scaling.unsqueeze(1).expand(-1, H_D).reshape(-1)

                # Apply scaling
                k_weight_dtype = attention.k_proj.weight.dtype
                attention.k_proj.weight.data *= k_scaling_expanded.unsqueeze(1).to(
                    k_weight_dtype
                )

    @torch.no_grad()
    def step(self, closure=None, model=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 重置当前步骤的统计
        self.qk_clip_stats["current_step_activations"] = 0
        self.qk_clip_stats["current_step_checks"] = 0

        # 使用临时设置的模型引用（如果可用）
        if (
            model is None
            and hasattr(self, "_temp_model_ref")
            and self._temp_model_ref is not None
        ):
            model = self._temp_model_ref

        unwrapped_model = self._get_unwrapped_model(model)

        # 第一步：聚合来自前向传播的统计信息
        if unwrapped_model is not None and hasattr(
            unwrapped_model, "update_attention_stats_from_forward"
        ):
            unwrapped_model.update_attention_stats_from_forward()

        # 第二步：执行标准的 Muon/Adam 更新 (按照 Algorithm 1)
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=True,
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        # 第三步：权重更新后应用 QK-Clip (按照 Algorithm 1)
        if model is not None:
            # 传入原始（可能被包装的）model，apply_qk_clip 内部处理解包
            self.apply_qk_clip(model)

        # 第四步：重置统计，为下一次迭代准备
        if (
            unwrapped_model is not None
            and hasattr(unwrapped_model, "attention_max_logits")
            and unwrapped_model.attention_max_logits is not None
        ):
            unwrapped_model.attention_max_logits.zero_()

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    分布式的 Muon 变体，可用于网络中的所有参数，因为它为
    不兼容 Muon 的参数运行一个内部的 AdamW。用户必须通过传入
    设置了 `use_muon` 标志的 param_groups 列表来手动指定哪些参数
    应使用 Muon 进行优化，哪些使用 Adam 进行优化。

    这个类的目的是让用户在代码中只有一个优化器，而不是
    同时拥有一个 Muon 和一个 Adam，每个都需要进行 step 操作。

    你可以在下面看到一个使用示例：

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                # 默认值
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"]
                )
            else:
                # 默认值
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                    dist.get_world_size() - len(params) % dist.get_world_size()
                )
                for base_i in range(len(params))[:: dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)  # 强制同步
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            nesterov=True,
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(
                        params_pad[base_i : base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # 强制同步
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    MuonWithAuxAdam 的非分布式变体。
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # 默认值
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"]
                )
            else:
                # 默认值
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # 强制同步
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=True,
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # 强制同步
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
