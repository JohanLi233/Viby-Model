"""
基础训练器类，提供通用的训练逻辑（MLX 单设备版）
"""

import os
import queue
import sys
import threading
import time
import json

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from .muon import create_mixed_optimizer
from .fast_norm import gradient_square_sum
from .psr_optim import ParameterView, split_gradients
from .utils import (
    Logger,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    apply_lr_schedule,
    log_training_progress,
    resolve_warmup_iters,
    resolve_lr_horizon,
    set_ddp_flag,
    get_optimizer_steps,
)

_SENTINEL = object()

# 梯度累加是否每个微批就物化（默认开）。关掉可退回旧的惰性链行为做 A/B；
# 依据见 _run_epoch_steps 里的注释与 research/MLX_PERF.md。
_ACCUM_EAGER = os.environ.get("VIBY_ACCUM_EAGER", "1") != "0"
# optimizer step 末尾是否按块 eval 参数树，而不是一次性 eval(参数 + state)。
# 一次性 eval 会把新旧参数 / m / v / 梯度 / Muon 临时量同时顶到峰值
# （prof_target_step：win_eval 277ms/窗口，占窗口 ~70%），分块能让中间量
# 提前释放。A/B 开关：VIBY_OPT_CHUNKED_EVAL=1。
_OPT_CHUNKED_EVAL = os.environ.get("VIBY_OPT_CHUNKED_EVAL", "0") == "1"
_OPT_CHUNK_BYTES = 512 << 20
# 注：MoE 负载均衡已从旧的分位数 QB（margin 样本拼接）换成 V4.1 的
# noaux_tc：forward 物化每专家 token 计数 out.moe_loads，窗口内直接累加，
# 不需要样本抽稀/拼接。


def _collate_numpy(samples):
    """样本堆叠为 numpy batch（纯 CPU 工作，在后台线程执行）"""
    first = samples[0]
    if isinstance(first, dict):
        return {
            key: np.stack([np.asarray(sample[key]) for sample in samples])
            for key in first
        }
    return tuple(
        np.stack([np.asarray(sample[i]) for sample in samples])
        for i in range(len(first))
    )


def _to_mx(batch):
    """numpy batch -> mx.array（在主线程执行，开销极小）"""
    if isinstance(batch, dict):
        return {key: mx.array(value) for key, value in batch.items()}
    return tuple(mx.array(value) for value in batch)


class _PrefetchIterator:
    """单后台线程预取：样本读取 + numpy 堆叠与 GPU 计算重叠。

    不用 ThreadPoolExecutor：它的 worker 是 daemon，Ctrl-C / 提前停之后
    若还在跑 dataset.__getitem__，解释器 finalizing 时会 Fatal
    `PyThreadState_Get`（GIL 已释放）。一条非池化线程能在 close() 里
    可靠 join，打包 mmap 路径下单线程也够喂饱 GPU。
    """

    def __init__(self, loader):
        self.loader = loader
        self._queue = queue.Queue(maxsize=loader.prefetch_batches)
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._produce, name="mlx-prefetch", daemon=True
        )
        self._thread.start()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _produce(self):
        loader = self.loader
        try:
            n = len(loader.dataset)
            loader.epoch_shuffle_state = np.random.get_state()
            indices = np.random.permutation(n) if loader.shuffle else np.arange(n)
            for start in range(0, n, loader.batch_size):
                if self._stop.is_set():
                    return
                batch_indices = indices[start : start + loader.batch_size]
                if loader.drop_last and len(batch_indices) < loader.batch_size:
                    break
                samples = [loader.dataset[int(i)] for i in batch_indices]
                if self._stop.is_set():
                    return
                self._put(_collate_numpy(samples))
            if not self._stop.is_set():
                self._put(_SENTINEL)
        except Exception as e:
            if not self._stop.is_set():
                self._put(e)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set() and not self._thread.is_alive():
                    raise StopIteration
                continue
            if item is _SENTINEL:
                self._stop.set()
                raise StopIteration
            if isinstance(item, Exception):
                self._stop.set()
                raise item
            return _to_mx(item)

    def close(self):
        """停生产者、排空队列、join 预取线程。可重入。"""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if sys.is_finalizing():
            return
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MLXDataLoader:
    """简单的单设备数据加载器（异步预取版）。

    每个 epoch 迭代时重新 shuffle，按 batch_size 切分，drop_last。
    数据准备（读文件 + tokenize + 堆叠）在后台线程池完成，
    主线程只做 numpy -> mx.array 转换，避免 GPU 等待数据。
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_batches=4,
        num_workers=1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prefetch_batches = prefetch_batches
        # 保留字段以兼容旧调用；预取已改为单线程，不再开 worker 池。
        self.num_workers = num_workers

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        return _PrefetchIterator(self)


class BaseTrainer:
    """基础训练器类"""

    def __init__(self, args, model, tokenizer, lm_config, training_type="pretrain"):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.lm_config = lm_config
        self.training_type = training_type

        # 单设备 MLX 训练，无分布式概念（保留属性以兼容调用方）
        self.ddp = False
        self.ddp_local_rank = 0
        self.device = args.device

        set_ddp_flag(False)

        # 初始化训练组件
        self._init_training_components()

        # 训练时长预算（max_train_minutes）的起始时间，train() 开始时设置
        self._train_start_time = None
        # Ctrl-C 时保存检查点使用的当前 step 位置（train_epoch 内实时更新）
        self._last_epoch = 0
        self._last_step = 0
        self.interrupted = False

        # expl-nest（VIBY_EXPL_NEST，默认 0=关）：参数空间前瞻——梯度在
        # θ + μ·δ 处求值，δ 为上一优化器步的实际更新（f32）。autoresearch-mlx
        # 战役 d4@467 champion（−0.0125 bpb，transient regime 独狼机制，
        # 不可与其他机制叠加的论断来自该战役账本）。δ 跨整个累积窗口
        # 保持不变（窗口内无优化器步）。0 时完全无开销。
        self._expl_nest_mu = float(os.environ.get("VIBY_EXPL_NEST", "0.0"))
        self._en_delta = None

    def _time_limit_exceeded(self):
        """是否已达最长训练时长（分钟）。未设置或训练未开始时返回 False。"""
        limit = getattr(self.args, "max_train_minutes", None)
        if not limit or self._train_start_time is None:
            return False
        return (time.time() - self._train_start_time) >= limit * 60

    def _init_training_components(self):
        """初始化训练组件"""
        protected = getattr(self.lm_config, 'psr_enabled', False)
        base_view = ParameterView(self.model) if protected else self.model
        self.psr_optimizer = optim.AdamW(learning_rate=getattr(self.args,'psr_learning_rate',1e-4),
                                        weight_decay=getattr(self.args,'psr_weight_decay',0.0)) if protected else None
        self._psr_step = 0
        # 创建优化器
        if getattr(self.args, "optimizer", "muon") == "adamw":
            from .muon import create_adamw_optimizer

            self.optimizer = create_adamw_optimizer(
                base_view, self.args, self.training_type
            )
        else:
            self.optimizer = create_mixed_optimizer(
                base_view, self.args, self.training_type
            )

        # Save/restore the side optimizer separately; never place it in baseline state.
        self.optimizer.psr_optimizer = self.psr_optimizer
        # noaux_tc 的 e_score_correction_bias 更新：forward 把 [L,E] 的每专家
        # token 计数作为图输出物化（out.moe_loads），_optimizer_step 在窗口
        # 末尾按 b += γ·sign(load_frac − 1/E) 覆写 frozen 的 bias
        self._moe_gates = list(getattr(self.model, "moe_gates", None) or [])

        # loss + 梯度函数（MoE 用 sorted gather_mm 分发，形状只随 (B,T,E,K)
        # 静态确定，mx.compile 可用，见 _build_loss_and_grad）
        self._loss_and_grad = self._build_loss_and_grad()

        # Metal 分配器缓存上限（--cache_limit_gb，默认 0=不限）：
        # 上限内的空闲块常驻复用、不归还 OS，避免每步"释放-重分配"抖动
        # （bs16x640 实测 10G→24G 提速 4.5%；该配置峰值 14.8G，峰值+缓存
        # ≈39G）。历史上设限是为防 optimizer 临时 buffer 污染 freelist
        # 拖慢激活分配（243→442ms/步）；BatchedMuon 批量化后临时块少且
        # 形状固定，各档上限扫描均未复现污染。大 batch 配置注意
        # 峰值+缓存上限不要超过物理内存。
        cache_gb = float(getattr(self.args, "cache_limit_gb", 24.0))
        if cache_gb > 0:
            mx.set_cache_limit(int(cache_gb * 1024**3))

        # 处理检查点恢复
        self.start_epoch, self.start_step = self._handle_checkpoint_resume()
        self._psr_step = int(getattr(self.args,"psr_resumed_microstep", self.start_step))

    def _loss_fn(self, X, Y, loss_mask, attn_mask, seg_ids=None, psr_key=0, psr_gate=None):
        """训练 loss 函数（model 走闭包引用）

        返回 (加权和 loss / accumulation_steps, mtp 分量 loss, moe_loads,
        lm_loss / accumulation_steps, z_loss)。mtp/lm/z 分量仅作日志展示、
        不缩放；总 loss 里已含加权后的 mtp 与 z-loss。

        moe_loads 是 forward 物化的 [L, E] 每专家 token 计数（compile 下纯
        侧信道 g._last_load 会被剪枝，必须经返回值出图）。累积窗口内按元素
        累加即可（计数可加），窗口末尾交给 model.update_moe_biases 做
        noaux_tc 偏置更新；无 MoE 时用零长占位保持图结构稳定。
        """
        psr_inputs = dict(psr_mode="off", return_metrics=getattr(self.lm_config, "psr_enabled", False))
        if self.training_type == "pretrain" and getattr(self.lm_config, "psr_enabled", False):
            from .psr_pretrain import text_psr_inputs

            if getattr(self.args, 'psr_training_mode', 'recurrent') == 'off':
                psr_inputs = dict(psr_mode='off', return_metrics=True)
            else:
                psr_inputs = text_psr_inputs(self.lm_config, X, Y, loss_mask, attn_mask, seg_ids, psr_key)
                psr_inputs['psr_mode'] = getattr(self.args, 'psr_training_mode', 'recurrent')
            psr_inputs['psr_gate'] = psr_gate
        res = self.model(
            input_ids=X,
            labels=Y,
            loss_mask=loss_mask,
            attention_mask=attn_mask,
            segment_ids=seg_ids,
            **psr_inputs,
        )
        mtp_loss = res.mtp_loss if res.mtp_loss is not None else mx.array(0.0)
        lm_loss = res.lm_loss if res.lm_loss is not None else res.loss
        z_loss = res.z_loss if res.z_loss is not None else mx.array(0.0)
        moe_loads = res.moe_loads
        if moe_loads is None:
            moe_loads = mx.zeros((0,), dtype=mx.float32)
        loss = res.loss
        if getattr(self.args, 'psr_freeze_base', False):
            loss = res.corrected_loss if res.corrected_loss is not None else mx.stop_gradient(res.loss)
        metrics = res.metrics if res.metrics is not None else mx.zeros((X.shape[0],8,3))
        if res.ncp_metrics is not None:
            metrics = res.ncp_metrics
        return (
            loss / self.args.accumulation_steps,
            mtp_loss,
            moe_loads,
            lm_loss / self.args.accumulation_steps,
            z_loss,
            metrics,
        )

    def _loss_and_grad_with_params(
        self,
        params,
        moe_biases,
        X,
        Y,
        loss_mask,
        attn_mask,
        seg_ids=None, psr_key=0, psr_gate=None,
    ):
        """参数与 MoE bias 显式作为入参的 loss 函数（value_and_grad 的目标）。

        不能用 nn.value_and_grad + mx.compile：那样 params 通过闭包
        （model.trainable_parameters()）被 compile 捕获为常量，梯度永远
        基于初始权重、优化器更新完全无效。显式传参后参数成为运行时输入。
        moe_biases 是 freeze buffer（e_score_correction_bias），同样必须当
        入参：只传 params 时 compile 会把 bias 收成 trace 时的快照，训练
        循环的 noaux_tc 覆写进不了图。
        value_and_grad 只对 params（argnums=0）求梯度。
        """
        self.model.update(params)
        if hasattr(self.model, "apply_moe_biases"):
            self.model.apply_moe_biases(moe_biases)
        return self._loss_fn(X, Y, loss_mask, attn_mask, seg_ids, psr_key, psr_gate)

    def _compute_loss_and_grad(
        self, X, Y, loss_mask, attn_mask, seg_ids=None
    ):
        """一次微批的 value_and_grad；compile 后立刻恢复 params 与 bias。"""
        params = self.model.trainable_parameters()
        eval_params = params
        if self._expl_nest_mu > 0 and self._en_delta is not None:
            # 前瞻副本：梯度在 θ + μ·δ 处求值。model 上的真实参数不动——
            # compiled 路径在函数末尾用未位移的 params 恢复，天然兼容。
            mu = self._expl_nest_mu
            eval_params = tree_map(
                lambda p, d: (p.astype(mx.float32) + mu * d).astype(p.dtype),
                params,
                self._en_delta,
            )
        biases = (
            self.model.moe_bias_stack()
            if hasattr(self.model, "moe_bias_stack")
            else mx.zeros((0,), dtype=mx.float32)
        )
        protected = getattr(getattr(self, 'lm_config', None), 'psr_enabled', False)
        extra = ()
        if protected:
            microstep = getattr(self, '_psr_step', 0)
            extra = (mx.array(microstep + getattr(self.args,'seed',1337),mx.uint32), self.model.psr.calibration_gate)
            self._psr_step = microstep + 1
            self.args.psr_microstep = self._psr_step
        outputs, grads = self._loss_and_grad(eval_params, biases, X, Y, loss_mask, attn_mask, seg_ids, *extra)
        # 无论 eager 还是 compile，f 内部的 model.update / apply_moe_biases 都
        # 只把模块的叶子换成了"trace 期的中间量"（compile 下是无 primitive 的
        # 占位数组，eager 下是 vjp trace 的 tracer）。不恢复的话，下一个微批的
        # model.trainable_parameters() 会把这些残留物当成参数再喂进新 trace，
        # 前向直接变 NaN（实测 SFT 第 2 个微批起 loss=nan）。这里统一用本步
        # 传入的真实参数与 bias 恢复模块状态。
        self.model.update(params)
        if hasattr(self.model, "apply_moe_biases"):
            self.model.apply_moe_biases(biases)
        return outputs, grads

    def _build_loss_and_grad(self):
        fn = mx.value_and_grad(self._loss_and_grad_with_params)
        use_compile = getattr(self.args, "compile_model", False)
        if use_compile:
            # mx.compile 按 trace 冻结 RNG：dropout 一旦开启，每步的随机
            # 掩码恒定不变，此时回退 eager 保证语义正确。
            dropout = float(getattr(self.lm_config, "dropout", 0.0) or 0.0)
            if dropout > 0.0:
                Logger(f"dropout={dropout} 在 mx.compile 下 RNG 被冻结，自动回退 eager")
                use_compile = False
        if use_compile:
            Logger("使用 mx.compile 编译 loss 函数")
            fn = mx.compile(fn)
        self._compiled = use_compile
        return fn

    def _resume_same_stage(self, checkpoint_path) -> bool:
        """跨阶段 checkpoint（pretrain→SFT 等）不继承步数/优化器状态。

        基座权重已由 init_model 严格加载；若把上一阶段的 step 计数器
        续进来，LR 日程会从中间开始、且 epoch 内前 start_step 个微批
        会被整体跳过，只训到尾部数据。sidecar 缺失/无 training_type
        字段时按同阶段处理（兼容旧 checkpoint）。
        """
        base = checkpoint_path
        if base.endswith(".safetensors"):
            base = base[: -len(".safetensors")]
        meta_path = f"{base}.json"
        if not os.path.exists(meta_path):
            return True
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                ckpt_type = json.load(f).get("training_type")
        except Exception:
            return True
        if ckpt_type and ckpt_type != self.training_type:
            Logger(
                f"检查点阶段({ckpt_type})与当前训练({self.training_type})不同，"
                "不继承步数/优化器，从 step 0 开始（基座权重已加载）"
            )
            return False
        return True

    def _handle_checkpoint_resume(self):
        """处理检查点恢复"""
        start_epoch = 0
        start_step = 0

        if self.args.resume:
            if os.path.exists(self.args.resume):
                if self._resume_same_stage(self.args.resume):
                    start_epoch, start_step = load_checkpoint(
                        self.args.resume, self.model, self.optimizer, self.args
                    )
            else:
                Logger(
                    f"Warning: Checkpoint file {self.args.resume} not found, starting from scratch"
                )
        elif getattr(self.args, "auto_resume", False):
            latest_checkpoint = find_latest_checkpoint(self.args.save_dir)
            if latest_checkpoint:
                if self._resume_same_stage(latest_checkpoint):
                    start_epoch, start_step = load_checkpoint(
                        latest_checkpoint,
                        self.model,
                        self.optimizer,
                        self.args,
                    )
            else:
                Logger("No checkpoint found for auto resume, starting from scratch")

        return start_epoch, start_step

    def create_data_loader(self, dataset):
        """创建数据加载器"""
        self._data_loader = MLXDataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return self._data_loader

    def _optimizer_step(self, accum_grads, accum_count, moe_loads=None):
        """累积窗口结束：梯度裁剪 + 优化器更新 + MoE 路由偏置更新"""
        if accum_count <= 0 or accum_grads is None:
            return 0.0

        protected = getattr(getattr(self, 'lm_config', None), 'psr_enabled', False)
        if protected:
            accum_grads, side_grads = split_gradients(accum_grads)
            if side_grads and getattr(self.args,'psr_training_mode','recurrent') != 'off':
                clip = getattr(self.args, 'psr_grad_clip', 1.0)
                if clip:
                    side_grads, norm = optim.clip_grad_norm(side_grads, clip)
                else:
                    norm = mx.sqrt(sum(gradient_square_sum(g) for _,g in tree_flatten(side_grads)))
                if bool(mx.isfinite(norm)):
                    self.psr_optimizer.update(ParameterView(self.model,side=True),side_grads)
                    mx.eval(self.model.psr.parameters(),self.psr_optimizer.state)
                    if getattr(self.args,"log_interval",10) <= getattr(self.args,"accumulation_steps",1):
                        Logger(f"PSR optimizer_step={int(self.psr_optimizer.step)} grad_norm={float(norm):.6g}")
                else:
                    Logger('PSR gradient non-finite: skip side update only; baseline remains independent')
            if getattr(self.args,'psr_freeze_base',False):
                return 0.0
        # 每个微批的 loss 已除以 accumulation_steps，
        # 累加的梯度即窗口平均梯度，无需再除
        # grad_clip=0（默认）表示不裁剪，只算范数用于日志/NaN 防护；
        # 直接调 clip_grad_norm(·, 0) 会把梯度整体缩放到 0
        if self.args.grad_clip and self.args.grad_clip > 0:
            grads, grad_norm = optim.clip_grad_norm(accum_grads, self.args.grad_clip)
        else:
            grads = accum_grads
            # FP32 register accumulation avoids low-precision square underflow,
            # FP16 overflow and a rounded bf16 sum across the parameter tree.
            # Large gradients are read once without full-size FP32 temporaries.
            sums = [gradient_square_sum(g) for _, g in tree_flatten(grads)]
            grad_norm = mx.sqrt(mx.sum(mx.stack(sums))) if sums else mx.array(0.0)

        # NaN/Inf 防护：梯度范数非有限时跳过本窗口更新。单个坏微批前向 NaN
        # 会把窗口累积梯度污染成 NaN，若照常应用，norm=nan 使缩放因子变 nan，
        # 一次 update 即永久污染全部权重与 Adam 二阶矩（r080 no-tie run 实测：
        # step ~1084 单微批前向 NaN → 窗口梯度 NaN → 应用后训练永久崩溃）。
        # 跳过的代价只是损失一个窗口；持续出现说明有系统性数值问题，日志告警。
        grad_norm_val = float(grad_norm)
        if not np.isfinite(grad_norm_val):
            Logger(
                f"梯度范数非有限（{grad_norm_val}），跳过本累积窗口的优化器更新"
                f"与 MoE 偏置更新；若持续出现请排查数据/数值稳定性"
            )
            return grad_norm_val

        if self._expl_nest_mu > 0:
            # MLX 参数数组不可变，update 前抓到的引用即更新前的值；
            # update 替换 module 树上的数组后再抓一次，差分即实际更新 δ。
            en_before = dict(tree_flatten(self.model.trainable_parameters()))
        self.optimizer.update(ParameterView(self.model) if protected else self.model, grads)
        if self._expl_nest_mu > 0:
            en_after = dict(tree_flatten(self.model.trainable_parameters()))
            self._en_delta = tree_unflatten(
                [
                    (k, en_after[k].astype(mx.float32) - pb.astype(mx.float32))
                    for k, pb in en_before.items()
                    if k in en_after
                ]
            )

        # noaux_tc 偏置更新：窗口内各微批的 [L,E] 计数已累加（计数可加），
        # 这里按 b -= γ·sign(load_frac − 1/E) 覆写每个 gate 的
        # e_score_correction_bias。必须在 compile 图外调用：它写的是冻结的
        # buffer，下一次前向作为显式入参进图。
        if moe_loads is not None and getattr(moe_loads, "size", 0) > 0:
            self.model.update_moe_biases(moe_loads)

        if _OPT_CHUNKED_EVAL:
            # 注意：不能在这里 `from mlx.utils import tree_flatten`——那会让
            # tree_flatten 成为本函数的局部变量，遮蔽模块级同名导入，并使本
            # 函数里更早的调用抛 UnboundLocalError。直接用模块级导入即可。
            _chunk, _bytes = [], 0
            for _p in tree_flatten(self.model.parameters()):
                _chunk.append(_p[1])
                _bytes += _p[1].size * _p[1].dtype.size
                if _bytes >= _OPT_CHUNK_BYTES:
                    mx.eval(_chunk)
                    _chunk, _bytes = [], 0
            if _chunk:
                mx.eval(_chunk)
            mx.eval(self.optimizer.state)
        else:
            mx.eval(self.model.parameters(), self.optimizer.state)
        # 注：optimizer step 的临时 buffer 治理由 __init__ 里的
        # mx.set_cache_limit（--cache_limit_gb）统一负责；不要在这里每步
        # mx.clear_cache()——那会把 fwd+bwd 可复用的激活缓存块一并清掉，
        # 实测反而更慢。
        return grad_norm_val

    def _save_if_needed(self, epoch, step):
        if (step + 1) % self.args.save_interval != 0:
            return
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            step,
            self.args,
            self.lm_config,
            self.training_type,
        )

    def train_epoch(
        self,
        epoch,
        train_loader,
        iter_per_epoch,
        total_training_steps,
        swanlab=None,
        skip_steps: int = 0,
    ):
        """训练一个epoch"""
        start_time = time.time()
        self.model.train()

        loader_iter = iter(train_loader)
        try:
            stopped = self._run_epoch_steps(
                loader_iter,
                epoch,
                iter_per_epoch,
                total_training_steps,
                swanlab,
                skip_steps,
                start_time,
                skip_steps,
            )
        finally:
            close = getattr(loader_iter, "close", None)
            if close is not None:
                close()
        return stopped

    def _run_epoch_steps(
        self,
        loader_iter,
        epoch,
        iter_per_epoch,
        total_training_steps,
        swanlab,
        skip_steps,
        start_time,
        base_step_offset_for_speed,
    ):
        accum_grads = None
        accum_count = 0
        last_grad_norm = 0.0
        last_moe_loads = None

        for step, batch in enumerate(loader_iter):
            # 跳过步骤（恢复训练时）
            if step < skip_steps:
                continue
            self._last_epoch = epoch
            self._last_step = step
            shuffle_state = getattr(getattr(self, '_data_loader', None), 'epoch_shuffle_state', None)
            if shuffle_state is not None:
                self.args.epoch_shuffle_state = (shuffle_state[0], shuffle_state[1].tolist(), *shuffle_state[2:])
            # doc_mask 打包模式下 dataset 多返回一项 segment_ids
            if len(batch) == 4:
                X, Y, loss_mask, seg_ids = batch
            else:
                X, Y, loss_mask = batch
                seg_ids = None

            # 应用学习率调度（每微批 step 应用）
            global_step = epoch * iter_per_epoch + step
            apply_lr_schedule(
                self.optimizer,
                global_step,
                total_training_steps,
                self.args.warmup_iters,
                min_lr_ratio=getattr(self.args, "min_lr_ratio", 0.05),
                schedule=getattr(self.args, "lr_schedule", "linear"),
                wsd_decay_frac=getattr(self.args, "wsd_decay_frac", 0.2),
                wsd_decay_shape=getattr(self.args, "wsd_decay_shape", "cosine"),
            )

            # 构造 attention_mask，屏蔽 PAD 位置（模型内部按 bool 掩码处理）
            attn_mask = (X != self.tokenizer.pad_token_id).astype(mx.int32)
            is_log_step = step % self.args.log_interval == 0

            # 前向 + 反向（loss 内部已除以 accumulation_steps）
            # params 与 moe bias 都显式传入：compile 下保证梯度基于当前权重、
            # 路由偏置基于当前快照，而不是 trace 时的常量。
            # mtp_loss 是辅助输出，仅用于日志展示，不参与梯度
            (
                (loss, mtp_loss, moe_loads, lm_loss, z_loss, metrics),
                grads,
            ) = self._compute_loss_and_grad(
                X, Y, loss_mask, attn_mask, seg_ids
            )

            # 立即物化本微批的 loss/grads 并释放反向图。MLX 是惰性求值，
            # 若不 eval，accumulation_steps 个微批的前向+反向图会全部存活到
            # optimizer step，显存按窗口大小成倍增长。
            # 拆成两次 eval：前向输出与梯度分开物化。一次性 eval(loss+grads)
            # 时 MLX 的图调度会把前向 tape 滞留与反向临时量同时顶到峰值
            # （r073 配置实测 46GB/4.0s）；拆开后前向先落定（~14.5GB）再跑
            # 反向（峰值 ~16GB），显存省 ~3×、单步快 ~2×，数值不变。
            mx.eval(loss, mtp_loss, moe_loads, lm_loss, z_loss, metrics)
            mx.eval(grads)
            if getattr(self.lm_config,'psr_enabled',False) and not getattr(self.args,'no_save',False):
                group_steps = get_optimizer_steps(self.optimizer)
                record = dict(microstep=global_step+1, optimizer_step=max(group_steps, default=0),
                              optimizer_group_steps=group_steps, phase="pre_update",
                              consumed_input_tokens=int(mx.sum(attn_mask)),
                              psr_optimizer_step=int(self.psr_optimizer.step) if self.psr_optimizer is not None else 0,
                              sums=mx.sum(metrics,axis=0).tolist())
                with open(os.path.join(self.args.out_dir,'psr_metrics.jsonl'),'a') as file:
                    file.write(json.dumps(record)+'\n')
            if getattr(self.lm_config, "ncp_enabled", False) and not getattr(self.args, "no_save", False):
                values = metrics.tolist()
                record = dict(microstep=global_step+1, optimizer_group_steps=get_optimizer_steps(self.optimizer),
                              ntp_loss=float(lm_loss)*self.args.accumulation_steps,
                              valid_label_count=int(mx.sum(loss_mask)),
                              ncp_loss=values[0], vq_loss=values[1], feedback_coverage=values[2],
                              valid_concepts=values[3], valid_pairs=values[4], codebook_usage=values[5],
                              target_mean_square=values[6], predicted_mean_square=values[7])
                with open(os.path.join(self.args.out_dir,"ncp_metrics.jsonl"),"a") as file:
                    file.write(json.dumps(record)+"\n")
            # P-0/P-1 快照（env 门控，默认零开销）：捕获本微批梯度，
            # 供跨 microbatch 方向相关分析（research/OPTIMIZER_RESEARCH §0.5）
            from . import snapshot

            if snapshot.active():
                acc = self.args.accumulation_steps
                snapshot.maybe_dump_grads(
                    step // acc + 1, step % acc, dict(tree_flatten(grads)), acc
                )
            # noaux_tc 的每专家 token 计数跨微批直接相加（计数可加，不需要
            # 旧 QB 的 margin 样本拼接）：窗口末尾一次性更新路由偏置
            if getattr(moe_loads, "size", 0) > 0:
                last_moe_loads = (
                    moe_loads if last_moe_loads is None else last_moe_loads + moe_loads
                )

            # 梯度累加
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(mx.add, accum_grads, grads)
                # 立即物化累加和。不物化时 accumulation_steps 个 add 会在
                # 计算图上串成惰性链，把窗口内每个微批的梯度树（本配置
                # ~3.1GB/份）全部钉在内存里直到窗口结束（accum=8 ≈ 22GB），
                # allocator 压力让窗口后半段的微批明显变慢。物化后常驻只有
                # 「累加器 + 当前梯度」两份。A/B 开关：VIBY_ACCUM_EAGER=0。
                if _ACCUM_EAGER:
                    mx.eval(accum_grads)
            accum_count += 1

            # 梯度累积窗口结束，执行更新
            if (step + 1) % self.args.accumulation_steps == 0:
                last_grad_norm = self._optimizer_step(
                    accum_grads, accum_count, moe_loads=last_moe_loads
                )
                accum_grads = None
                accum_count = 0
                last_moe_loads = None

                # 时长/步数预算只在窗口边界检查：中途停止会丢掉已累加但未更新的梯度
                if self._time_limit_exceeded():
                    Logger(
                        f"已达最长训练时长 {self.args.max_train_minutes} 分钟，"
                        f"在梯度累积窗口边界（epoch {epoch + 1}, step {step}）停止训练"
                    )
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        step,
                        self.args,
                        self.lm_config,
                        self.training_type,
                    )
                    return True

                max_steps = getattr(self.args, "max_steps", None)
                if max_steps and (epoch * iter_per_epoch + step + 1) >= max_steps:
                    Logger(
                        f"已达最大步数 {max_steps}（微批口径），"
                        f"在梯度累积窗口边界（epoch {epoch + 1}, step {step}）停止训练"
                    )
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        step,
                        self.args,
                        self.lm_config,
                        self.training_type,
                    )
                    return True

            # 日志记录
            if is_log_step:
                # 无论何时记录，都计算当前微批次的原始损失值
                # loss 是已经被 accumulation_steps 缩放过的损失
                # 将其乘回去，就得到了当前单个微批次的原始损失，确保日志值量级一致
                mx.eval(loss)
                current_loss = float(loss.item()) * self.args.accumulation_steps
                # 各 loss 分量（未加权），用于日志与 swanlab
                has_mtp = getattr(self.lm_config, "n_mtp_layers", 0) > 0
                current_mtp_loss = float(mtp_loss.item()) if has_mtp else None
                current_main_loss = float(lm_loss.item()) * self.args.accumulation_steps
                current_z_loss = float(z_loss.item())

                # 使用上次计算的梯度范数
                grad_norm_to_log = last_grad_norm

                # 详细指标（swanlab 上报 + VIBY_DEBUG_MEM stderr 打印共用一份
                # 计算）：MoE 分 gate 负载最大值、内存三件套
                debug_mem = bool(os.environ.get("VIBY_DEBUG_MEM"))
                extra = None
                if swanlab is not None or debug_mem:
                    act = mx.get_active_memory() / 2**30
                    cache = mx.get_cache_memory() / 2**30
                    peak = mx.get_peak_memory() / 2**30
                    # moe_loads 是随 loss 输出物化的 [L,E] 每专家 token 计数
                    # （compile 下侧信道 g._last_load 会被剪枝，不可直接 eval）；
                    # 行序与 self._moe_gates 一致
                    gate_max = (
                        [
                            float(moe_loads[gi].max())
                            for gi in range(
                                min(len(self._moe_gates), int(moe_loads.shape[0]))
                            )
                        ]
                        if getattr(moe_loads, "ndim", 0) == 2
                        else []
                    )
                    extra = {
                        "mem/active_gb": round(act, 3),
                        "mem/cache_gb": round(cache, 3),
                        "mem/peak_gb": round(peak, 3),
                    }
                    for gi, v in enumerate(gate_max):
                        extra[f"moe/gate{gi}_max_load_k"] = round(v / 2**10, 3)
                    if debug_mem:
                        Logger(
                            f"[mem] active={act:.2f}G cache={cache:.2f}G "
                            f"peak={peak:.2f}G "
                            f"gate_maxK={'/'.join(f'{v / 2**10:.1f}' for v in gate_max)}"
                        )

                if getattr(self.lm_config, "ncp_enabled", False):
                    extra = dict(extra or {})
                    for index, name in enumerate(("loss", "vq_loss", "feedback_coverage", "valid_concepts",
                                                   "valid_pairs", "codebook_usage", "target_ms", "predicted_ms")):
                        extra["ncp/"+name] = float(metrics[index])
                    Logger(f"NCP loss={float(metrics[0]):.5f} VQ={float(metrics[1]):.5f} "
                           f"coverage={float(metrics[2]):.3f} usage={float(metrics[5]):.3f}")
                log_training_progress(
                    epoch,
                    step,
                    iter_per_epoch,
                    current_loss,
                    self.optimizer,
                    start_time,
                    self.args,
                    swanlab,
                    grad_norm_to_log,
                    base_step_offset=base_step_offset_for_speed,
                    mtp_loss=current_mtp_loss,
                    main_loss=current_main_loss,
                    z_loss=current_z_loss,
                    extra=extra,
                )
            # 模型保存
            self._save_if_needed(epoch, step)

        return False

    def train(self, train_loader, swanlab=None):
        """主训练循环"""
        iter_per_epoch = len(train_loader)
        total_training_steps = resolve_lr_horizon(self.args, iter_per_epoch)
        resolve_warmup_iters(self.args, total_training_steps)
        Logger(
            f"训练总步数: {total_training_steps}, 每轮步数: {iter_per_epoch}, "
            f"warmup: {self.args.warmup_iters}, "
            f"lr_schedule: {getattr(self.args, 'lr_schedule', 'linear')}, "
            f"min_lr_ratio: {getattr(self.args, 'min_lr_ratio', 0.05)}"
        )
        if getattr(self.args, "max_train_minutes", None):
            Logger(f"最长训练时长: {self.args.max_train_minutes} 分钟")

        self._train_start_time = time.time()
        time_limit_hit = False

        for epoch in range(self.start_epoch, self.args.epochs):
            # 计算需要跳过的步骤
            skip_steps = self.start_step if epoch == self.start_epoch else 0

            try:
                time_limit_hit = self.train_epoch(
                    epoch,
                    train_loader,
                    iter_per_epoch,
                    total_training_steps,
                    swanlab,
                    skip_steps,
                )
            except KeyboardInterrupt:
                self.interrupted = True
                if getattr(self.args, "no_save", False):
                    Logger(
                        "检测到 Ctrl-C：--no_save，直接退出"
                        f"（epoch {self._last_epoch + 1}, step {self._last_step}）"
                    )
                else:
                    Logger(
                        "检测到 Ctrl-C：在最后完成的微批位置保存检查点后退出"
                        f"（epoch {self._last_epoch + 1}, step {self._last_step}）"
                    )
                try:
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self._last_epoch,
                        self._last_step,
                        self.args,
                        self.lm_config,
                        self.training_type,
                    )
                except KeyboardInterrupt:
                    Logger("保存检查点时再次收到 Ctrl-C，跳过本次保存")
                time_limit_hit = True

            # 重置start_step
            if epoch == self.start_epoch:
                self.start_step = 0

            if time_limit_hit:
                break

        # 训练结束保存最后一个检查点（若最后一个步恰好按 save_interval 已保存则跳过；
        # 因时长限制停止时已保存过，不再重复保存）
        if (
            not time_limit_hit
            and self.args.epochs > 0
            and iter_per_epoch > 0
            and iter_per_epoch % self.args.save_interval != 0
        ):
            last_epoch = max(self.start_epoch, self.args.epochs - 1)
            save_checkpoint(
                self.model,
                self.optimizer,
                last_epoch,
                iter_per_epoch - 1,
                self.args,
                self.lm_config,
                self.training_type,
            )
