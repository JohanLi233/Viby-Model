"""目标 2B 配置（MLA + LatentMoE，--no_muonh）单步训练时间分解。

复用 train_pretrain 的真实构建路径（同一 parser / dataset / trainer），
训练循环与 BaseTrainer._run_epoch_steps 同构，在分段处插入 perf_counter
计时。覆盖：数据加载暴露、prep（lr/mask/host sync）、compiled vg 调用、
前向物化、反向物化、margin 拼接、梯度累加、优化器窗口（grad_norm 同步 /
update 构图 / 参数物化 / QB 偏置 / state 物化）、日志路径。

随后做两项归因：
- 优化器分组：MultiOptimizer 各子优化器（Muon / FusedAdamW 标量组含专家栈 /
  embed / router）分别 apply+eval 计时；
- 组件归因：13×MLA 注意力链、13×MoE 链（含 QB margin 收集）、主+MTP
  lm_head+CE，各自 mx.compile(value_and_grad) 计时，与整步 fwd/bwd 对照。

用法:
    .venv/bin/python experiments/prof_target_step.py [windows] [--synthetic]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from dataset.lm_dataset import PretrainDataset
from model.config import VibyConfig
from trainer.base_trainer import BaseTrainer
from trainer.config import get_pretrain_parser, setup_training_args
from trainer.muon import FusedAdamW
from trainer.utils import (
    apply_lr_schedule,
    build_model_and_tokenizer,
    log_training_progress,
    resolve_compute_scaled_hparams,
    resolve_lr_horizon,
    resolve_warmup_iters,
)

TARGET_ARGS = """
--data_path /Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl
--hidden_size 768 --num_hidden_layers 12 --num_attention_heads 12
--kv_lora_rank 256 --qk_rope_head_dim 48 --no-use_linear_attn
--n_routed_experts 384 --num_experts_per_tok 8 --n_shared_experts 2
--moe_intermediate_size 512 --moe_latent_dim 256 --routed_scaling_factor 2.5
--ngram_table_size 0 --mtp_depth 1 --mtp_steps 1
--use_attn_gate --pack_sequences --doc_mask
--batch_size 12 --accumulation_steps 2 --max_seq_len 1024
--epochs 1 --compile_model --cache_limit_gb 0
--log_interval 1 --seed 1337 --no_muonh --no_save
""".split()


def build_everything(argv):
    parser = get_pretrain_parser()
    args = parser.parse_args(argv)
    args = setup_training_args(args, "pretrain")
    lm_config = VibyConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
        mtp_depth=args.mtp_depth,
        mtp_loss_weight=args.mtp_loss_weight,
        mtp_steps=args.mtp_steps,
        use_attn_gate=args.use_attn_gate,
        n_routed_experts=args.n_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        n_shared_experts=args.n_shared_experts,
        moe_intermediate_size=args.moe_intermediate_size,
        routed_scaling_factor=args.routed_scaling_factor,
        moe_router_logit_norm=args.moe_router_logit_norm,
        moe_router_logit_temp=args.moe_router_logit_temp,
        moe_diversity_loss_weight=args.moe_diversity_loss_weight,
        z_loss_weight=args.z_loss_weight,
        moe_latent_dim=args.moe_latent_dim,
        moe_write_spread=getattr(args, "moe_write_spread", False),
        moe_route_scale=getattr(args, "moe_route_scale", False),
        first_k_dense_replace=getattr(args, "first_k_dense_replace", 0),
        **(
            {"dense_intermediate_size": args.dense_intermediate_size}
            if getattr(args, "dense_intermediate_size", None) is not None
            else {}
        ),
        kda_v_head_ratio=args.kda_v_head_ratio,
        ngram_table_size=args.ngram_table_size,
        ngram_layer=args.ngram_layer,
        ngram_logit_skip=getattr(args, "ngram_logit_skip", False),
        ngram_conf_gate=getattr(args, "ngram_conf_gate", False),
        tie_word_embeddings=args.tie_word_embeddings,
        use_linear_attn=args.use_linear_attn,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        attn_res_window=getattr(args, "attn_res_window", 4),
        attn_res_register=getattr(args, "attn_res_register", False),
        attn_res_read_h=getattr(args, "attn_res_read_h", False),
        ihc=getattr(args, "ihc", False),
        ihc_streams=getattr(args, "ihc_streams", 4),
        ihc_typed=getattr(args, "ihc_typed", False),
        ihc_collapse=getattr(args, "ihc_collapse", None),
        ihc_ngram_stream=getattr(args, "ihc_ngram_stream", 1),
    )
    model, tokenizer = build_model_and_tokenizer(lm_config, args)
    t0 = time.perf_counter()
    train_ds = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        pack_sequences=args.pack_sequences,
        doc_mask=args.doc_mask,
    )
    print(f"[setup] dataset 构建 {time.perf_counter() - t0:.1f}s, {len(train_ds)} 样本")
    iter_per_epoch = len(train_ds) // args.batch_size
    args = resolve_compute_scaled_hparams(args, iter_per_epoch)
    trainer = BaseTrainer(args, model, tokenizer, lm_config, "pretrain")
    loader = trainer.create_data_loader(train_ds)
    return args, lm_config, model, tokenizer, trainer, loader, iter_per_epoch


def instrumented_loop(trainer, loader, args, n_windows, synthetic=False):
    """与 _run_epoch_steps 同构的分段计时循环。n_windows 个累积窗口。"""
    model = trainer.model
    accum = args.accumulation_steps
    total_training_steps = resolve_lr_horizon(args, len(loader))
    resolve_warmup_iters(args, total_training_steps)
    B, T = args.batch_size, args.max_seq_len

    rng = np.random.default_rng(0)

    def synth_batch():
        X = mx.array(rng.integers(1, args.vocab_size, size=(B, T), dtype=np.int64))
        Y = mx.array(rng.integers(1, args.vocab_size, size=(B, T), dtype=np.int64))
        loss_mask = mx.ones((B, T), dtype=mx.int64)
        seg = mx.array(
            np.cumsum(rng.random((B, T)) < 1.0 / 340.0, axis=1).astype(np.int64)
        )
        return X, Y, loss_mask, seg

    seg_names = [
        "data",
        "prep",
        "vg_call",
        "fwd",
        "bwd",
        "stats_accum_build",
        "win_norm",
        "win_build",
        "win_eval",
        "log",
    ]
    acc_t = {k: [] for k in seg_names}
    acc_t["window"] = []

    loader_iter = iter(loader)
    start_time = time.time()
    n_mb = (2 + n_windows) * accum  # 2 个 warmup 窗口
    accum_grads = None
    last_moe_stats = None
    last_grad_norm = 0.0
    current_unemb_lar = None

    def mark(bucket, ts):
        acc_t[bucket].append(time.perf_counter() - ts)

    for step in range(n_mb):
        rec = step >= 2 * accum  # 前 2 窗口 warmup 不记录
        tw0 = time.perf_counter()

        ts = time.perf_counter()
        if synthetic:
            batch = synth_batch()
        else:
            batch = next(loader_iter)
        if rec:
            mark("data", ts)
        X, Y, loss_mask, seg_ids = batch

        ts = time.perf_counter()
        apply_lr_schedule(
            trainer.optimizer,
            step,
            total_training_steps,
            args.warmup_iters,
            min_lr_ratio=getattr(args, "min_lr_ratio", 0.05),
            schedule=getattr(args, "lr_schedule", "linear"),
            wsd_decay_frac=getattr(args, "wsd_decay_frac", 0.2),
        )
        attn_mask = (X != trainer.tokenizer.pad_token_id).astype(mx.int32)
        mask_has_pad = bool(mx.any(attn_mask != 1).item())
        is_log_step = step % args.log_interval == 0
        if rec:
            mark("prep", ts)

        ts = time.perf_counter()
        (
            (
                loss,
                mtp_loss,
                moe_stats,
                lm_loss,
                div_loss,
                unemb_lar,
                z_loss,
                load_stats,
            ),
            grads,
        ) = trainer._compute_loss_and_grad(
            X, Y, loss_mask, attn_mask, mask_has_pad, seg_ids, compute_lar=is_log_step
        )
        if rec:
            mark("vg_call", ts)

        ts = time.perf_counter()
        mx.eval(
            loss, mtp_loss, moe_stats, lm_loss, div_loss, unemb_lar, z_loss, load_stats
        )
        if rec:
            mark("fwd", ts)

        ts = time.perf_counter()
        mx.eval(grads)
        if rec:
            mark("bwd", ts)

        # 与 _run_epoch_steps 完全同构：stats 拼接 + 梯度累加均为惰性，
        # 不插入额外 eval（避免改变真实同步结构）
        ts = time.perf_counter()
        if last_moe_stats is None or last_moe_stats.size == 0:
            last_moe_stats = moe_stats
        elif moe_stats.size > 0:
            last_moe_stats = mx.concatenate([last_moe_stats, moe_stats], axis=1)
        accum_grads = (
            grads if accum_grads is None else tree_map(mx.add, accum_grads, grads)
        )
        if rec:
            mark("stats_accum_build", ts)

        if (step + 1) % accum == 0:
            # === 与 _optimizer_step 同构 ===
            ts = time.perf_counter()
            g_flat = tree_flatten(accum_grads)
            grad_norm = mx.sqrt(sum(mx.sum(mx.square(g)) for _, g in g_flat))
            last_grad_norm = float(grad_norm)  # sync：累加 add + 范数归约在此执行
            if rec:
                mark("win_norm", ts)

            ts = time.perf_counter()
            trainer.optimizer.update(model, accum_grads)
            if last_moe_stats is not None and last_moe_stats.size > 0:
                model.update_moe_biases(last_moe_stats)
            if rec:
                mark("win_build", ts)

            ts = time.perf_counter()
            mx.eval(model.parameters(), trainer.optimizer.state)
            if rec:
                mark("win_eval", ts)
            accum_grads = None
            last_moe_stats = None

        if is_log_step:
            ts = time.perf_counter()
            current_loss = float(loss.item()) * accum
            has_mtp = getattr(trainer.lm_config, "mtp_depth", 0) > 0
            current_mtp_loss = float(mtp_loss.item()) if has_mtp else None
            current_main_loss = float(lm_loss.item()) * accum
            current_div_loss = float(div_loss.item()) * float(
                getattr(trainer.lm_config, "moe_diversity_loss_weight", 0.0) or 0.0
            )
            current_z_loss = float(z_loss.item())
            if unemb_lar is not None:
                current_unemb_lar = float(unemb_lar.item())
            log_training_progress(
                0,
                step,
                27863,
                current_loss,
                trainer.optimizer,
                start_time,
                args,
                None,
                last_grad_norm,
                mtp_loss=current_mtp_loss,
                main_loss=current_main_loss,
                diversity_loss=current_div_loss,
                z_loss=current_z_loss,
                unemb_lar=current_unemb_lar,
                extra=None,
            )
            if rec:
                mark("log", ts)
        if rec:
            acc_t["window"].append(time.perf_counter() - tw0)

    close = getattr(loader_iter, "close", None)
    if close is not None:
        close()

    n_win = len(acc_t["window"]) // accum
    print(f"\n=== 分段计时（{n_win} 窗口 × {accum} 微批，中位数 ms/微批）===")
    tot = sum(np.median(acc_t[k]) for k in seg_names if acc_t[k])
    for k in seg_names:
        v = acc_t[k]
        if not v:
            continue
        med = np.median(v) * 1e3 / 1  # per event
        # 窗口级事件按 accum 摊到微批
        per_mb_events = len(v) / (n_win * accum)
        share = np.median(v) * per_mb_events / (tot / accum) if tot > 0 else 0
        print(
            f"  {k:<10}{med:>9.1f}ms/次  ×{per_mb_events:.1f}/微批  "
            f"占比 {share * 100:>5.1f}%"
        )
    win = np.median(acc_t["window"]) * 1e3
    print(f"  微批墙钟中位 {win:.1f}ms（→ {B * T / (win / 1e3):.0f} tokens/s）")
    print(f"  峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
    return acc_t


def optimizer_group_breakdown(trainer):
    """对真实梯度按 MultiOptimizer 分组逐一 apply+eval 计时（在末尾跑，会改 state）。"""
    model = trainer.model
    opt = trainer.optimizer
    print("\n=== 优化器分组耗时（一次窗口更新，apply+eval，中位 of 3）===")
    params = model.trainable_parameters()
    dict(tree_flatten(params))
    # 构造一份确定性的假梯度（与真实梯度同形状同 dtype，全 1e-3 尺度）
    grads = tree_map(
        lambda a: (mx.random.normal(a.shape) * 1e-3).astype(a.dtype), params
    )
    mx.eval(grads)
    parts = opt._split_dictionary(grads)
    names = [type(o).__name__ for o in opt.optimizers]
    for name, o, g in zip(names, opt.optimizers, parts):
        flat_g = tree_flatten(g)
        n_params = sum(x.size for _, x in flat_g)
        if not flat_g:
            print(f"  {name:<12} 空组")
            continue
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            newp = o.apply_gradients(g, params)
            mx.eval(newp)
            ts.append(time.perf_counter() - t0)
        med = np.median(ts) * 1e3
        # FusedAdamW 流量口径：读 p,g,m,v 写 p,m,v（bf16=2B）
        gb = n_params * 2 * 7 / 2**30 if isinstance(o, FusedAdamW) else 0
        extra = f"  ({gb:.1f}GB → {gb / (med / 1e3):.0f} GB/s)" if gb else ""
        print(f"  {name:<12}{med:>8.1f}ms  {n_params / 1e6:>8.1f}M 参数{extra}")
    del grads
    mx.clear_cache()


def component_breakdown(model, trainer, args):
    """组件级 fwd/bwd 归因：13×MLA 链、13×MoE 链、主+MTP lm_head+CE。"""
    from model.kernels.ce import cross_entropy

    B, T = args.batch_size, args.max_seq_len
    D = model.config.hidden_size
    V = model.config.vocab_size
    print(f"\n=== 组件归因（B{B}×T{T}，compiled value_and_grad，min of 4）===")
    x0 = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    C = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    labels = mx.random.randint(0, V, (B, T))
    loss_mask = mx.ones((B, T), dtype=mx.int64)
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same_doc = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    bias = mx.where((same_doc & tril[None])[:, None], 0.0, -1e9).astype(mx.bfloat16)
    pos = model.model.position_embeddings(0, T, mx.bfloat16)
    mx.eval(x0, C, labels, bias, pos[0], pos[1])

    def timed_vg(vg, extra_outputs=False, iters=4, warm=1):
        """vg(x, params) -> (loss[, extra]), grads；分开 eval 得 fwd/bwd。"""
        fwd_ts, tot_ts = [], []
        for i in range(warm + iters):
            t0 = time.perf_counter()
            out, grads = vg()
            if extra_outputs:
                mx.eval(*out)
            else:
                mx.eval(out)
            t1 = time.perf_counter()
            mx.eval(grads)
            t2 = time.perf_counter()
            if i >= warm:
                fwd_ts.append(t1 - t0)
                tot_ts.append(t2 - t0)
        return min(fwd_ts) * 1e3, (min(tot_ts) - min(fwd_ts)) * 1e3

    def chain_vg(mods, call, collect_gates=False):
        params = [m.trainable_parameters() for m in mods]

        def fn(x, ps):
            for m, p in zip(mods, ps):
                m.update(p)
            if collect_gates:
                for g in model.moe_gates():
                    g.last_margins = None
                    g.last_load = None
            h = x
            for m in mods:
                h = call(m, h)
            loss = (h.astype(mx.float32) * C.astype(mx.float32)).sum()
            if collect_gates:
                return loss, model.qb_margin_stats()
            return loss

        vg = mx.compile(mx.value_and_grad(fn, argnums=(0, 1)))
        return timed_vg(lambda: vg(x0, params), extra_outputs=collect_gates)

    attn_mods = [l.self_attn for l in model.model.stack.layers] + [
        model.mtp_modules[0].block.self_attn
    ]
    fa, ba = chain_vg(
        attn_mods,
        lambda m, h: m(
            h, position_embeddings=pos, causal_bias=bias, mask_is_full=False
        )[0],
    )
    print(
        f"  {'13×MLA 注意力':<16} fwd {fa:7.1f}ms  bwd {ba:7.1f}ms  合计 {fa + ba:7.1f}ms"
    )

    moe_mods = [l.mlp for l in model.model.stack.layers] + [
        model.mtp_modules[0].block.mlp
    ]
    for g in model.moe_gates():
        g.collect_stats = True
    fm, bm = chain_vg(moe_mods, lambda m, h: m(h), collect_gates=True)
    print(
        f"  {'13×MoE(含QB统计)':<16} fwd {fm:7.1f}ms  bwd {bm:7.1f}ms  "
        f"合计 {fm + bm:7.1f}ms"
    )

    # lm_head + CE（主 LM 与 MTP 各一次，形状相同）
    W = model.lm_head.weight

    def ce_fn(h, w):
        logits = h @ w.T
        ce, z = cross_entropy(logits, labels, mask=loss_mask, return_z=True)
        return ce + 1e-4 * z

    vg = mx.compile(mx.value_and_grad(ce_fn, argnums=(0, 1)))
    fc, bc = timed_vg(lambda: vg(x0, W))
    print(
        f"  {'lm_head+CE ×2':<16} fwd {2 * fc:7.1f}ms  bwd {2 * bc:7.1f}ms  "
        f"合计 {2 * (fc + bc):7.1f}ms（单次 {fc:.1f}/{bc:.1f}）"
    )
    return (fa, ba), (fm, bm), (fc, bc)


def data_loader_cost(loader, n=24):
    """GPU 空闲时纯数据准备耗时（预取线程产出速率上限）。"""
    it = iter(loader)
    # 先让预取队列填满
    t0 = time.perf_counter()
    next(it)
    t_first = time.perf_counter() - t0
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        next(it)
        ts.append(time.perf_counter() - t0)
    it.close()
    print(
        f"\n=== 数据加载（GPU 空闲、队列预热后）===\n"
        f"  首批 {t_first * 1e3:.0f}ms（冷启动），之后 next() 中位 "
        f"{np.median(ts) * 1e3:.1f}ms / max {max(ts) * 1e3:.1f}ms"
    )


def micro_probes(model, trainer, args):
    """窗口级子项的孤立测量 + 候选优化对拍：
    1) grad_norm  eager 现状 vs mx.compile 融合（数值差）；
    2) QB _col_quantile  全量 sort vs partition+尾部 min（逐位对拍）；
    3) mx.eval 阻塞语义 sanity。
    """
    params = model.trainable_parameters()
    grads = tree_map(
        lambda a: (mx.random.normal(a.shape) * 1e-3).astype(a.dtype), params
    )
    mx.eval(grads)
    g_flat = tree_flatten(grads)

    print("\n=== 微探针 ===")
    # 0) eval 阻塞语义：大 GEMM 后 eval 返回时间应 ≈ GPU 时间
    a = mx.random.normal((8192, 8192)).astype(mx.bfloat16)
    mx.eval(a)
    t0 = time.perf_counter()
    b = a @ a
    t_submit = time.perf_counter() - t0
    t0 = time.perf_counter()
    mx.eval(b)
    t_eval = time.perf_counter() - t0
    print(
        f"  [eval语义] 8192³ bf16 GEMM: 构图 {t_submit * 1e3:.2f}ms / "
        f"mx.eval {t_eval * 1e3:.2f}ms（eval 阻塞则后者≈GPU 时间）"
    )
    del a, b

    # 1) grad_norm：eager（现状，square 逐 tensor 物化）vs 编译融合
    def norm_eager():
        return mx.sqrt(sum(mx.sum(mx.square(g)) for _, g in g_flat))

    @mx.compile
    def norm_fused(gs):
        return mx.sqrt(sum(mx.sum(mx.square(g)) for g in gs))

    def norm_compile():
        return norm_fused([g for _, g in g_flat])

    v_ref = float(norm_eager())
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        float(norm_eager())
        ts.append(time.perf_counter() - t0)
    t_eager = np.median(ts) * 1e3
    v_new = float(norm_compile())
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        float(norm_compile())
        ts.append(time.perf_counter() - t0)
    t_new = np.median(ts) * 1e3
    print(
        f"  [grad_norm] eager {t_eager:.1f}ms vs compile {t_new:.1f}ms；"
        f"数值差 {abs(v_ref - v_new):.3e}（rel {abs(v_ref - v_new) / v_ref:.2e}）"
    )
    del grads
    mx.clear_cache()

    # 2) QB 分位数：全量 sort vs 生产路径（partition + 尾部 min）
    from model.moe import _col_quantile

    q = 1.0 - 8 / 384
    m = (mx.random.normal((24576, 384)) * 0.1).astype(mx.bfloat16)
    mx.eval(m)
    n = m.shape[0]
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo

    def qb_sort(x):
        s = mx.sort(x, axis=0)
        return s[lo].astype(mx.float32) * (1.0 - frac) + s[hi].astype(mx.float32) * frac

    def qb_part(x):
        return _col_quantile(x, q)

    r_sort = qb_sort(m)
    r_part = qb_part(m)
    mx.eval(r_sort, r_part)
    same = bool(mx.all(r_sort == r_part).item())
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval(qb_sort(m))
        ts.append(time.perf_counter() - t0)
    t_sort = np.median(ts) * 1e3
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval(qb_part(m))
        ts.append(time.perf_counter() - t0)
    t_part = np.median(ts) * 1e3
    print(
        f"  [QB quantile] (24576,384) bf16: sort {t_sort:.1f}ms vs partition "
        f"{t_part:.1f}ms；逐位一致={same}"
    )
    ms = mx.stack([m] * 13)
    mx.eval(ms)

    def qb_all_sort():
        s = mx.sort(ms, axis=1)
        return (
            s[:, lo].astype(mx.float32) * (1.0 - frac)
            + s[:, hi].astype(mx.float32) * frac
        )

    def qb_all_part():
        return _col_quantile(ms, q, axis=1)

    t0 = time.perf_counter()
    mx.eval(qb_all_sort())
    t_sort13 = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    mx.eval(qb_all_part())
    t_part13 = (time.perf_counter() - t0) * 1e3
    print(
        f"  [QB quantile] 13 gates 整窗口: sort {t_sort13:.1f}ms "
        f"vs partition {t_part13:.1f}ms"
    )
    del m, ms
    mx.clear_cache()

    # 3) FusedAdamW kernel 带宽：单个大专家栈 (384,1024,256) 逐 tensor 更新
    from trainer.muon import _adamw_kernel

    fn = _adamw_kernel(0.9, 0.99981, 1.142e-15, 0.0, True)
    for shape in [(384, 1024, 256), (384, 256, 512)]:
        p = (mx.random.normal(shape) * 0.02).astype(mx.bfloat16)
        g = (mx.random.normal(shape) * 1e-3).astype(mx.bfloat16)
        mm = mx.zeros_like(p)
        vv = mx.zeros_like(p)
        lr = mx.array(1e-3).astype(mx.bfloat16)
        step = mx.array(10)
        mx.eval(p, g, mm, vv)
        ts = []
        for _ in range(6):
            t0 = time.perf_counter()
            mx.eval(fn(p, g, mm, vv, lr, step))
            ts.append(time.perf_counter() - t0)
        med = np.median(ts) * 1e3
        gb = p.size * 2 * 7 / 2**30
        print(
            f"  [adamw kernel] {str(shape):<18} {med:6.1f}ms  "
            f"{gb:.2f}GB → {gb / (med / 1e3):.0f} GB/s"
        )
        del p, g, mm, vv
    # 参考：纯拷贝带宽上限（读 1 写 1）
    big = (mx.random.normal((384, 1024, 256))).astype(mx.bfloat16)
    mx.eval(big)
    ts = []
    for _ in range(6):
        t0 = time.perf_counter()
        mx.eval(big + big)
        ts.append(time.perf_counter() - t0)
    med = np.median(ts) * 1e3
    gb = big.size * 2 * 3 / 2**30
    print(
        f"  [参考] add 流式 kernel: {med:.1f}ms  {gb:.2f}GB → {gb / (med / 1e3):.0f} GB/s"
    )
    del big
    mx.clear_cache()


def main():
    argv = sys.argv[1:]
    synthetic = "--synthetic" in argv
    argv = [a for a in argv if a != "--synthetic"]
    quick = "--quick" in argv  # 只跑分段循环，跳过归因/微探针
    argv = [a for a in argv if a != "--quick"]
    # "--" 之后的参数覆盖 TARGET_ARGS（同名后置生效），便于扫 cache_limit 等
    override = []
    if "--" in argv:
        i = argv.index("--")
        override = argv[i + 1 :]
        argv = argv[:i]
    n_windows = int(argv[0]) if argv else 8
    args, lm_config, model, tokenizer, trainer, loader, iter_per_epoch = (
        build_everything(TARGET_ARGS + override)
    )
    print(
        f"开始分段循环：{n_windows} 窗口（+2 warmup），"
        f"{'合成 batch' if synthetic else '真实数据 loader'}"
    )
    instrumented_loop(trainer, loader, args, n_windows, synthetic=synthetic)
    if quick:
        return
    if not synthetic:
        data_loader_cost(loader)
    optimizer_group_breakdown(trainer)
    micro_probes(model, trainer, args)
    component_breakdown(model, trainer, args)


if __name__ == "__main__":
    main()
