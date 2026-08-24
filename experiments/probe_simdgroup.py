"""探明 mx.fast.metal_kernel 里能用哪些 simdgroup 矩阵形式。

flash 反向的算力全靠 simdgroup_matrix（8×8 片段的硬件 MMA）。但 Metal 对
类型组合的支持随版本而异，而 MLX 的 metal_kernel 是把源码片段塞进它自己的
模板里编译，能不能 #include <metal_simdgroup_matrix>、能不能混精度
（half/bfloat 输入 + float 累加）都得先试出来，否则整个 kernel 架构会建在
错误假设上。

逐个编译并跑一个 8×8 matmul，与 numpy 结果比对。

用法: uv run experiments/probe_simdgroup.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

HDR = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""


def try_variant(name, mat_t, in_dtype, acc_t="float", header=HDR):
    """8×8×8 单片段 matmul：C = A@B，A/B 从 threadgroup 载入。"""
    src = f"""
        uint tid = thread_position_in_grid.x;
        threadgroup {mat_t} As[64];
        threadgroup {mat_t} Bs[64];
        for (uint i = tid; i < 64; i += 32) {{ As[i] = {mat_t}(a[i]); Bs[i] = {mat_t}(b[i]); }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<{mat_t}, 8, 8> A, B;
        simdgroup_matrix<{acc_t}, 8, 8> C;
        C = make_filled_simdgroup_matrix<{acc_t}, 8, 8>(0.0f);
        simdgroup_load(A, As, 8);
        simdgroup_load(B, Bs, 8);
        simdgroup_multiply_accumulate(C, A, B, C);
        threadgroup {acc_t} Cs[64];
        simdgroup_store(C, Cs, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 64; i += 32) out[i] = Cs[i];
    """
    try:
        k = mx.fast.metal_kernel(
            name=f"sgtest_{name}",
            input_names=["a", "b"],
            output_names=["out"],
            source=src,
            header=header,
        )
        a = mx.random.normal((64,)).astype(in_dtype)
        b = mx.random.normal((64,)).astype(in_dtype)
        out = k(
            inputs=[a, b],
            output_shapes=[(64,)],
            output_dtypes=[mx.float32],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
        )[0]
        mx.eval(out)
        ref = (
            a.astype(mx.float32).reshape(8, 8) @ b.astype(mx.float32).reshape(8, 8)
        ).reshape(-1)
        err = mx.abs(out - ref).max().item() / max(mx.abs(ref).max().item(), 1e-9)
        return True, f"最大相对误差 {err:.2e}"
    except Exception as exc:
        msg = str(exc).replace("\n", " ")
        return False, msg[:150]


def try_transpose():
    """simdgroup_load 的 transpose 参数——flash 里 QKᵀ / dSᵀQ 都要它。"""
    src = """
        uint tid = thread_position_in_grid.x;
        threadgroup float As[64];
        threadgroup float Bs[64];
        for (uint i = tid; i < 64; i += 32) { As[i] = a[i]; Bs[i] = b[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 A, B, C;
        C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_load(A, As, 8);
        simdgroup_load(B, Bs, 8, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C, A, B, C);
        threadgroup float Cs[64];
        simdgroup_store(C, Cs, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 64; i += 32) out[i] = Cs[i];
    """
    try:
        k = mx.fast.metal_kernel(
            name="sgtest_transpose",
            input_names=["a", "b"],
            output_names=["out"],
            source=src,
            header=HDR,
        )
        a = mx.random.normal((64,))
        b = mx.random.normal((64,))
        out = k(
            inputs=[a, b],
            output_shapes=[(64,)],
            output_dtypes=[mx.float32],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
        )[0]
        mx.eval(out)
        ref = (a.reshape(8, 8) @ b.reshape(8, 8).T).reshape(-1)
        err = mx.abs(out - ref).max().item() / max(mx.abs(ref).max().item(), 1e-9)
        return True, f"最大相对误差 {err:.2e}"
    except Exception as exc:
        return False, str(exc).replace("\n", " ")[:150]


def try_tgmem_limit():
    """实测单个 threadgroup 能申请多少静态 threadgroup 内存。"""
    _lo, _hi = 8, 128
    ok = 0
    for kb in (16, 24, 28, 30, 32, 48, 64):
        n = kb * 1024 // 4
        src = f"""
            uint tid = thread_position_in_grid.x;
            threadgroup float buf[{n}];
            for (uint i = tid; i < {n}; i += 256) buf[i] = float(i) * a[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) out[0] = buf[{n - 1}];
        """
        try:
            k = mx.fast.metal_kernel(
                name=f"tgmem_{kb}",
                input_names=["a"],
                output_names=["out"],
                source=src,
                header=HDR,
            )
            o = k(
                inputs=[mx.ones((1,))],
                output_shapes=[(1,)],
                output_dtypes=[mx.float32],
                grid=(256, 1, 1),
                threadgroup=(256, 1, 1),
            )[0]
            mx.eval(o)
            ok = kb
        except Exception:
            return ok, kb
    return ok, None


def main():
    print("=== simdgroup_matrix 类型组合 ===")
    for name, mat_t, dt, acc in (
        ("f32", "float", mx.float32, "float"),
        ("half", "half", mx.float16, "float"),
        ("half_acc_half", "half", mx.float16, "half"),
        ("bf16", "bfloat16_t", mx.bfloat16, "float"),
    ):
        ok, msg = try_variant(name, mat_t, dt, acc)
        print(f"  {name:<14}{'可用' if ok else '不可用'}  {msg}")

    print("\n=== simdgroup_load transpose ===")
    ok, msg = try_transpose()
    print(f"  {'可用' if ok else '不可用'}  {msg}")

    print("\n=== threadgroup 内存上限 ===")
    ok_kb, fail_kb = try_tgmem_limit()
    print(
        f"  通过 {ok_kb}KB" + (f"，{fail_kb}KB 失败" if fail_kb else "，测试档位全通过")
    )


if __name__ == "__main__":
    main()
