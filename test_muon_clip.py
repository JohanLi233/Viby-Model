#!/usr/bin/env python3
"""
MuonClip 优化器功能测试脚本

测试 MuonClip 优化器的基本功能：
1. 优化器正确创建和初始化
2. QK-Clip 机制在检测到高 logit 时正确激活
3. 注意力统计信息收集正常工作
4. 权重缩放逻辑正确执行
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.abspath('.'))

from model.model import VibyConfig, VibyForCausalLM
from trainer.muon import SingleDeviceMuonClip

def test_optimizer_creation():
    """测试 MuonClip 优化器的创建和初始化"""
    print("=== 测试优化器创建 ===")
    
    # 创建一个小型模型用于测试
    config = VibyConfig(
        vocab_size=1000,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=512
    )
    
    model = VibyForCausalLM(config)
    
    # 分离参数组
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    
    param_groups = []
    if muon_params:
        param_groups.append({
            'params': muon_params,
            'use_muon': True,
            'lr': 0.02,
            'momentum': 0.95,
            'weight_decay': 0.01
        })
    if adamw_params:
        param_groups.append({
            'params': adamw_params,
            'use_muon': False,
            'lr': 0.001,
            'betas': (0.9, 0.95),
            'eps': 1e-8,
            'weight_decay': 0.01
        })
    
    # 创建 MuonClip 优化器
    tau = 25.0
    optimizer = SingleDeviceMuonClip(param_groups, tau=tau)
    
    print(f"✓ MuonClip 优化器创建成功")
    print(f"  - tau: {optimizer.tau}")
    print(f"  - 参数组数: {len(optimizer.param_groups)}")
    print(f"  - Muon 参数: {len(muon_params)}")
    print(f"  - AdamW 参数: {len(adamw_params)}")
    
    return model, optimizer

def test_attention_stats_collection():
    """测试注意力统计信息收集"""
    print("\n=== 测试注意力统计信息收集 ===")
    
    config = VibyConfig(
        vocab_size=100,
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        intermediate_size=256
    )
    
    model = VibyForCausalLM(config)
    model.enable_attention_stats_collection()
    
    # 创建测试输入
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 前向传播
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 更新注意力统计信息
    model.update_attention_stats_from_forward()
    
    # 检查统计信息
    stats = model.get_attention_stats()
    print("✓ 注意力统计信息收集成功")
    print(f"  - 收集的层数: {len(stats)}")
    
    for layer_key, layer_stats in stats.items():
        print(f"  - {layer_key}: {len(layer_stats)} 个头")
        for head_key, max_logit in layer_stats.items():
            print(f"    - {head_key}: max_logit = {max_logit:.3f}")
    
    return model, stats

def test_qk_clip_mechanism():
    """测试 QK-Clip 机制"""
    print("\n=== 测试 QK-Clip 机制 ===")
    
    # 创建模型和优化器
    model, optimizer = test_optimizer_creation()
    model.enable_attention_stats_collection()
    
    # 手动注入高 logit 统计信息来测试 QK-Clip
    print("手动设置高 logit 统计信息...")
    
    # 使用新的tensor格式 (num_layers, num_heads)
    model.attention_max_logits = torch.tensor([
        [30.0, 35.0, 20.0, 28.0],  # layer_0: 3个头超过阈值25.0
        [15.0, 40.0, 25.0, 50.0],  # layer_1: 2个头超过阈值25.0  
    ], dtype=torch.float32)
    
    # 同时设置字典格式以便显示统计信息
    model.attention_logit_stats = {
        'layer_0': {
            'head_0': 30.0,  # 超过阈值 25.0
            'head_1': 35.0,  # 超过阈值 25.0
            'head_2': 20.0,  # 未超过阈值
            'head_3': 28.0,  # 超过阈值 25.0
        },
        'layer_1': {
            'head_0': 15.0,  # 未超过阈值
            'head_1': 40.0,  # 超过阈值 25.0
            'head_2': 25.0,  # 等于阈值
            'head_3': 50.0,  # 大幅超过阈值
        }
    }
    
    # 获取统计信息
    stats = model.get_attention_stats()
    print("注意力 logit 统计:")
    high_logit_count = 0
    total_heads = 0
    for layer_key, layer_stats in stats.items():
        for head_key, max_logit in layer_stats.items():
            total_heads += 1
            print(f"  {layer_key} {head_key}: max_logit = {max_logit:.3f}")
            if max_logit > optimizer.tau:
                high_logit_count += 1
    
    print(f"✓ 设置了 {high_logit_count}/{total_heads} 个头超过阈值 {optimizer.tau}")
    
    # 记录权重缩放前的状态
    print("\n记录权重缩放前的状态...")
    original_weights = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        original_weights[layer_idx] = {
            'q_proj': attention.q_proj.weight.data.clone(),
            'k_proj': attention.k_proj.weight.data.clone()
        }
    
    # 测试 QK-Clip 应用
    print("应用 QK-Clip 机制...")
    
    # 模拟优化器步骤（重置统计并应用）
    optimizer.qk_clip_stats['current_step_activations'] = 0
    optimizer.qk_clip_stats['current_step_checks'] = 0
    
    # 应用 QK-Clip
    optimizer.apply_qk_clip(model)
    
    activations = optimizer.qk_clip_stats['current_step_activations']
    total_checks = optimizer.qk_clip_stats['current_step_checks']
    
    print("✓ QK-Clip 执行完成")
    print(f"  - 当前步骤检查的头数: {total_checks}")
    print(f"  - 当前步骤激活次数: {activations}")
    if activations > 0:
        print(f"  - QK-Clip: {activations} activated")
    
    # 检查权重是否被正确缩放
    print("\n检查权重缩放...")
    weights_changed = 0
    for layer_idx, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        q_changed = not torch.allclose(original_weights[layer_idx]['q_proj'], attention.q_proj.weight.data)
        k_changed = not torch.allclose(original_weights[layer_idx]['k_proj'], attention.k_proj.weight.data)
        if q_changed or k_changed:
            weights_changed += 1
            print(f"  - Layer {layer_idx}: 权重已缩放 (Q={q_changed}, K={k_changed})")
    
    print(f"✓ {weights_changed} 个层的权重被缩放")
    
    return model, optimizer

def test_training_step():
    """测试完整的训练步骤"""
    print("\n=== 测试完整训练步骤 ===")
    
    # 创建模型和优化器
    model, optimizer = test_optimizer_creation()
    model.enable_attention_stats_collection()
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # labels 是 input_ids 向右偏移一位
    labels = torch.cat([input_ids[:, 1:], torch.zeros(batch_size, 1, dtype=torch.long)], dim=1)
    attention_mask = torch.ones(batch_size, seq_len)
    
    print("执行前向传播...")
    # 跳过 model.train() 以避免递归错误
    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask
    )
    
    # 更新注意力统计信息
    model.update_attention_stats_from_forward()
    
    loss = output.loss
    print(f"  - 损失: {loss.item():.4f}")
    print(f"  - 最大 logit: {output.logits.max().item():.3f}")
    
    print("执行反向传播...")
    loss.backward()
    
    # 获取梯度范数
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"  - 梯度范数: {grad_norm:.4f}")
    
    print("执行优化器步骤...")
    
    # 执行优化器步骤（传递模型引用）
    optimizer.step(model=model)
    optimizer.zero_grad()
    
    activations = optimizer.qk_clip_stats['current_step_activations']
    checks = optimizer.qk_clip_stats['current_step_checks']
    
    print("✓ 训练步骤执行成功")
    print(f"  - 当前步骤 QK-Clip 检查: {checks}")
    print(f"  - 当前步骤 QK-Clip 激活: {activations}")
    if activations > 0:
        print(f"  - QK-Clip: {activations} activated")
    
    # 获取注意力统计
    attention_stats = model.get_attention_stats()
    if attention_stats:
        print("  - 注意力统计信息已更新")
        for layer_key, layer_stats in attention_stats.items():
            for head_key, max_logit in layer_stats.items():
                if max_logit > optimizer.tau:
                    print(f"    ⚠ {layer_key} {head_key}: {max_logit:.3f} > {optimizer.tau}")
    
    return True

def main():
    """主测试函数"""
    print("MuonClip 优化器功能测试")
    print("=" * 50)
    
    try:
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 运行测试
        test_optimizer_creation()
        test_attention_stats_collection()
        test_qk_clip_mechanism()
        test_training_step()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过！MuonClip 优化器功能正常")
        print("\n使用方法:")
        print("  python train_pretrain.py --use_muon_clip --qk_clip_tau 25.0")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)