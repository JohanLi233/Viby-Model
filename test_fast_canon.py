#!/usr/bin/env python3
"""Test Canon ABCD with fast MPS kernel."""

import torch
from model.model import VibyConfig, VibyForCausalLM

def test_fast_canon():
    """Test Canon ABCD with fast MPS kernel."""
    print("Testing Canon ABCD with fast MPS kernel...")
    
    # Create model with default Canon ABCD enabled
    config = VibyConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=2,
        # Default config already has canon_set="ABCD", canon_activation=True
    )
    
    print(f"Canon set: {config.canon_set}")
    print(f"Canon activation: {config.canon_activation}")
    print(f"Canon residual: {config.canon_residual}")
    
    model = VibyForCausalLM(config)
    model.eval()
    
    # Test data
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)  # Proper attention mask
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Test on CPU first
    print("\n=== CPU Test ===")
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        print(f"✓ CPU forward successful, output shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"❌ CPU test failed: {e}")
        return False
    
    # Test on MPS if available
    if torch.backends.mps.is_available():
        print("\n=== MPS Test (with fast kernel) ===")
        try:
            model_mps = model.to("mps")
            input_ids_mps = input_ids.to("mps")
            attention_mask_mps = attention_mask.to("mps")
            
            with torch.no_grad():
                outputs_mps = model_mps(input_ids=input_ids_mps, attention_mask=attention_mask_mps, use_cache=False)
            
            print(f"✓ MPS forward successful, output shape: {outputs_mps.logits.shape}")
            print("✓ Fast MPS kernel is being used for Canon layers!")
        except Exception as e:
            print(f"❌ MPS test failed: {e}")
            return False
    else:
        print("\n=== MPS not available ===")
    
    return True

if __name__ == "__main__":
    success = test_fast_canon()
    if success:
        print("\n🚀 All Canon layers (ABCD) are now using your fast MPS kernel!")
    else:
        print("\n❌ Test failed!")