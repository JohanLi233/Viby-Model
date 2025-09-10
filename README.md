# Viby

A compact ~1M parameter language model featuring **causal convolution layers** for efficient sequence modeling.

## Key Features

- **Causal Convolution Integration**: Implements Canon layers (A, B, C, D) [PhysicsLM4](https://github.com/facebookresearch/PhysicsLM4)

- Trainable on mps

### Model Specifications
- **Parameters**: ~1M 
- **Hidden Size**: 640
- **Layers**: 18
- **Attention Heads**: 4 (with 1 KV head using GQA)
- **Vocabulary**: 25,600 tokens
- **Context Length**: 1,024 (extendable with YaRN)

### Canon Layers
The model incorporates four types of causal convolution layers:
- **Canon A**: Applied after input normalization in attention blocks
- **Canon B**: Applied to QKV projections in attention
- **Canon C**: Applied after post-attention normalization 
- **Canon D**: Applied to gate and up projections in feed-forward layers


### Causal Convolution Implementation
The Canon layers implement efficient causal 1D convolution with:
- **Caching**: Maintains convolution state for incremental generation
- **Batched Processing**: Supports both prefill and decode phases
- **Device Optimization**: Separate CUDA and MPS kernels for optimal performance
- **Attention Mask Support**: Proper handling of padding tokens

### Training Features
- **Z-Loss**: Auxiliary loss for training stability
- **QK-Clip**: Attention logit clipping for gradient stability with MuonClip optimizer
- **Gradient Accumulation**: Support for large effective batch sizes
- **Mixed Precision**: bfloat16 training support

## Acknowledgments

- **Causal Convolution**: Inspired by the causal convolution implementation in [PhysicsLM4](https://github.com/facebookresearch/PhysicsLM4)
- **Base Architecture**: Built upon [minimind](https://github.com/jingyaogong/minimind)
