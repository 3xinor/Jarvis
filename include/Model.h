#pragma once
#include "Weights.h"
#include "KVCache.h"

// --- 1. RMSNorm ---
class RMSNorm {
private:
    mx::array weight;
    float eps;

public:
    // Default constructor needed for TransformerBlock initialization
    RMSNorm() : eps(1e-5f) {} 
    RMSNorm(mx::array w, float e = 1e-5f);
    
    mx::array forward(mx::array x);
};

// --- 2. Attention (GQA & RoPE) ---
class Phi3Attention {
private:
    // Many MLX quantized models fuse Q, K, and V into a single matrix for speed
    mx::array qkv_proj; 
    mx::array o_proj;
    
    int n_heads = 32;
    int n_kv_heads = 8;
    int head_dim = 96; // 3072 / 32
    float scale;

public:
    Phi3Attention() {} // Default
    Phi3Attention(WeightLoader& loader, int layer_idx);

    mx::array forward(mx::array x, KVCache& cache, mx::array mask);
};

// --- 3. Phi3MLP (Fully Declared) ---
class Phi3MLP {
private:
    mx::array gate_up_proj; // Quantized models often fuse these
    mx::array down_proj;

public:
    Phi3MLP() {} // Default constructor
    Phi3MLP(WeightLoader& loader, int layer_idx);

    mx::array forward(mx::array x);
};

// --- 4. TransformerBlock ---
class TransformerBlock {
public:
    RMSNorm input_layernorm;
    Phi3Attention attention;
    RMSNorm post_attention_layernorm;
    Phi3MLP mlp;
    int layer_idx;

    // We must pass the loader and index down to the components
    TransformerBlock(WeightLoader& loader, int idx);

    mx::array forward(mx::array x, KVCache& cache, mx::array mask);
};
