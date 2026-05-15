#pragma once
#include "Weights.h"
#include "KVCache.h"

// --- 1. RMSNorm ---
class RMSNorm {
private:
    mx::array weight;
    float eps;

public:
    RMSNorm(mx::array w, float e = 1e-5f);
    
    mx::array forward(mx::array x);
};

// --- 2. Attention (GQA & RoPE) ---
class Phi3Attention {
private:
    mx::array qkv_proj; 
    mx::array o_proj;
    
    int n_heads = 32;
    int n_kv_heads = 8;
    int head_dim = 96; 
    float scale;

public:
    // REMOVED the default constructor here!
    Phi3Attention(WeightLoader& loader, int layer_idx);

    mx::array forward(mx::array x, KVCache& cache, mx::array mask);
};

// --- 3. Phi3MLP ---
class Phi3MLP {
private:
    mx::array gate_up_proj; 
    mx::array down_proj;

public:
    // REMOVED the default constructor here!
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

    TransformerBlock(WeightLoader& loader, int idx);

    mx::array forward(mx::array x, KVCache& cache, mx::array mask);
};