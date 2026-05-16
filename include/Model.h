#pragma once
#include "Weights.h"
#include "KVCache.h"

// Quantized Linear Layer ---
class QuantizedLinear {
private:
    mx::array weight;
    mx::array scales;
    mx::array biases;

public:
    QuantizedLinear(WeightLoader& loader, const std::string& prefix);
    mx::array forward(mx::array x);
};

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
    // SWAPPED: Using QuantizedLinear instead of mx::array
    QuantizedLinear qkv_proj; 
    QuantizedLinear o_proj;
    
    int n_heads = 32;
    int n_kv_heads = 32;
    int head_dim = 96; 
    float scale;

public:
    Phi3Attention(WeightLoader& loader, int layer_idx);

    mx::array forward(mx::array x, KVCache& cache, mx::array mask);
};

// --- 3. Phi3MLP ---
class Phi3MLP {
private:
    // SWAPPED: Using QuantizedLinear instead of mx::array
    QuantizedLinear gate_up_proj; 
    QuantizedLinear down_proj;

public:
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

// --- 5. Phi-3 Model Wrapper ---
class Phi3Model {
public:
    mx::array embed_tokens;                 // Turns token IDs into vectors
    std::vector<TransformerBlock> layers;   // The 32 brain layers
    RMSNorm norm;                           // Final normalization
    mx::array lm_head;                      // The final prediction layer

    // Constructor to load everything
    Phi3Model(WeightLoader& loader);

    // The main forward pass
    mx::array forward(mx::array inputs, KVCache& cache);
};
