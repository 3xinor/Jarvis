#pragma once
#include "Weights.h"
#include "KVCache.h"

class Phi3MLP {
private:
    mx::array gate_up_proj; // Quantized models often fuse these
    mx::array down_proj;

public:
    Phi3MLP(WeightLoader& loader, int layer_idx) {
        std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.";
        // Depending on your safetensors format, you map the keys here
    }

    mx::array forward(mx::array x) {
        // x = SiLU(x * gate) * up
        // output = x * down
        return x; // Placeholder logic
    }
};

class TransformerBlock {
public:
    Phi3MLP mlp;
    int layer_idx;

    TransformerBlock(WeightLoader& loader, int idx) : mlp(loader, idx), layer_idx(idx) {}

    mx::array forward(mx::array x, KVCache& cache, mx::array mask) {
        // 1. RMSNorm
        // 2. Attention (QKV extraction, RoPE application, Scaled Dot Product)
        // 3. Add Residual
        // 4. RMSNorm
        // 5. MLP
        // 6. Add Residual
        return x; 
    }
};