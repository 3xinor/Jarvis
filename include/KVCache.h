#pragma once
#include <mlx/mlx.h>

namespace mx = mlx::core;

struct KVCache {
    mx::array keys;
    mx::array values;
    int offset = 0;
    int max_tokens = 2048; // Hard cap to protect 16GB RAM limit

    KVCache(int batch_size, int n_kv_heads, int head_dim, mx::Dtype dtype) {
        // Pre-allocate the entire block. No dynamic allocation during inference.
        keys = mx::zeros({batch_size, max_tokens, n_kv_heads, head_dim}, dtype);
        values = mx::zeros({batch_size, max_tokens, n_kv_heads, head_dim}, dtype);
    }
};
