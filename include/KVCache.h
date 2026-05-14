#pragma once
#include <mlx/mlx.h>

namespace mx = mlx::core;

struct KVCache {
    mx::array keys;
    mx::array values;
    int offset;
    int max_tokens; // Hard cap to protect 16GB RAM limit

    KVCache(int batch_size, int n_kv_heads, int head_dim, mx::Dtype dtype);
};
