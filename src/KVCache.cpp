#include "KVCache.h"

KVCache::KVCache(int batch_size, int n_kv_heads, int head_dim, mx::Dtype dtype) {
    offset = 0;
    max_tokens = 2048; 
    
    // Pre-allocate the entire block
    keys = mx::zeros({batch_size, max_tokens, n_kv_heads, head_dim}, dtype);
    values = mx::zeros({batch_size, max_tokens, n_kv_heads, head_dim}, dtype);
}
