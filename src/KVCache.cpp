#include "KVCache.h"

// Initialize everything before the brackets open
KVCache::KVCache(int batch_size, int n_kv_heads, int head_dim, mx::Dtype dtype) 
    : offset(0), 
      max_tokens(2048),
      keys(mx::zeros({batch_size, 2048, n_kv_heads, head_dim}, dtype)),
      values(mx::zeros({batch_size, 2048, n_kv_heads, head_dim}, dtype)) 
{
    // empty
}
