#include "Model.h"
#include <mlx/fast.h> 
#include <cmath>

// --- TransformerBlock Implementation ---

TransformerBlock::TransformerBlock(WeightLoader& loader, int idx) 
    : layer_idx(idx), 
      mlp(loader, idx), 
      attention(loader, idx), // <-- Initialize attention directly here!
      input_layernorm(loader.get("model.layers." + std::to_string(idx) + ".input_layernorm.weight")),
      post_attention_layernorm(loader.get("model.layers." + std::to_string(idx) + ".post_attention_layernorm.weight"))
{
    // Everything is initalized in the member initializer list for efficiency and clarity.
}

mx::array TransformerBlock::forward(mx::array x, KVCache& cache, mx::array mask) {
    // 1. Attention Path
    auto h = input_layernorm.forward(x);
    h = attention.forward(h, cache, mask);
    x = mx::add(x, h); // First Residual

    // 2. MLP Path
    h = post_attention_layernorm.forward(x);
    h = mlp.forward(h);
    x = mx::add(x, h); // Second Residual

    return x;
}

Phi3MLP::Phi3MLP(WeightLoader& loader, int layer_idx) 
    : gate_up_proj(loader.get("model.layers." + std::to_string(layer_idx) + ".mlp.gate_up_proj.weight")),
      down_proj(loader.get("model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight")) {}

mx::array Phi3MLP::forward(mx::array x) {
    auto fused_output = mx::matmul(x, mx::transpose(gate_up_proj));
    auto parts = mx::split(fused_output, 2, -1);
    auto gate = parts[0];
    auto up = parts[1];
    
    auto activated_gate = mx::multiply(gate, mx::sigmoid(gate));
    auto intermediate = mx::multiply(activated_gate, up);
    
    return mx::matmul(intermediate, mx::transpose(down_proj));
}

// --- RMSNorm Implementation ---

RMSNorm::RMSNorm(mx::array w, float e) : weight(w), eps(e) {}

mx::array RMSNorm::forward(mx::array x) {
    // 1. Calculate the variance: mean of squares along the last axis
    auto variance = mx::mean(mx::square(x), -1, true);
    
    // 2. Calculate the inverse square root: 1 / sqrt(variance + epsilon)
    auto inv_rms = mx::rsqrt(mx::add(variance, mx::array(eps)));
    
    // 3. Scale input by the inverse RMS, then multiply by the learned weights
    auto normalized = mx::multiply(x, inv_rms);
    return mx::multiply(normalized, weight);
}

// --- Attention Implementation ---

Phi3Attention::Phi3Attention(WeightLoader& loader, int layer_idx) 
    : qkv_proj(loader.get("model.layers." + std::to_string(layer_idx) + ".self_attn.qkv_proj.weight")),
      o_proj(loader.get("model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight"))
{
    scale = 1.0f / std::sqrt((float)head_dim);
}

mx::array Phi3Attention::forward(mx::array x, KVCache& cache, mx::array mask) {
    int batch_size = x.shape(0);
    int seq_len = x.shape(1);

    // 1. Fused QKV projection & Split
    auto qkv = mx::matmul(x, mx::transpose(qkv_proj));

    int q_size = n_heads * head_dim;
    int kv_size = n_kv_heads * head_dim;
    
    auto q = mx::slice(qkv, {0, 0, 0}, {batch_size, seq_len, q_size});
    auto k = mx::slice(qkv, {0, 0, q_size}, {batch_size, seq_len, q_size + kv_size});
    auto v = mx::slice(qkv, {0, 0, q_size + kv_size}, {batch_size, seq_len, q_size + 2 * kv_size});

    q = mx::reshape(q, {batch_size, seq_len, n_heads, head_dim});
    k = mx::reshape(k, {batch_size, seq_len, n_kv_heads, head_dim});
    v = mx::reshape(v, {batch_size, seq_len, n_kv_heads, head_dim});

    // 2. Apply RoPE (Rotary Positional Embeddings)
    // We rotate Q and K based on their position in the sequence (cache.offset).
    // Phi-3 uses a base of 10000.0f and non-traditional rotation.
    q = mx::fast::rope(q, head_dim, false, 10000.0f, 1.0f, cache.offset);
    k = mx::fast::rope(k, head_dim, false, 10000.0f, 1.0f, cache.offset);

    // 3. Update KV Cache (Lean Memory Strategy)
    std::vector<int> start = {0, cache.offset, 0, 0};
    std::vector<int> end = {batch_size, cache.offset + seq_len, n_kv_heads, head_dim};
    
    // Grab the cache data BEFORE our current offset
    auto k_before = mx::slice(cache.keys, {0, 0, 0, 0}, {batch_size, cache.offset, n_kv_heads, head_dim});
    auto v_before = mx::slice(cache.values, {0, 0, 0, 0}, {batch_size, cache.offset, n_kv_heads, head_dim});
    
    // Grab the empty buffer AFTER our new tokens
    auto k_after = mx::slice(cache.keys, {0, cache.offset + seq_len, 0, 0}, {batch_size, cache.max_tokens, n_kv_heads, head_dim});
    auto v_after = mx::slice(cache.values, {0, cache.offset + seq_len, 0, 0}, {batch_size, cache.max_tokens, n_kv_heads, head_dim});
    
    // Paste them together along axis 1 (the sequence length axis)
    cache.keys = mx::concatenate({k_before, k, k_after}, 1);
    cache.values = mx::concatenate({v_before, v, v_after}, 1);
    
    // Retrieve the active context window so far
    auto active_k = mx::slice(cache.keys, {0, 0, 0, 0}, {batch_size, cache.offset + seq_len, n_kv_heads, head_dim});
    auto active_v = mx::slice(cache.values, {0, 0, 0, 0}, {batch_size, cache.offset + seq_len, n_kv_heads, head_dim});

    // 4. Grouped-Query Attention (GQA) Broadcasting
    // Phi-3 has 32 Q heads but only 8 K/V heads. We must duplicate the K/V heads 
    // 4 times each so they align for the matrix multiplication.
    int repeat_factor = n_heads / n_kv_heads; // 32 / 8 = 4
    active_k = mx::repeat(active_k, repeat_factor, 2); 
    active_v = mx::repeat(active_v, repeat_factor, 2);

    // 5. Transpose for Matrix Math [batch, heads, seq_len, head_dim]
    q = mx::transpose(q, {0, 2, 1, 3});
    active_k = mx::transpose(active_k, {0, 2, 1, 3});
    active_v = mx::transpose(active_v, {0, 2, 1, 3});

    // 6. Scaled Dot-Product Attention
    // Scores = (Q * K^T) / sqrt(head_dim)
    auto scores = mx::matmul(q, mx::transpose(active_k, {0, 1, 3, 2}));
    scores = mx::multiply(scores, mx::array(scale));
    
    // Apply causal mask (prevents looking into the future during training/prompting)
    if (mask.size() > 0) {
        scores = mx::add(scores, mask);
    }
    
    auto weights = mx::softmax(scores, {-1});
    auto context = mx::matmul(weights, active_v);

    // 7. Output Projection
    // Transpose back to [batch, seq_len, heads, head_dim] and flatten
    context = mx::transpose(context, {0, 2, 1, 3});
    auto flat_context = mx::reshape(context, {batch_size, seq_len, n_heads * head_dim});
    
    return mx::matmul(flat_context, mx::transpose(o_proj));
}

