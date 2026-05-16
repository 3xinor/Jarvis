#include "Model.h"
#include <mlx/fast.h> 
#include <cmath>

// --- QuantizedLinear Implementation ---

QuantizedLinear::QuantizedLinear(WeightLoader& loader, const std::string& prefix)
    // load all 3 pieces of the compressed matrix
    : weight(loader.get(prefix + ".weight")),
      scales(loader.get(prefix + ".scales")),
      biases(loader.get(prefix + ".biases")) {}

mx::array QuantizedLinear::forward(mx::array x) {
    // mx::fast::quantized_matmul uses the M1 GPU to uncompress the math on the fly!
    // The 'true' means transpose the weights. 64 is the group size, and 4 is the bits.
    return mx::quantized_matmul(x, weight, scales, biases, true, 64, 4);
}

// --- TransformerBlock Implementation ---

TransformerBlock::TransformerBlock(WeightLoader& loader, int idx) 
    : layer_idx(idx), 
      mlp(loader, idx), 
      attention(loader, idx),
      input_layernorm(loader.get("model.layers." + std::to_string(idx) + ".input_layernorm.weight")),
      post_attention_layernorm(loader.get("model.layers." + std::to_string(idx) + ".post_attention_layernorm.weight"))
{
}

mx::array TransformerBlock::forward(mx::array x, KVCache& cache, mx::array mask) {
    auto h = input_layernorm.forward(x);
    h = attention.forward(h, cache, mask);
    x = mx::add(x, h); 

    h = post_attention_layernorm.forward(x);
    h = mlp.forward(h);
    x = mx::add(x, h); 

    return x;
}

// --- Phi3MLP Implementation ---

Phi3MLP::Phi3MLP(WeightLoader& loader, int layer_idx) 
    // Notice how we just pass the prefix now, dropping the ".weight"!
    : gate_up_proj(loader, "model.layers." + std::to_string(layer_idx) + ".mlp.gate_up_proj"),
      down_proj(loader, "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj") {}

mx::array Phi3MLP::forward(mx::array x) {
    // Quantized forward pass
    auto fused_output = gate_up_proj.forward(x);
    
    auto parts = mx::split(fused_output, 2, -1);
    auto gate = parts[0];
    auto up = parts[1];
    
    auto activated_gate = mx::multiply(gate, mx::sigmoid(gate));
    auto intermediate = mx::multiply(activated_gate, up);
    
    // Quantized forward pass
    return down_proj.forward(intermediate);
}

// --- RMSNorm Implementation ---

RMSNorm::RMSNorm(mx::array w, float e) : weight(w), eps(e) {}

mx::array RMSNorm::forward(mx::array x) {
    // Upcast to float32 to prevent variance math overflow
    auto x_f32 = mx::astype(x, mx::float32);
    auto variance = mx::mean(mx::square(x_f32), -1, true);
    auto inv_rms = mx::rsqrt(mx::add(variance, mx::array(eps)));
    auto normalized = mx::multiply(x_f32, inv_rms);
    
    // Downcast back to original precision and apply weights
    auto out = mx::astype(normalized, x.dtype());
    return mx::multiply(out, weight);
}

// --- Attention Implementation ---

Phi3Attention::Phi3Attention(WeightLoader& loader, int layer_idx) 
    // Notice how we just pass the prefix now, dropping the ".weight"!
    : qkv_proj(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.qkv_proj"),
      o_proj(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj")
{
    scale = 1.0f / std::sqrt((float)head_dim);
}

mx::array Phi3Attention::forward(mx::array x, KVCache& cache, mx::array mask) {
    int batch_size = x.shape(0);
    int seq_len = x.shape(1);

    auto qkv = qkv_proj.forward(x);

    int q_size = n_heads * head_dim;
    int kv_size = n_kv_heads * head_dim;
    
    auto q = mx::slice(qkv, {0, 0, 0}, {batch_size, seq_len, q_size});
    auto k = mx::slice(qkv, {0, 0, q_size}, {batch_size, seq_len, q_size + kv_size});
    auto v = mx::slice(qkv, {0, 0, q_size + kv_size}, {batch_size, seq_len, q_size + 2 * kv_size});

    q = mx::reshape(q, {batch_size, seq_len, n_heads, head_dim});
    k = mx::reshape(k, {batch_size, seq_len, n_kv_heads, head_dim});
    v = mx::reshape(v, {batch_size, seq_len, n_kv_heads, head_dim});

    // 1. Transpose to [Batch, Heads, SeqLen, HeadDim] so RoPE rotates the words
    q = mx::transpose(q, {0, 2, 1, 3});
    k = mx::transpose(k, {0, 2, 1, 3});

    // 2. Apply RoPE
    q = mx::fast::rope(q, head_dim, true, 10000.0f, 1.0f, cache.offset);
    k = mx::fast::rope(k, head_dim, true, 10000.0f, 1.0f, cache.offset);

    // 3. Transpose BACK to [Batch, SeqLen, Heads, HeadDim] for the KVCacheff
    q = mx::transpose(q, {0, 2, 1, 3});
    k = mx::transpose(k, {0, 2, 1, 3});

    auto k_before = mx::slice(cache.keys, {0, 0, 0, 0}, {batch_size, cache.offset, n_kv_heads, head_dim});
    auto v_before = mx::slice(cache.values, {0, 0, 0, 0}, {batch_size, cache.offset, n_kv_heads, head_dim});
    
    auto k_after = mx::slice(cache.keys, {0, cache.offset + seq_len, 0, 0}, {batch_size, cache.max_tokens, n_kv_heads, head_dim});
    auto v_after = mx::slice(cache.values, {0, cache.offset + seq_len, 0, 0}, {batch_size, cache.max_tokens, n_kv_heads, head_dim});
    
    cache.keys = mx::concatenate({k_before, k, k_after}, 1);
    cache.values = mx::concatenate({v_before, v, v_after}, 1);
    
    auto active_k = mx::slice(cache.keys, {0, 0, 0, 0}, {batch_size, cache.offset + seq_len, n_kv_heads, head_dim});
    auto active_v = mx::slice(cache.values, {0, 0, 0, 0}, {batch_size, cache.offset + seq_len, n_kv_heads, head_dim});

    q = mx::transpose(q, {0, 2, 1, 3});
    active_k = mx::transpose(active_k, {0, 2, 1, 3});
    active_v = mx::transpose(active_v, {0, 2, 1, 3});

    auto scores = mx::matmul(q, mx::transpose(active_k, {0, 1, 3, 2}));
    scores = mx::multiply(scores, mx::array(scale));
    
    if (mask.size() > 0) {
        // Cast mask to match scores dtype so we don't rely on implicit type promotion
        scores = mx::add(scores, mx::astype(mask, scores.dtype()));
    }
    
    // Upcast scores to float32 before softmax to prevent exponential overflow
    auto scores_f32 = mx::astype(scores, mx::float32);
    auto weights_f32 = mx::softmax(scores_f32, std::vector<int>{-1});
    
    // Downcast weights back to float16
    auto weights = mx::astype(weights_f32, scores.dtype());
    
    auto context = mx::matmul(weights, active_v);
    context = mx::transpose(context, {0, 2, 1, 3});
    auto flat_context = mx::reshape(context, {batch_size, seq_len, n_heads * head_dim});

    return o_proj.forward(flat_context);
}

// --- Phi3Model Implementation ---

Phi3Model::Phi3Model(WeightLoader& loader) 
    : embed_tokens(loader.get("model.embed_tokens.weight")),
      norm(loader.get("model.norm.weight")),
      lm_head(loader.get("lm_head.weight"))
{
    // 1. Decompress the input dictionary
    if (embed_tokens.shape(1) == 384) {
        auto scales = loader.get("model.embed_tokens.scales");
        
        // Initialize with zeros first to avoid default constructor error!
        auto biases = mx::zeros(scales.shape(), scales.dtype()); 
        
        // Check if the safetensors file contains the zero-point shifts, overwrite if true
        if (loader.weights.find("model.embed_tokens.biases") != loader.weights.end()) {
            biases = loader.get("model.embed_tokens.biases");
            std::cout << "[SYSTEM] SafeTensors contains Embedded Tokens bias'.\n";
        }
        
        embed_tokens = mx::dequantize(embed_tokens, scales, biases, 64, 4);
        std::cout << "[SYSTEM] Decompressed Token Embeddings to 3072 dimensions.\n";
    }

    // 2. Decompress the final output vocabulary (LM Head)
    if (lm_head.shape(1) == 384) {
        auto scales = loader.get("lm_head.scales");
        
        // Initialize with zeros first to avoid default constructor error!
        auto biases = mx::zeros(scales.shape(), scales.dtype()); 
        
        // Check if the safetensors file contains the zero-point shifts, overwrite if true
        if (loader.weights.find("lm_head.biases") != loader.weights.end()) {
            biases = loader.get("lm_head.biases");
            std::cout << "[SYSTEM] SafeTensors contains LM Head bias'.\n";
        }
        
        lm_head = mx::dequantize(lm_head, scales, biases, 64, 4);
        std::cout << "[SYSTEM] Decompressed LM Head to 3072 dimensions.\n";
    }

    // 3. Initialize the 32 layers
    for (int i = 0; i < 32; ++i) {
        layers.emplace_back(loader, i);
    }
}

mx::array Phi3Model::forward(mx::array inputs, std::vector<KVCache>& caches) {
    auto h = mx::take(embed_tokens, inputs, 0);
    int seq_len = inputs.shape(1);
    mx::array mask = mx::array({}); 

    if (seq_len > 1) {
        std::vector<float> mask_data(seq_len * seq_len, 0.0f);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = i + 1; j < seq_len; ++j) {
                mask_data[i * seq_len + j] = -10000.0f; 
            }
        }
        mask = mx::array(mask_data.data(), {1, 1, seq_len, seq_len});
    }

    // Pass the matching cache to each specific layer
    for (int i = 0; i < layers.size(); ++i) {
        h = layers[i].forward(h, caches[i], mask);
    }

    h = norm.forward(h);

    int batch_size = h.shape(0);
    int hidden_dim = h.shape(2);
    
    auto last_token_h = mx::slice(h, {0, seq_len - 1, 0}, {batch_size, seq_len, hidden_dim});
    auto logits = mx::matmul(last_token_h, mx::transpose(lm_head));

    return logits;
}
