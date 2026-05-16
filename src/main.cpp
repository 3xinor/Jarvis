#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <mlx/mlx.h>
#include "Model.h"
#include "Weights.h"
#include "KVCache.h"
#include <tokenizers_cpp.h> 

namespace mx = mlx::core;
using namespace tokenizers;

// A lightweight scanner to pull numbers out of config.json
std::string extract_val(const std::string& line) {
    std::string res = "";
    for (char c : line) {
        if (std::isdigit(c)) res += c;
    }
    return res;
}

int main() {
    std::cout << "\n======================================================\n";
    std::cout << "                 BOOTING JARVIS v1.0                  \n";
    std::cout << "======================================================\n";

    // --- 1. SET PATHS HERE ---
    std::string model_path = "../models/phi3/model.safetensors"; 
    std::string tokenizer_path = "../models/phi3/tokenizer.json";
    std::string config_path = "../models/phi3/config.json";

    // --- 1.5 DYNAMIC SYSTEM READOUT ---
    std::ifstream c_file(config_path);
    std::string line;
    std::string bits = "16", ctx = "Unknown", heads = "Unknown", layers = "Unknown";
    
    if (c_file.is_open()) {
        while (std::getline(c_file, line)) {
            if (line.find("\"bits\"") != std::string::npos) bits = extract_val(line);
            if (line.find("\"max_position_embeddings\"") != std::string::npos) ctx = extract_val(line);
            if (line.find("\"num_hidden_layers\"") != std::string::npos) layers = extract_val(line);
            if (line.find("\"num_attention_heads\"") != std::string::npos) heads = extract_val(line);
        }
    }

    std::cout << "[SYSTEM] Engine       : C++ / Apple MLX Backend\n";
    std::cout << "[SYSTEM] Precision    : " << bits << "-bit Quantized (Unified Memory Optimized)\n";
    std::cout << "[SYSTEM] Architecture : Phi-3 (" << layers << " Layers, " << heads << " Attention Heads)\n";
    std::cout << "[SYSTEM] Context Size : " << ctx << " Tokens\n";
    std::cout << "======================================================\n\n";

    // --- 2. INITIALIZATION ---
    std::cout << "[SYSTEM] Loading Tokenizer..." << std::endl;
    // Read the tokenizer.json file into a string
    std::ifstream t_file(tokenizer_path);
    std::stringstream t_buffer;
    t_buffer << t_file.rdbuf();
    auto tokenizer = Tokenizer::FromBlobJSON(t_buffer.str());

    std::cout << "[SYSTEM] Loading Neural Network Weights into Unified Memory..." << std::endl;
    WeightLoader loader;
    loader.load(model_path);

    std::cout << "[SYSTEM] Assembling Phi-3 Architecture..." << std::endl;
    Phi3Model model(loader);

    std::cout << "\n[JARVIS IS ONLINE. TYPE 'exit' TO QUIT]\n";
    std::cout << "------------------------------------------------------\n";

    // --- 3. CHAT LOOP ---
    while (true) {
        std::string user_input;
        std::cout << "\nDave: ";
        std::getline(std::cin, user_input);

        if (user_input == "exit" || user_input == "quit") break;

        // Phi-3 Instruct Prompt Format
        // The model was trained to recognize these exact tags to know who is speaking.
        std::string prompt = "<|user|>\n" + user_input + "<|end|>\n<|assistant|>\n";

        // Turn text into integer IDs
        std::vector<int> tokens = tokenizer->Encode(prompt);

        // Phi-3 requires Token ID 1 (<s>) at position 0 to anchor its spatial awareness.
        tokens.insert(tokens.begin(), 1);

        // --- DEBUG PRINT 1: TOKEN INSPECTION ---
        // std::cout << "[DEBUG] Total Input Tokens: " << tokens.size() << "\n";
        // std::cout << "[DEBUG] First 10 Token IDs: ";
        // for(int i = 0; i < std::min(10, (int)tokens.size()); i++) {
        //     std::cout << tokens[i] << " ";
        // }
        // std::cout << "\n----------------------------------------\n";

        // Initialize an array of 32 KV Caches (one for each layer)
        std::vector<KVCache> caches;
        for(int i = 0; i < 32; i++) {
            caches.emplace_back(1, 32, 96, mx::float16);
        }

        std::cout << "Jarvis: ";

        // --- 4. THE AUTO-REGRESSIVE GENERATION LOOP ---
        // We feed the current sequence, predict the next token, and append it.
        mx::array current_tokens = mx::array(tokens.data(), {1, (int)tokens.size()});
        int max_gen_tokens = 256;
        int eos_token_id = 32000; 

        for (int step = 0; step < max_gen_tokens; ++step) {
            
            // Pass the vector of caches
            mx::array logits = model.forward(current_tokens, caches);

            // --- DEBUG PRINT 2: NAN DETECTOR ---
            // Force MLX to evaluate the logits early so we can inspect them
            mx::eval(logits); 
            
            // std::cout << "[DEBUG] Logits Shape: [" << logits.shape(0) << ", " 
            //         << logits.shape(1) << ", " << logits.shape(2) << "]\n";
                    
            // // Summing the logits is a quick trick. If there is a single NaN 
            // // anywhere in the massive array, the sum will evaluate to NaN.
            // float logits_sum = mx::sum(logits).item<float>();
            // std::cout << "[DEBUG] Logits Sum (Should be a number): " << logits_sum << "\n";
            // std::cout << "----------------------------------------\n";

            auto next_token_array = mx::argmax(logits, -1);

            // Evaluate the token AND the state of all 32 caches
            // to prevent the MLX computation graph from infinitely expanding.
            mx::eval(next_token_array);
            for(auto& c : caches) {
                mx::eval(c.keys, c.values);
            }

            int next_token_id = next_token_array.item<int>();
            if (next_token_id == eos_token_id) break;

            std::cout << tokenizer->Decode({next_token_id}) << std::flush;
            current_tokens = mx::reshape(next_token_array, {1, 1});
            
            // Advance the offset for ALL layers
            if (step == 0) {
                for(auto& c : caches) c.offset += tokens.size();
            } else {
                for(auto& c : caches) c.offset += 1;
            }
        }
        std::cout << std::endl;
    }

    return 0;
}
