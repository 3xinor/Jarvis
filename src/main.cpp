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

        // Initialize our KV Cache for a batch size of 1, 32 KV heads, and 96 dimensions.
        // We use float32 to match the un-quantized intermediate math.
        KVCache cache(1, 32, 96, mx::float32);

        std::cout << "Jarvis: ";

        // --- 4. THE AUTO-REGRESSIVE GENERATION LOOP ---
        // We feed the current sequence, predict the next token, and append it.
        mx::array current_tokens = mx::array(tokens.data(), {1, (int)tokens.size()});
        
        int max_gen_tokens = 256;
        int eos_token_id = 32000; // Phi-3's <|end|> token ID

        for (int step = 0; step < max_gen_tokens; ++step) {
            
            // Forward pass through all 32 layers
            mx::array logits = model.forward(current_tokens, cache);

            // Get the ID of the highest probability word (argmax)
            auto next_token_array = mx::argmax(logits, -1);

            // CRITICAL: MLX is lazy. This eval() command forces the M1 GPU to 
            // actually execute all the math in the graph we just built.
            mx::eval(next_token_array);

            // Extract the integer from the MLX array
            int next_token_id = next_token_array.item<int>();

            // Stop if Jarvis generates the End-Of-Sequence token
            if (next_token_id == eos_token_id) {
                break;
            }

            // Decode and print the single token to the screen
            std::cout << tokenizer->Decode({next_token_id}) << std::flush;

            // Prepare the generated token to be the input for the next loop
            current_tokens = mx::reshape(next_token_array, {1, 1});
            
            // Advance the cache offset so the model remembers the past!
            if (step == 0) {
                cache.offset += tokens.size(); // First step processes the whole prompt
            } else {
                cache.offset += 1; // Subsequent steps process 1 token at a time
            }
        }
        std::cout << std::endl;
    }

    return 0;
}
