#include "Weights.h"

void WeightLoader::load(const std::string& safetensors_path) {
    // 1. Load the struct that contains BOTH the tensors and the metadata
    auto load_result = mx::load_safetensors(safetensors_path);
    
    // 2. Extract ONLY the tensor map (.arrays) and assign it to our variable
    weights = load_result.first; // .first contains the tensor map, .second contains the metadata
    
    std::cout << "[Brain] Loaded " << weights.size() << " tensors from " << safetensors_path << std::endl;
}

mx::array WeightLoader::get(const std::string& key) {
    if (weights.find(key) == weights.end()) {
        std::cerr << "Fatal: Missing weight " << key << std::endl;
        exit(1);
    }
    // Use .at(key) instead of [key]
    return weights.at(key);
}
