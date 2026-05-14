#include "Weights.h"

void WeightLoader::load(const std::string& safetensors_path) {
    weights = mx::load(safetensors_path);
    std::cout << "[Brain] Loaded " << weights.size() << " tensors from " << safetensors_path << std::endl;
}

mx::array WeightLoader::get(const std::string& key) {
    if (weights.find(key) == weights.end()) {
        std::cerr << "Fatal: Missing weight " << key << std::endl;
        exit(1);
    }
    return weights[key];
}
