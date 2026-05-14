#include <mlx/mlx.h>
#include <unordered_map>
#include <string>
#include <iostream>

namespace mx = mlx::core;

class WeightLoader {
public:
    std::unordered_map<std::string, mx::array> weights;

    void load(const std::string& safetensors_path) {
        // Loads .safetensors directly into Unified Memory
        weights = mx::load(safetensors_path);
        std::cout << "[Brain] Loaded " << weights.size() << " tensors from " << safetensors_path << std::endl;
    }

    mx::array get(const std::string& key) {
        if (weights.find(key) == weights.end()) {
            std::cerr << "Fatal: Missing weight " << key << std::endl;
            exit(1);
        }
        return weights[key];
    }
};
