#pragma once
#include <mlx/mlx.h>
#include <unordered_map>
#include <string>
#include <iostream>

namespace mx = mlx::core;

class WeightLoader {
public:
    std::unordered_map<std::string, mx::array> weights;

    void load(const std::string& safetensors_path);

    mx::array get(const std::string& key);
};
