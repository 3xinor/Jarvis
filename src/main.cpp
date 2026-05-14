#include <iostream>
#include <mlx/mlx.h>

namespace mx = mlx::core;

int main() {
    std::cout << "Booting Jarvis Inference Engine..." << std::endl;

    // Create a simple 2x2 matrix
    mx::array a = mx::array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    
    // Multiply it by 2
    mx::array b = mx::multiply(a, mx::array(2.0f));

    // Evaluate forces MLX to execute the computational graph on the GPU/CPU
    mx::eval(b);

    std::cout << "MLX Matrix Math Test:\n" << b << std::endl;
    std::cout << "All systems nominal. Ready for Phase 2." << std::endl;

    return 0;
}
