#include <torch/torch.h>
#include <string>
#include <vector>

class Llama {
public:
  Llama(const std::string& model_path) {
    // Load the model
    torch::jit::script::Module module;
    module = torch::jit::load(model_path);
    model_ = module;
  }

  std::vector<std::string> generate(
      const std::vector<std::string>& prompts,
      int max_gen_len,
      float temperature,
      float top_p) {
    // Create input tensor
    torch::Tensor input_tensor = torch::tensor(prompts);

    // Run the model
    torch::Tensor output_tensor = model_.forward({input_tensor, max_gen_len, temperature, top_p});

    // Get the output strings
    std::vector<std::string> output;
    for (int i = 0; i < output_tensor.size(0); i++) {
      output.push_back(output_tensor[i].item<std::string>());
    }

    return output;
  }

private:
  torch::jit::script::Module model_;
};

int main() {
  Llama model("path/to/model");
  std::vector<std::string> prompts = {"Hello, ", "World!"};
  int max_gen_len = 10;
  float temperature = 0.6;
  float top_p = 0.9;
  std::vector<std::string> output = model.generate(prompts, max_gen_len, temperature, top_p);
  for (const auto& str : output) {
    std::cout << str << std::endl;
  }
  return 0;
}