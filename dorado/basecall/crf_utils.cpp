#include "crf_utils.h"

#include "CRFModelConfig.h"
#include "utils/memory_utils.h"
#include "utils/tensor_utils.h"

#include <algorithm>
#include <thread>

namespace dorado::basecall {
std::vector<at::Tensor> load_crf_model_weights(const std::filesystem::path &dir,
                                               bool decomposition,
                                               bool linear_layer_bias) {
    auto tensors = std::vector<std::string>{

            "0.conv.weight.tensor",      "0.conv.bias.tensor",

            "1.conv.weight.tensor",      "1.conv.bias.tensor",

            "2.conv.weight.tensor",      "2.conv.bias.tensor",

            "4.rnn.weight_ih_l0.tensor", "4.rnn.weight_hh_l0.tensor",
            "4.rnn.bias_ih_l0.tensor",   "4.rnn.bias_hh_l0.tensor",

            "5.rnn.weight_ih_l0.tensor", "5.rnn.weight_hh_l0.tensor",
            "5.rnn.bias_ih_l0.tensor",   "5.rnn.bias_hh_l0.tensor",

            "6.rnn.weight_ih_l0.tensor", "6.rnn.weight_hh_l0.tensor",
            "6.rnn.bias_ih_l0.tensor",   "6.rnn.bias_hh_l0.tensor",

            "7.rnn.weight_ih_l0.tensor", "7.rnn.weight_hh_l0.tensor",
            "7.rnn.bias_ih_l0.tensor",   "7.rnn.bias_hh_l0.tensor",

            "8.rnn.weight_ih_l0.tensor", "8.rnn.weight_hh_l0.tensor",
            "8.rnn.bias_ih_l0.tensor",   "8.rnn.bias_hh_l0.tensor",

            "9.linear.weight.tensor"};

    if (linear_layer_bias) {
        tensors.push_back("9.linear.bias.tensor");
    }

    if (decomposition) {
        tensors.push_back("10.linear.weight.tensor");
    }

    return utils::load_tensors(dir, tensors);
}

size_t auto_calculate_num_runners(const CRFModelConfig &model_config,
                                  size_t batch_size,
                                  float memory_fraction) {
    auto model_name = std::filesystem::canonical(model_config.model_path).filename().string();

    // very hand-wavy determination
    // these numbers were determined empirically by running 1, 2, 4 and 8 runners for each model
    auto required_ram_per_runner_GB = 0.f;
    if (model_name.find("_fast@v") != std::string::npos) {
        required_ram_per_runner_GB = 1.5;
    } else if (model_name.find("_hac@v") != std::string::npos) {
        required_ram_per_runner_GB = 4.5;
    } else if (model_name.find("_sup@v") != std::string::npos) {
        required_ram_per_runner_GB = 12.5;
    } else {
        return 1;
    }

    // numbers were determined with a batch_size of 128, assume this just scales
    required_ram_per_runner_GB *= batch_size / 128.f;

    auto free_ram_GB = utils::available_host_memory_GB() * memory_fraction;
    auto num_runners = static_cast<size_t>(free_ram_GB / required_ram_per_runner_GB);
    return std::clamp(num_runners, size_t(1), std::size_t(std::thread::hardware_concurrency()));
}

}  // namespace dorado::basecall
