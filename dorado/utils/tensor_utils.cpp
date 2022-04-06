#include <fstream>
#include <torch/csrc/jit/serialization/pickle.h>
#include "torch/torch.h"

void serialise_tensor(torch::Tensor t, std::string path) {
    auto bytes = torch::jit::pickle_save(t);
    std::ofstream fout(path);
    fout.write(bytes.data(), bytes.size());
    fout.close();
}

std::vector<torch::Tensor> load_weights(std::string dir) {

    auto weights = std::vector<torch::Tensor>();
    auto tensors = std::vector{

            "0.conv.weight.tensor",
            "0.conv.bias.tensor",

            "1.conv.weight.tensor",
            "1.conv.bias.tensor",

            "2.conv.weight.tensor",
            "2.conv.bias.tensor",

            "4.rnn.weight_ih_l0.tensor",
            "4.rnn.weight_hh_l0.tensor",
            "4.rnn.bias_ih_l0.tensor",
            "4.rnn.bias_hh_l0.tensor",

            "5.rnn.weight_ih_l0.tensor",
            "5.rnn.weight_hh_l0.tensor",
            "5.rnn.bias_ih_l0.tensor",
            "5.rnn.bias_hh_l0.tensor",

            "6.rnn.weight_ih_l0.tensor",
            "6.rnn.weight_hh_l0.tensor",
            "6.rnn.bias_ih_l0.tensor",
            "6.rnn.bias_hh_l0.tensor",

            "7.rnn.weight_ih_l0.tensor",
            "7.rnn.weight_hh_l0.tensor",
            "7.rnn.bias_ih_l0.tensor",
            "7.rnn.bias_hh_l0.tensor",

            "8.rnn.weight_ih_l0.tensor",
            "8.rnn.weight_hh_l0.tensor",
            "8.rnn.bias_ih_l0.tensor",
            "8.rnn.bias_hh_l0.tensor",

            "9.linear.weight.tensor",
            "9.linear.bias.tensor"
    };

    for (auto weight : tensors) {
        torch::load(weights, dir + "/" + weight);
    }

    return weights;
}
