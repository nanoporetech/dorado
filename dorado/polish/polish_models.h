#pragma once

#include <torch/torch.h>

#include <memory>

namespace dorado::polisher {

class TorchModel : public torch::nn::Module {
public:
    // This function is virtual and must be overridden by derived classes
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    // // Get the device for this model (CPU or GPU)
    // torch::Device device() const {
    //     for (const auto& p : parameters()) {
    //         if (p.device() != torch::kCPU) {
    //             return p.device();
    //         }
    //     }
    //     return torch::kCPU;
    // }

    // Convert the model to half precision
    virtual void to_half() {
        this->to(torch::kHalf);
        m_half_precision = true;
    }

    // Predict on a batch with device and precision handling
    torch::Tensor predict_on_batch(torch::Tensor x, const torch::Device& device) {
        x = x.to(device);
        if (m_half_precision) {
            x = x.to(torch::kHalf);
        }
        x = forward(std::move(x)).detach().cpu();
        if (m_half_precision) {
            x = x.to(torch::kFloat);
        }
        return x;
    }

protected:
    bool m_half_precision = false;
};

class GRUModel : public TorchModel {
public:
    GRUModel(int num_features, int num_classes, int gru_size, bool normalise = true)
            : m_num_features(num_features),
              m_num_classes(num_classes),
              m_gru_size(gru_size),
              m_normalise(normalise),
              m_gru(torch::nn::GRUOptions(num_features, gru_size)
                            .num_layers(2)
                            .bidirectional(true)
                            .batch_first(true)),
              m_linear(2 * gru_size, num_classes) {
        register_module("gru", m_gru);
        register_module("linear", m_linear);
    }

    torch::Tensor forward(torch::Tensor x) override {
        // for (const auto& name_param_pair : named_parameters()) {
        //     const auto& name = name_param_pair.key();
        //     const auto& param = name_param_pair.value();
        //     // if (param.device() != torch::kCUDA) {
        //         std::cerr << "[Warning] Parameter '" << name << ": param.device() = " << param.device() << "\n";
        //     // }
        // }

        // for (const auto& p : m_gru->parameters()) {
        //     if (p.device() != torch::kCUDA) {
        //         std::cerr << "[gru] Parameter '" << p.name() << "' not on GPU!\n";
        //     }
        // }
        // for (const auto& p : m_linear->parameters()) {
        //     if (p.device() != torch::kCUDA) {
        //         std::cerr << "[linear] Parameter '" << p.name() << " not on GPU!\n";
        //     }
        // }
        // std::cerr << "m_half_precision = " << m_half_precision << "\n";
        // for (const auto& p : this->parameters()) {
        //     if (p.device() != x.device()) {
        //         std::cerr << "Parameter not on expected device!" << std::endl;
        //     }
        // }
        // this->

        x = std::move(std::get<0>(m_gru->forward(x)));
        x = m_linear->forward(x);
        if (m_normalise) {
            x = torch::softmax(x, -1);
        }
        return x;
    }

    // void move_to_device(torch::Device& device) {
    //     m_gru->to(device);
    //     m_linear->to(device);
    //     this->to(device);
    // }

    // private:
    int m_num_features = 10;
    int m_num_classes = 5;
    int m_gru_size = 128;
    bool m_normalise = true;
    torch::nn::GRU m_gru{nullptr};
    torch::nn::Linear m_linear{nullptr};
};

}  // namespace dorado::polisher
