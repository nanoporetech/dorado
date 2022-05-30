#pragma once

#include "../decode/Decoder.h"
#include "CRFModel.h"
#include "MetalCRFModel.h"

#include <toml.hpp>
#include <torch/torch.h>

#include <string>

class ModelRunnerBase {
public:
    virtual void accept_chunk(int chunk_idx, at::Tensor slice) = 0;
    virtual std::vector<DecodedChunk> call_chunks(int num_chunks) = 0;
};

using Runner = std::shared_ptr<ModelRunnerBase>;

template <typename T>
class ModelRunner : public ModelRunnerBase {
public:
    ModelRunner(const std::string &model,
                const std::string &device,
                int chunk_size,
                int batch_size);
    void accept_chunk(int chunk_idx, at::Tensor slice) final;
    std::vector<DecodedChunk> call_chunks(int num_chunks) final;

private:
    std::string m_device;
    torch::Tensor m_input;
    torch::TensorOptions m_options;
    std::unique_ptr<T> m_decoder;
    DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
};

template <typename T>
ModelRunner<T>::ModelRunner(const std::string &model,
                            const std::string &device,
                            int chunk_size,
                            int batch_size) {
    auto config = toml::parse(model + "/config.toml");
    const auto &qscore = toml::find(config, "qscore");
    const auto qbias = toml::find<float>(qscore, "bias");
    const auto qscale = toml::find<float>(qscore, "scale");

    m_decoder_options = DecoderOptions();
    m_decoder_options.q_shift = qbias;
    m_decoder_options.q_scale = qscale;
    m_decoder = std::make_unique<T>();

#ifdef __APPLE__
    if (device == "metal") {
        m_options = torch::TensorOptions().dtype(T::dtype).device("cpu");
        m_module = load_crf_mtl_model(model, batch_size, chunk_size, m_options);
    } else {
        m_options = torch::TensorOptions().dtype(T::dtype).device(device);
        m_module = load_crf_model(model, batch_size, chunk_size, m_options);
    }
#else
    m_options = torch::TensorOptions().dtype(T::dtype).device(device);
    m_module = load_crf_model(model, batch_size, chunk_size, m_options);
#endif

    m_input = torch::zeros({batch_size, 1, chunk_size},
                           torch::TensorOptions().dtype(T::dtype).device(torch::kCPU));
}

template <typename T>
std::vector<DecodedChunk> ModelRunner<T>::call_chunks(int num_chunks) {
    torch::InferenceMode guard;
    auto scores = m_module->forward(m_input.to(m_options.device_opt().value()));
    return m_decoder->beam_search(scores, num_chunks, m_decoder_options);
}

template <typename T>
void ModelRunner<T>::accept_chunk(int num_chunks, at::Tensor slice) {
    m_input.index_put_({num_chunks, 0}, slice);
}
