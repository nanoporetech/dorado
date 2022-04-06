#include "CRFModel.h"
#include "ModelRunnerGPU.h"

ModelRunnerGPU::ModelRunnerGPU(const std::string &model, const std::string &device, int chunk_size, int batch_size, DecoderOptions d_options) {
    m_decoder_options = d_options;
    m_decoder = std::make_unique<GPUDecoder>();
    m_options = torch::TensorOptions().dtype(torch::kF16).device(device);
    m_input = torch::zeros({batch_size, 1, chunk_size}, m_options);
    m_module = load_crf_model(model, batch_size, chunk_size, m_options);
}

std::vector<DecodedChunk> ModelRunnerGPU::call_chunks(int num_chunks) {
    torch::InferenceMode guard;
    m_input = m_input.to(m_options.device_opt().value());
    auto scores = m_module->forward(m_input);
    return m_decoder->beam_search(scores, num_chunks, m_decoder_options);
}

void ModelRunnerGPU::accept_chunk(int num_chunks, at::Tensor slice) {
    m_input.index_put_({num_chunks, 0}, slice);
}
