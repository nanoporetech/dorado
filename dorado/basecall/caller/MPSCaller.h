#pragma once

#include "MetalCaller.h"

namespace dorado::basecall {

class MetalDecoder;
namespace model {
class Model;
}

class MPSCaller final : public MetalCaller {
public:
    explicit MPSCaller(const config::BasecallModelConfig &model_config);
    ~MPSCaller();

    at::Tensor create_input_tensor() const override;

private:
    MPSCaller(MPSCaller &&other) = delete;
    MPSCaller &operator=(MPSCaller &&) = delete;
    MPSCaller(const MPSCaller &) = delete;
    MPSCaller &operator=(const MPSCaller &) = delete;

private:
    DecodedData decode(int chunk_idx) const override;
    bool call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) override;

    std::unique_ptr<model::Model> m_model;
    NS::SharedPtr<MTL::CommandQueue> m_command_queue;

    int m_in_chunk_size, m_out_chunk_size, m_batch_size;
    std::unique_ptr<MetalDecoder> m_decoder;
};

}  // namespace dorado::basecall
