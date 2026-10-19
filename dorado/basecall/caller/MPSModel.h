#pragma once

#include "MetalModel.h"
#include "config/BasecallModelConfig.h"

namespace dorado::basecall {

class MetalDecoder;
namespace model {
class Model;
}

class MPSModel final : public MetalModel {
public:
    explicit MPSModel(const config::BasecallModelConfig &model_config);
    ~MPSModel() override;

    at::Tensor create_input_tensor() const override;

private:
    MPSModel(MPSModel &&other) = delete;
    MPSModel &operator=(MPSModel &&) = delete;
    MPSModel(const MPSModel &) = delete;
    MPSModel &operator=(const MPSModel &) = delete;

private:
    DecodedData decode(int chunk_idx) const override;
    bool call_task(MetalCallerTask &task, std::mutex &inter_caller_mutex, int try_count) override;
    MTL::SharedEvent *decode_complete_event() const override {
        return m_decode_complete_event.get();
    }

    const config::BasecallModelConfig m_config;
    std::unique_ptr<model::Model> m_model;
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL::CommandQueue> m_command_queue;
    NS::SharedPtr<MTL::SharedEvent> m_decode_complete_event;

    int m_in_chunk_size, m_out_chunk_size, m_batch_size;
    std::unique_ptr<MetalDecoder> m_decoder;
};

}  // namespace dorado::basecall
