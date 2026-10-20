#pragma once

#include "MetalModel.h"
#include "config/BasecallModelConfig.h"
#include "config/ModBaseModelConfigFwd.h"
#include "decode/ModBaseDecoder.h"
#include "modbase/nn/ModBaseModel.h"

#include <memory>
#include <mutex>
#include <optional>
#include <span>

namespace dorado::basecall {

class MetalDecoder;
namespace model {
class Model;
}

class MPSModel final : public MetalModel {
public:
    MPSModel(const config::BasecallModelConfig &model_config,
             std::span<const config::ModBaseModelConfig> modbase_configs);
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
    std::shared_ptr<ModBaseInfo> modbase_info() const override { return m_modbase_info; }
    void decode_modbase(MetalCallerTask &task, std::span<decode::DecodedChunk> chunks) override;

    const config::BasecallModelConfig m_config;
    std::unique_ptr<model::Model> m_model;
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL::CommandQueue> m_command_queue;
    NS::SharedPtr<MTL::SharedEvent> m_decode_complete_event;

    int m_in_chunk_size, m_out_chunk_size, m_batch_size;
    std::unique_ptr<MetalDecoder> m_decoder;

    decode::ModBaseDecoder m_modbase_decoder;
    std::optional<modbase::ModBaseLatentSpaceModels> m_modbase_modules;
    std::shared_ptr<ModBaseInfo> m_modbase_info;
    // Decode threads may finish different tasks concurrently.
    std::mutex m_modbase_mutex;
};

}  // namespace dorado::basecall
