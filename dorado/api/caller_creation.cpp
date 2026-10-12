#include "api/caller_creation.h"

#include "basecall/CPUCaller.h"
#include "basecall/ModelCaller.h"
#include "modbase/ModBaseCaller.h"

#include <stdexcept>

#if DORADO_CUDA_BUILD
#include "basecall/CudaCaller.h"
#elif DORADO_METAL_BUILD
#include "basecall/MetalCaller.h"
#endif

namespace dorado::api {

std::shared_ptr<basecall::ModelCaller> create_basecall_caller(
        const basecall::BasecallerCreationParams& params) {
    if (params.device == "cpu") {
        return std::make_shared<basecall::CPUCaller>(params.model_config);
    }
#if DORADO_CUDA_BUILD
    return std::make_shared<basecall::CudaCaller>(params);
#elif DORADO_METAL_BUILD
    if (params.device == "metal") {
        return basecall::MetalCaller::create(params.model_config, params.memory_limit_fraction);
    }
#endif
    throw std::runtime_error("Unsupported device: " + params.device);
}

std::shared_ptr<modbase::ModBaseCaller> create_modbase_caller(
        const std::vector<config::ModBaseModelConfig>& model_configs,
        int batch_size,
        const std::string& device) {
    return std::make_shared<modbase::ModBaseCaller>(model_configs, batch_size, device);
}

}  // namespace dorado::api
