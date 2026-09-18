#include "api/caller_creation.h"

#include "modbase/ModBaseCaller.h"

#if DORADO_CUDA_BUILD
#include "basecall/CudaCaller.h"
#elif DORADO_METAL_BUILD
#include "basecall/MetalCaller.h"
#endif

namespace dorado::api {

#if DORADO_CUDA_BUILD
std::shared_ptr<basecall::CudaCaller> create_cuda_caller(
        const basecall::BasecallerCreationParams& params) {
    return std::make_shared<basecall::CudaCaller>(params);
}
#elif DORADO_METAL_BUILD
std::shared_ptr<basecall::MetalCaller> create_metal_caller(
        const config::BasecallModelConfig& model_config,
        float memory_limit_fraction) {
    return basecall::MetalCaller::create(model_config, memory_limit_fraction);
}
#endif

std::shared_ptr<modbase::ModBaseCaller> create_modbase_caller(
        const std::vector<config::ModBaseModelConfig>& model_configs,
        int batch_size,
        const std::string& device) {
    return std::make_shared<modbase::ModBaseCaller>(model_configs, batch_size, device);
}

}  // namespace dorado::api