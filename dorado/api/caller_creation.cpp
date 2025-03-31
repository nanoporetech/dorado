#include "caller_creation.h"

#if DORADO_CUDA_BUILD
#include "basecall/CudaCaller.h"
#include "config/BatchParams.h"
#elif DORADO_METAL_BUILD
#include "basecall/MetalCaller.h"
#endif

#include "modbase/ModBaseCaller.h"

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
    if (model_config.is_tx_model()) {
        return std::make_shared<basecall::MetalTxCaller>(model_config);
    }
    return std::make_shared<basecall::MetalLSTMCaller>(model_config, memory_limit_fraction);
}
#endif

std::shared_ptr<modbase::ModBaseCaller> create_modbase_caller(
        const std::vector<std::filesystem::path>& model_paths,
        int batch_size,
        const std::string& device) {
    return std::make_shared<modbase::ModBaseCaller>(model_paths, batch_size, device);
}

}  // namespace dorado::api