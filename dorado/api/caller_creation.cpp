#include "caller_creation.h"

#if DORADO_CUDA_BUILD
#include "basecall/CudaCaller.h"
#elif DORADO_METAL_BUILD
#include "basecall/MetalCaller.h"
#endif

#include "modbase/ModBaseCaller.h"

namespace dorado::api {

#if DORADO_CUDA_BUILD
std::shared_ptr<basecall::CudaCaller> create_cuda_caller(
        const basecall::CRFModelConfig& model_config,
        const std::string& device,
        float memory_limit_fraction,
        PipelineType pipeline_type,
        float batch_size_time_penalty) {
    return std::make_shared<basecall::CudaCaller>(model_config, device, memory_limit_fraction,
                                                  pipeline_type, batch_size_time_penalty);
}
#elif DORADO_METAL_BUILD
std::shared_ptr<basecall::MetalCaller> create_metal_caller(
        const basecall::CRFModelConfig& model_config,
        float memory_limit_fraction) {
    return std::make_shared<basecall::MetalCaller>(model_config, memory_limit_fraction);
}
#endif

std::shared_ptr<modbase::ModBaseCaller> create_modbase_caller(
        const std::vector<std::filesystem::path>& model_paths,
        int batch_size,
        const std::string& device) {
    return std::make_shared<modbase::ModBaseCaller>(model_paths, batch_size, device);
}

}  // namespace dorado::api