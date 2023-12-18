#include "caller_creation.h"

#if DORADO_GPU_BUILD && !defined(__APPLE__)
#include "basecall/CudaCaller.h"
#endif

namespace dorado::callers {

#if DORADO_GPU_BUILD && !defined(__APPLE__)
std::shared_ptr<basecall::CudaCaller> create_cuda_caller(
        const basecall::CRFModelConfig &model_config,
        int chunk_size,
        int batch_size,
        const std::string &device,
        float memory_limit_fraction,
        bool exclusive_gpu_access) {
    return std::make_shared<basecall::CudaCaller>(model_config, chunk_size, batch_size, device,
                                                  memory_limit_fraction, exclusive_gpu_access);
}
#endif

}  // namespace dorado::callers