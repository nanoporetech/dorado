#include "torch_utils/torch_utils.h"

#include "compat/compat_utils.h"

#include <c10/util/Backtrace.h>
#include <torch/torch.h>
#include <torch/version.h>

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAAllocatorConfig.h>
#endif  // DORADO_CUDA_BUILD

namespace dorado::utils {

void initialise_torch() {
    // By default Torch spins up a thread per core for every operation that might benefit from OMP. A lot of these
    // operations are small and the threads don't appear to be pooled and so torch continuously spawns and destroys
    // them, significantly hurting performance. Limiting this to the minimum number of threads (1) reduces runtime
    // on one of our 44 core CI machine from ~4mins to ~20sec.
    torch::set_num_threads(1);

    // We don't want empty tensors to be initialised with data since we always overwrite them.
    torch::globalContext().setDeterministicFillUninitializedMemory(false);
}

void make_torch_deterministic() {
#if DORADO_CUDA_BUILD
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
    torch::globalContext().setDeterministicCuDNN(true);
    torch::globalContext().setBenchmarkCuDNN(false);
#endif

    torch::globalContext().setDeterministicAlgorithms(true, false);
}

void set_torch_allocator_max_split_size() {
#if DORADO_CUDA_BUILD

    // Do not re-use smaller chunks of large buffers
    // This prevents small allocations from reusing large sections of cached allocated memory
    // which can lead to OoM errors when the original large allocation is needed again
#if DORADO_ORIN
    // Transformer models fail to reuse the buffers correctly on Orin with the smaller value
    // so increase it to a value that works (see INSTX-9750).
    auto max_split_size_mb = 250;
#else
    auto max_split_size_mb = 25;
#endif
    std::string settings = "max_split_size_mb:" + std::to_string(max_split_size_mb);

    const char* pytorch_cuda_alloc_conf = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (pytorch_cuda_alloc_conf != nullptr) {
        std::string_view str(pytorch_cuda_alloc_conf);
        if (str.find("max_split_size_mb") != std::string_view::npos) {
            // user has set this via env_var - let torch parse and use their value
            return;
        }
        settings += std::string(",") + pytorch_cuda_alloc_conf;
    }

    c10::cuda::CUDACachingAllocator::setAllocatorSettings(settings);
#endif
}

std::string torch_stacktrace() {
    auto trace = c10::get_backtrace();
    if (trace.empty()) {
        trace = "Couldn't get backtrace from torch";
    }
    return trace;
}

}  // namespace dorado::utils
