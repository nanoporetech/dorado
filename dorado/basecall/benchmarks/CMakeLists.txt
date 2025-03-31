add_library(dorado_benchmarks STATIC
    CudaChunkBenchmarks.cpp
    CudaChunkBenchmarks.h
    NVIDIA_A100_80GB_PCIe.cpp
    NVIDIA_A100_80GB_PCIe.h
    NVIDIA_RTX_A6000.cpp
    NVIDIA_RTX_A6000.h
    Orin.cpp
    Orin.h
    Quadro_GV100.cpp
    Quadro_GV100.h
    Tesla_V100-PCIE-16GB.cpp
    Tesla_V100-PCIE-16GB.h
)

set(HOPPER_BENCHMARKS NVIDIA_H100_NVL.h)
if(CUDAToolkit_VERSION VERSION_LESS 12.0)
    list(APPEND HOPPER_BENCHMARKS
        NVIDIA_H100_NVL_cuda11.cpp
    )
else()
    list(APPEND HOPPER_BENCHMARKS
        NVIDIA_H100_NVL_cuda12.cpp
    )
endif()

target_sources(dorado_benchmarks PRIVATE ${HOPPER_BENCHMARKS})

enable_warnings_as_errors(dorado_benchmarks)
