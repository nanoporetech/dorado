add_library(dorado_benchmarks STATIC
    CudaChunkBenchmarks.cpp
    CudaChunkBenchmarks.h
    NVIDIA_A100_80GB_PCIe.cpp
    NVIDIA_A100_80GB_PCIe.h
    NVIDIA_H100_PCIe.cpp
    NVIDIA_H100_PCIe.h
    NVIDIA_RTX_A6000.cpp
    NVIDIA_RTX_A6000.h
    Orin.cpp
    Orin.h
    Quadro_GV100.cpp
    Quadro_GV100.h
    Tesla_V100-PCIE-16GB.cpp
    Tesla_V100-PCIE-16GB.h
)
enable_warnings_as_errors(dorado_benchmarks)
