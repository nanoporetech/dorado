add_subdirectory(benchmarks)

add_library(dorado_basecall STATIC
    crf_utils.cpp
    crf_utils.h
    ModelRunner.cpp
    ModelRunner.h
    ModelRunnerBase.cpp
    ModelRunnerBase.h
    decode/beam_search.cpp
    decode/beam_search.h
    decode/CPUDecoder.cpp
    decode/CPUDecoder.h
    decode/Decoder.cpp
    decode/Decoder.h
    nn/CRFModel.cpp
    nn/CRFModel.h
    nn/TxModel.cpp
    nn/TxModel.h
)

if (DORADO_GPU_BUILD)
    if(APPLE)
        target_sources(dorado_basecall
            PRIVATE
            MetalCaller.cpp
            MetalCaller.h
            MetalModelRunner.cpp
            MetalModelRunner.h
            nn/MetalCRFModel.cpp
            nn/MetalCRFModel.h
        )
    else()
        target_sources(dorado_basecall
            PRIVATE
            CudaModelRunner.cpp
            CudaModelRunner.h
            CudaCaller.cpp
            CudaCaller.h
            decode/CUDADecoder.cpp
            decode/CUDADecoder.h
        )
        target_link_libraries(dorado_basecall PRIVATE dorado_benchmarks)
    endif()
endif()

target_include_directories(dorado_basecall
    SYSTEM
    PRIVATE
        ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
)


target_link_libraries(dorado_basecall
    PUBLIC
        torch_lib
        dorado_torch_utils
        dorado_utils
        dorado_config
    PRIVATE
        spdlog::spdlog
)

if (TARGET koi)
    target_link_libraries(dorado_basecall PRIVATE koi)
endif()

enable_warnings_as_errors(dorado_basecall)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_basecall REUSE_FROM dorado_torch_utils)
endif()
