add_library(dorado_basecall STATIC
    crf_utils.cpp
    crf_utils.h
    CRFModelConfig.cpp
    CRFModelConfig.h
    ModelRunner.cpp
    ModelRunner.h
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
    nn/tensor_utils.cpp
    nn/tensor_utils.h
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
    endif()
endif()

target_include_directories(dorado_basecall
    SYSTEM
    PRIVATE
        ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
        ${DORADO_3RD_PARTY_SOURCE}/toml11
)


target_link_libraries(dorado_basecall
    PUBLIC
        ${TORCH_LIBRARIES}
        dorado_utils
    PRIVATE
        ${KOI_LIBRARIES}
        spdlog::spdlog
)

target_include_directories(dorado_basecall
    PRIVATE
        ${KOI_INCLUDE}
)

enable_warnings_as_errors(dorado_basecall)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_basecall REUSE_FROM dorado_utils)
endif()
