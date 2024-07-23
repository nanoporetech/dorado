add_library(dorado_basecall STATIC
    BasecallerParams.cpp
    BasecallerParams.h
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
            CudaChunkBenchmarks.cpp
            CudaChunkBenchmarks.h
            CudaModelRunner.cpp
            CudaModelRunner.h
            CudaCaller.cpp
            CudaCaller.h
            decode/CUDADecoder.cpp
            decode/CUDADecoder.h
            Quadro_GV100.cpp
            Quadro_GV100.h
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
        torch_lib
        dorado_torch_utils
        dorado_utils
    PRIVATE
        spdlog::spdlog
)

if (TARGET koi)
    target_link_libraries(dorado_basecall PRIVATE koi)

    # Current koi can't be linked to directly since it doesn't provide an include path.
    # TODO: once koi is updated we can remove this.
    if (BUILD_KOI_FROM_SOURCE)
        if (KOI_VERSION VERSION_GREATER "0.4.8")
            message(FATAL_ERROR "This should no longer be necessary")
        endif()
        target_include_directories(dorado_basecall PRIVATE ${KOI_DIR}/koi/lib)
    endif()
endif()

enable_warnings_as_errors(dorado_basecall)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_basecall REUSE_FROM dorado_torch_utils)
endif()
