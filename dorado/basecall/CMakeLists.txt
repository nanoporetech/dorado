add_library(dorado_basecall STATIC
    crf_utils.cpp
    crf_utils.h
    CRFModel.cpp
    CRFModel.h
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
)

if (DORADO_GPU_BUILD)
    if(APPLE)
        target_sources(dorado_basecall
            PRIVATE
            MetalCRFModel.cpp
            MetalCRFModel.h
        )
    else()
        target_sources(dorado_basecall
            PRIVATE
            CudaCRFModel.cpp
            CudaCRFModel.h
            decode/GPUDecoder.cpp
            decode/GPUDecoder.h
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
        dorado_models_lib
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
    # these are defined publicly by minimap2, which we're not using.
    # we need to define them here so that the environment matches the one
    # used when building the PCH
    target_compile_definitions(dorado_basecall PRIVATE PTW32_CLEANUP_C PTW32_STATIC_LIB)
endif()
