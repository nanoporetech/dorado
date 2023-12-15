add_library(dorado_basecall STATIC
    crf_utils.h
    crf_utils.cpp
    CRFModel.h
    CRFModel.cpp
    CRFModelConfig.h
    CRFModelConfig.cpp
    ModelRunner.h
    decode/beam_search.cpp
    decode/beam_search.h
    decode/CPUDecoder.cpp
    decode/CPUDecoder.h
)

if (DORADO_GPU_BUILD)
    if(APPLE)
        target_sources(dorado_basecall
            PRIVATE
            MetalCRFModel.h
            MetalCRFModel.cpp
        )
    else()
        target_sources(dorado_basecall
            PRIVATE
            CudaCRFModel.h
            CudaCRFModel.cpp
            decode/GPUDecoder.cpp
            decode/GPUDecoder.h
        )
    endif()
endif()

target_include_directories(dorado_basecall
    SYSTEM
    PUBLIC
        ${DORADO_3RD_PARTY_SOURCE}/toml11
    PRIVATE
        ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
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
endif()
