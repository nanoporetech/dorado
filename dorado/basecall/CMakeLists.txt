set(LIB_SOURCE_FILES
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
        list(APPEND LIB_SOURCE_FILES
            MetalCRFModel.h
            MetalCRFModel.cpp
        )
    else()
        list(APPEND LIB_SOURCE_FILES
            CudaCRFModel.h
            CudaCRFModel.cpp
            decode/GPUDecoder.cpp
            decode/GPUDecoder.h
        )
    endif()
endif()

add_library(dorado_basecall STATIC ${LIB_SOURCE_FILES})

target_compile_definitions(dorado_basecall
    PUBLIC
        DORADO_GPU_BUILD=$<BOOL:${DORADO_GPU_BUILD}>
)

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
        dorado_models_lib
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
    # these are defined publicly by minimap2, which we're not using.
    # we need to define them here so that the environment matches the one
    # used when building the PCH
    target_compile_definitions(dorado_basecall PRIVATE PTW32_CLEANUP_C PTW32_STATIC_LIB)
endif()
