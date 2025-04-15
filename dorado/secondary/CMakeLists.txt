add_library(dorado_secondary
    bam_file.cpp
    bam_file.h
    bam_info.cpp
    bam_info.h
    batching.cpp
    batching.h
    interval.h
    region.cpp
    region.h
    variant.cpp
    variant.h
    vcf_writer.cpp
    vcf_writer.h
    architectures/model_config.cpp
    architectures/model_config.h
    architectures/model_factory.cpp
    architectures/model_factory.h
    architectures/model_gru.cpp
    architectures/model_gru.h
    architectures/model_latent_space_lstm.cpp
    architectures/model_latent_space_lstm.h
    architectures/model_torch_base.cpp
    architectures/model_torch_base.h
    architectures/model_torch_script.cpp
    architectures/model_torch_script.h
    consensus/consensus_result.h
    consensus/consensus_utils.cpp
    consensus/consensus_utils.h
    consensus/sample.cpp
    consensus/sample.h
    consensus/sample_trimming.cpp
    consensus/sample_trimming.h
    consensus/variant_calling.cpp
    consensus/variant_calling.h
    consensus/variant_calling_sample.cpp
    consensus/variant_calling_sample.h
    consensus/window.cpp
    consensus/window.h
    features/decoder_base.cpp
    features/decoder_base.h
    features/decoder_factory.cpp
    features/decoder_factory.h
    features/encoder_base.h
    features/encoder_counts.cpp
    features/encoder_counts.h
    features/encoder_factory.cpp
    features/encoder_factory.h
    features/encoder_read_alignment.cpp
    features/encoder_read_alignment.h
    features/medaka_bamiter.cpp
    features/medaka_bamiter.h
    features/medaka_counts.cpp
    features/medaka_counts.h
    features/medaka_read_matrix.cpp
    features/medaka_read_matrix.h
 )

target_compile_definitions(dorado_secondary
    PUBLIC
        DORADO_GPU_BUILD=$<BOOL:${DORADO_GPU_BUILD}>
        DORADO_CUDA_BUILD=$<AND:$<BOOL:${DORADO_GPU_BUILD}>,$<NOT:$<BOOL:${APPLE}>>>
        DORADO_METAL_BUILD=$<AND:$<BOOL:${DORADO_GPU_BUILD}>,$<BOOL:${APPLE}>>
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
    target_compile_definitions(dorado_secondary
        PUBLIC
            DORADO_TX2=$<VERSION_LESS:${CUDAToolkit_VERSION},11.0>
            DORADO_ORIN=$<VERSION_GREATER_EQUAL:${CUDAToolkit_VERSION},11.0>
    )
else()
    target_compile_definitions(dorado_secondary
        PUBLIC
            DORADO_TX2=0
            DORADO_ORIN=0
    )
endif()

target_include_directories(dorado_secondary
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# 3rdparty libs should be considered SYSTEM headers
target_include_directories(dorado_secondary
    SYSTEM
    PUBLIC
    ${DORADO_3RD_PARTY_SOURCE}/cxxpool/src
    ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
    ${DORADO_3RD_PARTY_SOURCE}/intervaltree
)

target_link_libraries(dorado_secondary
    PUBLIC
        torch_lib
        dorado_utils
        dorado_torch_utils
    PRIVATE
        htslib
        spdlog::spdlog
        toml11::toml11
)

if (NOT APPLE)
    target_link_libraries(dorado_secondary PRIVATE OpenSSL::SSL)
endif()

if(NOT WIN32)
    add_dependencies(dorado_secondary htslib_project)
endif()

enable_warnings_as_errors(dorado_secondary)
