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
)

if (NOT APPLE)
    target_link_libraries(dorado_secondary PRIVATE OpenSSL::SSL)
endif()

if(NOT WIN32)
    add_dependencies(dorado_secondary htslib_project)
endif()

enable_warnings_as_errors(dorado_secondary)
