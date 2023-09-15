add_library(dorado_utils
    alignment_utils.cpp
    alignment_utils.h
    AsyncQueue.h
    bam_utils.cpp
    bam_utils.h
    barcode_kits.cpp
    barcode_kits.h
    basecaller_utils.cpp
    basecaller_utils.h
    compat_utils.cpp
    compat_utils.h
    dev_utils.cpp
    dev_utils.h
    duplex_utils.cpp
    duplex_utils.h
    log_utils.cpp
    log_utils.h
    math_utils.h
    memory_utils.cpp
    memory_utils.h
    module_utils.h
    parameters.h
    sequence_utils.cpp
    sequence_utils.h
    stats.h
    tensor_utils.cpp
    tensor_utils.h
    time_utils.h
    trim.cpp
    trim.h
    tty_utils.h
    types.cpp
    types.h
    uuid_utils.cpp
    uuid_utils.h
)

if (DORADO_GPU_BUILD)
    if(APPLE)
        target_sources(dorado_utils
            PRIVATE
            metal_utils.cpp
            metal_utils.h
        )
    else()
        target_sources(dorado_utils
            PRIVATE
            cuda_utils.cpp
            cuda_utils.h
        )
    endif()
endif()

target_compile_definitions(dorado_utils
    PUBLIC
        DORADO_GPU_BUILD=$<BOOL:${DORADO_GPU_BUILD}>
)

target_precompile_headers(dorado_utils
    PRIVATE
    <torch/torch.h>
)

# 3rdparty libs should be considered SYSTEM headers
target_include_directories(dorado_utils
    SYSTEM
    PUBLIC
    ${TORCH_INCLUDE_DIRS}
    ${DORADO_3RD_PARTY}/spdlog/include
    ${DORADO_3RD_PARTY}/NVTX/c/include
    ${DORADO_3RD_PARTY}/ont-minimap2/src/3rdparty/minimap2
    ${DORADO_3RD_PARTY}/metal-cpp/metal-cpp
)

target_include_directories(dorado_utils
    SYSTEM
    PRIVATE
    ${HTSLIB_DIR}
)

target_link_libraries(dorado_utils
    ${TORCH_LIBRARIES}
    htslib
    edlib
    minimap2
    date::date
    OpenSSL::SSL
)

if(APPLE AND DORADO_GPU_BUILD)
    add_dependencies(dorado_utils metal-lib)
    target_link_libraries(
        dorado_utils
        ${APPLE_FWK_FOUNDATION}
        ${APPLE_FWK_QUARTZ_CORE}
        ${APPLE_FWK_METAL}
    )   
endif()

if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    target_link_libraries(dorado_utils ${IOKIT})
endif()

if(NOT WIN32)
    add_dependencies(dorado_utils htslib_project)
endif()