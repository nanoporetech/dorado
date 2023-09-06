set(LIBNAME dorado_utils)

set(UTILS_SOURCE_FILES
    alignment_utils.cpp
    alignment_utils.h
    AsyncQueue.h
    bam_utils.cpp
    bam_utils.h
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
        list(APPEND UTILS_SOURCE_FILES
            metal_utils.cpp
            metal_utils.h
        )
    else()
        list(APPEND UTILS_SOURCE_FILES
            cuda_utils.cpp
            cuda_utils.h
        )
    endif()
endif()

add_library(${LIBNAME} ${UTILS_SOURCE_FILES})

target_compile_definitions(${LIBNAME}
    PUBLIC
        DORADO_GPU_BUILD=$<BOOL:${DORADO_GPU_BUILD}>
)

target_precompile_headers(${LIBNAME}
    PRIVATE
    <torch/torch.h>
)

# 3rdparty libs should be considered SYSTEM headers
target_include_directories(${LIBNAME}
    SYSTEM
    PUBLIC
    ${TORCH_INCLUDE_DIRS}
    ${DORADO_3RD_PARTY}/spdlog/include
    ${DORADO_3RD_PARTY}/NVTX/c/include
    ${DORADO_3RD_PARTY}/ont-minimap2/src/3rdparty/minimap2
    ${DORADO_3RD_PARTY}/metal-cpp/metal-cpp
)

target_include_directories(${LIBNAME}
    SYSTEM
    PRIVATE
    ${HTSLIB_DIR}
)

target_link_libraries(${LIBNAME}
    ${APPLE_FWK_FOUNDATION}
    ${APPLE_FWK_QUARTZ_CORE}
    ${APPLE_FWK_METAL}
    ${TORCH_LIBRARIES}
    htslib
    edlib
    minimap2
    date::date
    OpenSSL::SSL
)

if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    target_link_libraries(${LIBNAME} ${IOKIT})
endif()

if(APPLE AND DORADO_GPU_BUILD)
    add_dependencies(${LIBNAME} metal-lib)
endif()

if(NOT WIN32)
    add_dependencies(${LIBNAME} htslib_project)
endif()