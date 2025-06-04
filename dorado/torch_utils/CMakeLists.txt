add_library(dorado_torch_utils
    # Public
    include/torch_utils/auto_detect_device.h
    include/torch_utils/cuda_utils.h
    include/torch_utils/duplex_utils.h
    include/torch_utils/gpu_monitor.h
    include/torch_utils/gpu_profiling.h
    include/torch_utils/metal_utils.h
    include/torch_utils/module_utils.h
    include/torch_utils/tensor_utils.h
    include/torch_utils/torch_utils.h
    include/torch_utils/trim.h

    # Private
    duplex_utils.cpp
    gpu_monitor.cpp
    tensor_utils.cpp
    torch_utils.cpp
    trim.cpp
)

if(APPLE)
    target_sources(dorado_torch_utils PRIVATE metal_utils.cpp)
else()
    target_sources(dorado_torch_utils PRIVATE cuda_utils.cpp)
endif()

target_include_directories(dorado_torch_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 3rdparty libs should be considered SYSTEM headers
target_include_directories(dorado_torch_utils
    SYSTEM
    PUBLIC
        ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
        ${DORADO_3RD_PARTY_DOWNLOAD}/metal-cpp/metal-cpp
)

target_link_libraries(dorado_torch_utils
    PUBLIC
        torch_lib
        date::date
        edlib
        spdlog::spdlog
    PRIVATE
        dorado_compat
        dorado_utils
        minimap2
        htslib
        toml11::toml11
)

if (DORADO_ENABLE_PCH)
    add_subdirectory(pch)
    target_link_libraries(dorado_torch_utils PUBLIC dorado_pch)
    target_precompile_headers(dorado_torch_utils REUSE_FROM dorado_pch)
endif()

if(APPLE)
    find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
    find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
    find_library(APPLE_FWK_METAL Metal REQUIRED)
    find_library(IOKIT IOKit REQUIRED)
    add_dependencies(dorado_torch_utils metal-lib)
    target_link_libraries(dorado_torch_utils
        PUBLIC
            ${APPLE_FWK_FOUNDATION}
            ${APPLE_FWK_QUARTZ_CORE}
            ${APPLE_FWK_METAL}
            ${IOKIT}
    )
endif()

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(duplex_utils.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()

enable_warnings_as_errors(dorado_torch_utils)
