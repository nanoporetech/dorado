if (DORADO_ENABLE_PCH)
    add_subdirectory(pch)
endif()

dorado_add_library(
    NAME
        dorado_torch_utils
    PUBLIC_DIR
        torch_utils
    SOURCES_PUBLIC
        auto_detect_device.h
        cuda_utils.h
        duplex_utils.h
        gpu_monitor.h
        gpu_profiling.h
        metal_utils.h
        module_utils.h
        tensor_utils.h
        torch_utils.h
        trim.h
    SOURCES_PRIVATE
        duplex_utils.cpp
        gpu_monitor.cpp
        tensor_utils.cpp
        torch_utils.cpp
        trim.cpp
    DEPENDS_PUBLIC
        date::date
        edlib
        spdlog::spdlog
        torch_lib
    DEPENDS_PRIVATE
        dorado_compat
        dorado_utils
        htslib
        minimap2
        toml11::toml11
)

if(APPLE)
    target_sources(dorado_torch_utils PRIVATE metal_utils.cpp)
else()
    target_sources(dorado_torch_utils PRIVATE cuda_utils.cpp)
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
            metal_cpp
    )
endif()

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(duplex_utils.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()

