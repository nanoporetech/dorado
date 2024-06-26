add_library(dorado_utils
    alignment_utils.cpp
    alignment_utils.h
    arg_parse_ext.h
    AsyncQueue.h
    bam_utils.cpp
    bam_utils.h
    barcode_kits.cpp
    barcode_kits.h
    basecaller_utils.cpp
    basecaller_utils.h
    concurrency/async_task_executor.cpp
    concurrency/async_task_executor.h
    concurrency/synchronisation.h
    crypto_utils.h
    crypto_utils.cpp
    dev_utils.cpp
    dev_utils.h
    duplex_utils.cpp
    duplex_utils.h
    fs_utils.cpp
    fs_utils.h
    gpu_monitor.cpp
    gpu_monitor.h
    hts_file.cpp
    hts_file.h
    locale_utils.cpp
    locale_utils.h
    log_utils.cpp
    log_utils.h
    math_utils.h
    memory_utils.cpp
    memory_utils.h
    MergeHeaders.cpp
    MergeHeaders.h
    module_utils.h
    parameters.cpp
    parameters.h
    parse_custom_kit.cpp
    parse_custom_kit.h
    PostCondition.h
    SampleSheet.cpp
    SampleSheet.h
    scoped_trace_log.cpp
    scoped_trace_log.h
    sequence_utils.cpp
    sequence_utils.h
    stats.cpp
    stats.h
    stream_utils.h
    sys_stats.cpp
    sys_stats.h
    tensor_utils.cpp
    tensor_utils.h
    thread_naming.cpp
    thread_naming.h
    time_utils.cpp
    time_utils.h
    torch_utils.cpp
    torch_utils.h
    trim_rapid_adapter.cpp
    trim_rapid_adapter.h
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
        DORADO_CUDA_BUILD=$<AND:$<BOOL:${DORADO_GPU_BUILD}>,$<NOT:$<BOOL:${APPLE}>>>
        DORADO_METAL_BUILD=$<AND:$<BOOL:${DORADO_GPU_BUILD}>,$<BOOL:${APPLE}>>        
)

target_include_directories(dorado_utils
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# 3rdparty libs should be considered SYSTEM headers
target_include_directories(dorado_utils
    SYSTEM
    PUBLIC
    ${TORCH_INCLUDE_DIRS}
    ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
    ${DORADO_3RD_PARTY_DOWNLOAD}/metal-cpp/metal-cpp
    ${DORADO_3RD_PARTY_SOURCE}/toml11
)

target_link_libraries(dorado_utils
    PUBLIC
        ${TORCH_LIBRARIES}
        date::date
        edlib
        spdlog::spdlog
    PRIVATE
        dorado_compat
        minimap2
        htslib
)

if (DORADO_ENABLE_PCH)
    add_subdirectory(pch)
    target_link_libraries(dorado_utils PUBLIC dorado_pch)
    target_precompile_headers(dorado_utils REUSE_FROM dorado_pch)
endif()

if(APPLE)
    find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
    target_link_libraries(dorado_utils
        PUBLIC
            ${APPLE_FWK_FOUNDATION}
    )
    if (DORADO_GPU_BUILD)
        find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
        find_library(APPLE_FWK_METAL Metal REQUIRED)
        add_dependencies(dorado_utils metal-lib)
        target_link_libraries(dorado_utils
            PUBLIC
                ${APPLE_FWK_QUARTZ_CORE}
                ${APPLE_FWK_METAL}
        )
    endif()
    if (NOT IOS)
        find_library(IOKIT IOKit REQUIRED)
        target_link_libraries(dorado_utils PUBLIC ${IOKIT})
    endif()
endif()

if (NOT APPLE)
    target_link_libraries(dorado_utils PRIVATE OpenSSL::SSL)
endif()

if(NOT WIN32)
    add_dependencies(dorado_utils htslib_project)
endif()

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(duplex_utils.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()

enable_warnings_as_errors(dorado_utils)