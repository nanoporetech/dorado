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
    cigar.cpp
    cigar.h
    concurrency/async_task_executor.cpp
    concurrency/async_task_executor.h
    concurrency/detail/priority_task_queue.cpp
    concurrency/detail/priority_task_queue.h
    concurrency/multi_queue_thread_pool.cpp
    concurrency/multi_queue_thread_pool.h
    concurrency/synchronisation.h
    concurrency/task_priority.h
    crypto_utils.h
    crypto_utils.cpp
    dev_utils.cpp
    dev_utils.h
    fasta_fastq_utils.cpp
    fasta_fastq_utils.h
    fasta_reader.cpp
    fasta_reader.h
    fastq_reader.cpp
    fastq_reader.h
    fs_utils.cpp
    fs_utils.h
    gzip_reader.cpp
    gzip_reader.h
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
    overlap.h
    parameters.cpp
    parameters.h
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
    thread_naming.cpp
    thread_naming.h
    time_utils.cpp
    time_utils.h
    timer_high_res.h
    tty_utils.h
    types.cpp
    types.h
    uuid_utils.cpp
    uuid_utils.h
    paf_utils.cpp
    paf_utils.h
    fai_utils.cpp
    fai_utils.h
    container_utils.cpp
    container_utils.h
    io_utils.cpp
    io_utils.h
    rle.h
    ssize.h
    span.h
 )

target_compile_definitions(dorado_utils
    PUBLIC
        DORADO_GPU_BUILD=$<BOOL:${DORADO_GPU_BUILD}>
        DORADO_CUDA_BUILD=$<AND:$<BOOL:${DORADO_GPU_BUILD}>,$<NOT:$<BOOL:${APPLE}>>>
        DORADO_METAL_BUILD=$<AND:$<BOOL:${DORADO_GPU_BUILD}>,$<BOOL:${APPLE}>>
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
    target_compile_definitions(dorado_utils
        PUBLIC
            DORADO_TX2=$<VERSION_LESS:${CUDAToolkit_VERSION},11.0>
            DORADO_ORIN=$<VERSION_GREATER_EQUAL:${CUDAToolkit_VERSION},11.0>
    )
else()
    target_compile_definitions(dorado_utils
        PUBLIC
            DORADO_TX2=0
            DORADO_ORIN=0
    )
endif()

target_include_directories(dorado_utils
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# 3rdparty libs should be considered SYSTEM headers
target_include_directories(dorado_utils
    SYSTEM
    PUBLIC
    ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
)

target_link_libraries(dorado_utils
    PUBLIC
        date::date
        edlib
        spdlog::spdlog
        argparse::argparse
    PRIVATE
        dorado_compat
        minimap2
        htslib
        toml11::toml11
)

if (NOT APPLE)
    target_link_libraries(dorado_utils PRIVATE OpenSSL::SSL)
endif()

if(NOT WIN32)
    add_dependencies(dorado_utils htslib_project)
endif()

enable_warnings_as_errors(dorado_utils)
