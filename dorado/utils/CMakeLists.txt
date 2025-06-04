add_library(dorado_utils
    # Public
    include/utils/alignment_utils.h
    include/utils/arg_parse_ext.h
    include/utils/AsyncQueue.h
    include/utils/bam_utils.h
    include/utils/barcode_kits.h
    include/utils/basecaller_utils.h
    include/utils/cigar.h
    include/utils/concurrency/async_task_executor.h
    include/utils/concurrency/detail/priority_task_queue.h
    include/utils/concurrency/multi_queue_thread_pool.h
    include/utils/concurrency/synchronisation.h
    include/utils/concurrency/task_priority.h
    include/utils/crypto_utils.h
    include/utils/dev_utils.h
    include/utils/fasta_fastq_utils.h
    include/utils/fasta_reader.h
    include/utils/fastq_reader.h
    include/utils/fs_utils.h
    include/utils/gzip_reader.h
    include/utils/hts_file.h
    include/utils/jthread.h
    include/utils/locale_utils.h
    include/utils/log_utils.h
    include/utils/math_utils.h
    include/utils/memory_utils.h
    include/utils/MergeHeaders.h
    include/utils/overlap.h
    include/utils/parameters.h
    include/utils/PostCondition.h
    include/utils/SampleSheet.h
    include/utils/scoped_trace_log.h
    include/utils/sequence_utils.h
    include/utils/stats.h
    include/utils/stream_utils.h
    include/utils/sys_stats.h
    include/utils/sys_utils.h
    include/utils/thread_utils.h
    include/utils/time_utils.h
    include/utils/timer_high_res.h
    include/utils/tty_utils.h
    include/utils/types.h
    include/utils/uuid_utils.h
    include/utils/paf_utils.h
    include/utils/fai_utils.h
    include/utils/container_utils.h
    include/utils/io_utils.h
    include/utils/rle.h
    include/utils/ssize.h
    include/utils/span.h

    # Private
    alignment_utils.cpp
    bam_utils.cpp
    barcode_kits.cpp
    basecaller_utils.cpp
    cigar.cpp
    concurrency/async_task_executor.cpp
    concurrency/detail/priority_task_queue.cpp
    concurrency/multi_queue_thread_pool.cpp
    crypto_utils.cpp
    dev_utils.cpp
    fasta_fastq_utils.cpp
    fasta_reader.cpp
    fastq_reader.cpp
    fs_utils.cpp
    gzip_reader.cpp
    hts_file.cpp
    locale_utils.cpp
    log_utils.cpp
    memory_utils.cpp
    MergeHeaders.cpp
    parameters.cpp
    SampleSheet.cpp
    scoped_trace_log.cpp
    sequence_utils.cpp
    stats.cpp
    sys_stats.cpp
    sys_utils.cpp
    thread_utils.cpp
    time_utils.cpp
    types.cpp
    uuid_utils.cpp
    paf_utils.cpp
    fai_utils.cpp
    container_utils.cpp
    io_utils.cpp
 )

target_compile_definitions(dorado_utils
    PUBLIC
        ENABLE_PER_READ_TRACE=$<BOOL:${DORADO_PER_READ_TRACE}>
        DORADO_CUDA_BUILD=$<NOT:$<BOOL:${APPLE}>>
        DORADO_METAL_BUILD=$<BOOL:${APPLE}>
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
    target_compile_definitions(dorado_utils
        PUBLIC
            DORADO_ORIN=1
    )
else()
    target_compile_definitions(dorado_utils
        PUBLIC
            DORADO_ORIN=0
    )
endif()

target_include_directories(dorado_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/.. # TODO: remove this once dependents are fixed
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

enable_warnings_as_errors(dorado_utils)
