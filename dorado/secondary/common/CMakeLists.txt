set(PUBLIC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/include/secondary/common)

target_sources(dorado_secondary
    PUBLIC
        ${PUBLIC_BASE}/alignment.h
        ${PUBLIC_BASE}/bam_file.h
        ${PUBLIC_BASE}/bam_info.h
        ${PUBLIC_BASE}/batched_data.h
        ${PUBLIC_BASE}/batching.h
        ${PUBLIC_BASE}/device_info.h
        ${PUBLIC_BASE}/interval.h
        ${PUBLIC_BASE}/interval_tree_types.h
        ${PUBLIC_BASE}/region.h
        ${PUBLIC_BASE}/resolve_model.h
        ${PUBLIC_BASE}/stats.h
        ${PUBLIC_BASE}/variant.h
        ${PUBLIC_BASE}/vcf_writer.h
        ${PUBLIC_BASE}/window.h
        ${PUBLIC_BASE}/worker_return_status.h
    PRIVATE
        alignment.cpp
        bam_file.cpp
        bam_info.cpp
        batching.cpp
        region.cpp
        resolve_model.cpp
        variant.cpp
        vcf_writer.cpp
        window.cpp
)

target_include_directories(dorado_secondary
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
