set(PUBLIC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/include/secondary/common)

target_sources(dorado_secondary
    PUBLIC
        ${PUBLIC_BASE}/bam_file.h
        ${PUBLIC_BASE}/bam_info.h
        ${PUBLIC_BASE}/batching.h
        ${PUBLIC_BASE}/interval.h
        ${PUBLIC_BASE}/region.h
        ${PUBLIC_BASE}/stats.h
        ${PUBLIC_BASE}/variant.h
        ${PUBLIC_BASE}/vcf_writer.h
    PRIVATE
        bam_file.cpp
        bam_info.cpp
        batching.cpp
        region.cpp
        variant.cpp
        vcf_writer.cpp
)

target_include_directories(dorado_secondary
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)