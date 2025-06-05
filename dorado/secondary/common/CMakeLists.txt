add_library(dorado_secondary_common
    # Public
    include/secondary/common/bam_file.h
    include/secondary/common/bam_info.h
    include/secondary/common/batching.h
    include/secondary/common/interval.h
    include/secondary/common/region.h
    include/secondary/common/stats.h
    include/secondary/common/variant.h
    include/secondary/common/vcf_writer.h

    # Private
    bam_file.cpp
    bam_info.cpp
    batching.cpp
    region.cpp
    variant.cpp
    vcf_writer.cpp
)

target_include_directories(dorado_secondary_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dorado_secondary_common
    PUBLIC
        dorado_torch_utils
        dorado_utils
    PRIVATE
        cxxpool
        dorado_version
        htslib
        intervaltree
        kadayashi::haplotag_lib
        spdlog::spdlog
)

enable_warnings_as_errors(dorado_secondary_common)
