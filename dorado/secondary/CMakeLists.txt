add_subdirectory(architectures)
add_subdirectory(consensus)
add_subdirectory(features)

add_library(dorado_secondary
    bam_file.cpp
    bam_file.h
    bam_info.cpp
    bam_info.h
    batching.cpp
    batching.h
    interval.h
    region.cpp
    region.h
    stats.h
    variant.cpp
    variant.h
    vcf_writer.cpp
    vcf_writer.h
)

target_include_directories(dorado_secondary
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# 3rdparty libs should be considered SYSTEM headers
target_include_directories(dorado_secondary
    SYSTEM
    PUBLIC
        ${DORADO_3RD_PARTY_SOURCE}/intervaltree
)

target_link_libraries(dorado_secondary
    PUBLIC
        dorado_torch_utils
        dorado_utils
    PRIVATE
        cxxpool
        dorado_secondary_architectures
        dorado_secondary_consensus
        dorado_secondary_features
        dorado_version
        htslib
        kadayashi::haplotag_lib
        spdlog::spdlog
)

enable_warnings_as_errors(dorado_secondary)
