add_library(dorado_secondary_features
    # Public
    include/secondary/features/decoder_base.h
    include/secondary/features/decoder_factory.h
    include/secondary/features/encoder_base.h
    include/secondary/features/encoder_factory.h
    include/secondary/features/haplotag_source.h
    include/secondary/features/kadayashi_utils.h

    # Private
    decoder_base.cpp
    decoder_factory.cpp
    encoder_counts.cpp
    encoder_counts.h
    encoder_factory.cpp
    encoder_read_alignment.cpp
    encoder_read_alignment.h
    kadayashi_utils.cpp
    medaka_bamiter.cpp
    medaka_bamiter.h
    medaka_counts.cpp
    medaka_counts.h
    medaka_read_matrix.cpp
    medaka_read_matrix.h
)

target_include_directories(dorado_secondary_features
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dorado_secondary_features
    PUBLIC
        dorado_secondary_architectures
        dorado_secondary_common
        dorado_secondary_consensus
        dorado_torch_utils
        dorado_utils
    PRIVATE
        htslib
        kadayashi::haplotag_lib
        spdlog::spdlog
)

enable_warnings_as_errors(dorado_secondary_features)
