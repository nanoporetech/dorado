set(PUBLIC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/include/secondary/features)

target_sources(dorado_secondary
    PUBLIC
        ${PUBLIC_BASE}/decoder_base.h
        ${PUBLIC_BASE}/decoder_factory.h
        ${PUBLIC_BASE}/encoder_base.h
        ${PUBLIC_BASE}/encoder_factory.h
        ${PUBLIC_BASE}/haplotag_source.h
        ${PUBLIC_BASE}/kadayashi_options.h
        ${PUBLIC_BASE}/kadayashi_utils.h
    PRIVATE
        decoder_base.cpp
        decoder_factory.cpp
        encoder_counts.cpp
        encoder_counts.h
        encoder_factory.cpp
        encoder_read_alignment.cpp
        encoder_read_alignment.h
        encoder_utils.cpp
        encoder_utils.h
        kadayashi_utils.cpp
        medaka_bamiter.cpp
        medaka_bamiter.h
        medaka_counts.cpp
        medaka_counts.h
        medaka_read_matrix.cpp
        medaka_read_matrix.h
)

target_include_directories(dorado_secondary
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)