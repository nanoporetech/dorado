dorado_add_library(
    NAME
        dorado_hts_io
    PUBLIC_DIR
        hts_io
    SOURCES_PUBLIC
        FastxRandomReader.h
    SOURCES_PRIVATE
        FastxRandomReader.cpp
    DEPENDS_PUBLIC
        # nothing
    DEPENDS_PRIVATE
        dorado_utils
        htslib
        spdlog::spdlog
    NO_TORCH
)
