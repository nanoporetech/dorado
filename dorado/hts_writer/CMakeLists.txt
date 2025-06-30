dorado_add_library(
    NAME
        dorado_hts_writer
    PUBLIC_DIR
        hts_writer
    SOURCES_PUBLIC
        hts_file_writer.h
        interface.h
    SOURCES_PRIVATE
        hts_file_writer.cpp
    DEPENDS_PUBLIC
        # nothing
    DEPENDS_PRIVATE
        dorado_hts_utils
        dorado_utils
        spdlog::spdlog
    NO_TORCH
)
