dorado_add_library(
    NAME
        dorado_summary
    PUBLIC_DIR
        summary
    SOURCES_PUBLIC
        summary.h
    SOURCES_PRIVATE
        summary.cpp
    DEPENDS_PUBLIC
        # nothing
    DEPENDS_PRIVATE
        dorado_lib
        dorado_hts_utils
        dorado_utils
        spdlog::spdlog
)
