add_library(dorado_hts_io STATIC
    # Public
    include/hts_io/FastxRandomReader.h

    # Private
    FastxRandomReader.cpp
)

target_link_libraries(dorado_hts_io
    PRIVATE
        dorado_utils
        htslib
        spdlog::spdlog
)

target_include_directories(dorado_hts_io
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_hts_io)
