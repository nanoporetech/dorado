add_library(dorado_polish STATIC
    # Public
    include/polish/polish_impl.h
    include/polish/polish_progress_tracker.h

    # Private
    polish_impl.cpp
    polish_progress_tracker.cpp
)

target_link_libraries(dorado_polish
    PUBLIC
        dorado_secondary
        dorado_utils
        indicators
    PRIVATE
        cxxpool
        dorado_hts_io
        dorado_torch_utils
        htslib
        spdlog::spdlog
)

target_include_directories(dorado_polish
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_polish)
