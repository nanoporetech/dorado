add_library(dorado_variant STATIC
    # Public
    include/variant/variant_progress_tracker.h

    # Private
    variant_progress_tracker.cpp
)

target_link_libraries(dorado_variant
    PUBLIC
        dorado_utils
        indicators
    PRIVATE
        spdlog::spdlog
)

target_include_directories(dorado_variant
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_variant)
