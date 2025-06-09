add_library(dorado_summary STATIC
    # Public
    include/summary/summary.h

    # Private
    summary.cpp
)

target_link_libraries(dorado_summary
    PRIVATE
        dorado_lib
        dorado_utils
        spdlog::spdlog
)

target_include_directories(dorado_summary
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_summary)
