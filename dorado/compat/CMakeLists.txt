add_library(dorado_compat
    # Public
    include/compat/compat_utils.h

    # Private
    compat_utils.cpp
)

target_include_directories(dorado_compat
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_compat)
