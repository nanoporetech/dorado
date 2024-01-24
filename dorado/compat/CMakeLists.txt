add_library(dorado_compat
    compat_utils.cpp
    compat_utils.h
)

target_include_directories(dorado_compat
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

enable_warnings_as_errors(dorado_compat)
