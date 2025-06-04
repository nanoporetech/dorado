add_library(dorado_file_info STATIC
    # Public
    include/file_info/file_info.h

    # Private
    file_info.cpp
)

target_link_libraries(dorado_file_info
    PUBLIC
        dorado_models_lib
        dorado_utils
    PRIVATE
        pod5_libs
)

target_include_directories(dorado_file_info
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_file_info)
