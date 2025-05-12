add_library(dorado_file_info STATIC
    file_info.cpp
    file_info.h
)

target_link_libraries(dorado_file_info
PUBLIC
    dorado_models_lib
    dorado_utils
PRIVATE
    pod5_libs
    ${ZLIB_LIBRARIES}
)

enable_warnings_as_errors(dorado_file_info)
