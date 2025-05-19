add_library(dorado_io_lib
    DataLoader.cpp
    DataLoader.h
)

target_link_libraries(dorado_io_lib
PUBLIC
    dorado_lib
    cxxpool
PRIVATE
    spdlog::spdlog
    pod5_libs
    ${ZLIB_LIBRARIES}
)

enable_warnings_as_errors(dorado_io_lib)
