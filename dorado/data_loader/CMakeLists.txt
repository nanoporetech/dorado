add_library(dorado_io_lib
    DataLoader.cpp
    DataLoader.h
)

target_link_libraries(dorado_io_lib
PUBLIC
    dorado_lib
PRIVATE
    spdlog::spdlog
    pod5_libs
    HDF5Wrapper
    vbz_hdf_plugin
    ${ZLIB_LIBRARIES}
    HighFive::Include
)

enable_warnings_as_errors(dorado_io_lib)
