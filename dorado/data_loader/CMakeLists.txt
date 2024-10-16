add_library(dorado_io_lib
    DataLoader.cpp
    DataLoader.h
)

target_include_directories(dorado_io_lib
    SYSTEM
    PRIVATE
        ${DORADO_3RD_PARTY_SOURCE}/HighFive/include
)

target_link_libraries(dorado_io_lib
PUBLIC
    dorado_lib
PRIVATE
    spdlog::spdlog
    pod5_libs
    HDF5::HDF5
    vbz_hdf_plugin
    ${ZLIB_LIBRARIES}
)

enable_warnings_as_errors(dorado_io_lib)
