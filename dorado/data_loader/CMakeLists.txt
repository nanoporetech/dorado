add_library(dorado_io_lib
    DataLoader.cpp
    DataLoader.h
)

target_link_libraries(dorado_io_lib
PUBLIC
    dorado_lib
PRIVATE
    ${POD5_LIBRARIES}
    HDF5::HDF5
    vbz_hdf_plugin
    ${CMAKE_DL_LIBS}
    ${ZLIB_LIBRARIES}
)

set_target_properties(dorado_io_lib
PROPERTIES
LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

enable_warnings_as_errors(dorado_io_lib)
