add_library(dorado_file_info STATIC
    file_info.cpp
    file_info.h
)

target_include_directories(dorado_file_info
SYSTEM
PRIVATE
    ${POD5_INCLUDE}
    ${DORADO_3RD_PARTY_SOURCE}/HighFive/include
)

target_link_libraries(dorado_file_info
PUBLIC    
    dorado_models_lib
    dorado_utils
PRIVATE
    ${POD5_LIBRARIES}
    HDF5::HDF5
    vbz_hdf_plugin
    ${CMAKE_DL_LIBS}
    ${ZLIB_LIBRARIES}
)

set_target_properties(dorado_file_info
PROPERTIES
LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

enable_warnings_as_errors(dorado_file_info)