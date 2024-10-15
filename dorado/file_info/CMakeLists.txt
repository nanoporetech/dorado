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
    ${ZLIB_LIBRARIES}
)

enable_warnings_as_errors(dorado_file_info)
