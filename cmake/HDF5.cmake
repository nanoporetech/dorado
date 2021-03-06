set(HDF_VER hdf5-1.12.1-3)
set(ZLIB_VER zlib-1.2.12)

option(DYNAMIC_HDF "Link HDF as dynamic libs" OFF)

# On windows, we need to build HDF5

if(WIN32)

    if(EXISTS ${DORADO_3RD_PARTY}/${HDF_VER})
        message(STATUS "Found ${HDF_VER}")
    else()
        # Need a zlib build for HDF to use
        download_and_extract(https://zlib.net/${ZLIB_VER}.tar.gz ${ZLIB_VER})
        execute_process(COMMAND cmake -S ${DORADO_3RD_PARTY}/${ZLIB_VER}/${ZLIB_VER} -B ${DORADO_3RD_PARTY}/${ZLIB_VER}/${ZLIB_VER}/cmake-build -A x64
            -DCMAKE_INSTALL_PREFIX=${DORADO_3RD_PARTY}/${ZLIB_VER}/install)
        execute_process(COMMAND cmake --build ${DORADO_3RD_PARTY}/${ZLIB_VER}/${ZLIB_VER}/cmake-build --config Release --target install)
        list(APPEND CMAKE_PREFIX_PATH ${DORADO_3RD_PARTY}/${ZLIB_VER}/install)

        # HDF5 itself
        download_and_extract(https://nanoporetech.box.com/shared/static/h5u267duw3sa4l814yirmxamx3hgouwp.zip ${HDF_VER})
    endif()

    install(FILES ${DORADO_3RD_PARTY}/${ZLIB_VER}/install/bin/zlib.dll DESTINATION bin)
    list(APPEND CMAKE_PREFIX_PATH ${DORADO_3RD_PARTY}/${HDF_VER}/${HDF_VER})

endif()


if(DYNAMIC_HDF)
    add_link_options(-ldl)
else()
    set(HDF5_USE_STATIC_LIBRARIES On)
endif()

find_package(HDF5 COMPONENTS C CXX HL)
