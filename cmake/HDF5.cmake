set(HDF_VER hdf5-1.12.1-3)
set(ZLIB_VER 1.2.12)

option(DYNAMIC_HDF "Link HDF as dynamic libs" OFF)

# On windows, we need to build HDF5

if(WIN32)

    if(EXISTS ${DORADO_3RD_PARTY}/${HDF_VER})
        message(STATUS "Found ${HDF_VER}")
    else()
        # Need a zlib build for HDF to use
        download_and_extract(https://github.com/madler/zlib/archive/refs/tags/v${ZLIB_VER}.tar.gz zlib-${ZLIB_VER})
        execute_process(COMMAND cmake -S ${DORADO_3RD_PARTY}/zlib-${ZLIB_VER}/zlib-${ZLIB_VER} -B ${DORADO_3RD_PARTY}/zlib-${ZLIB_VER}/zlib-${ZLIB_VER}/cmake-build -A x64
            -DCMAKE_INSTALL_PREFIX=${DORADO_3RD_PARTY}/zlib-${ZLIB_VER}/install)
        execute_process(COMMAND cmake --build ${DORADO_3RD_PARTY}/zlib-${ZLIB_VER}/zlib-${ZLIB_VER}/cmake-build --config Release --target install)
        list(APPEND CMAKE_PREFIX_PATH ${DORADO_3RD_PARTY}/zlib-${ZLIB_VER}/install)

        # HDF5 itself
        download_and_extract(https://nanoporetech.box.com/shared/static/h5u267duw3sa4l814yirmxamx3hgouwp.zip ${HDF_VER})
    endif()

    install(FILES ${DORADO_3RD_PARTY}/zlib-${ZLIB_VER}/install/bin/zlib.dll DESTINATION bin)
    list(APPEND CMAKE_PREFIX_PATH ${DORADO_3RD_PARTY}/${HDF_VER}/${HDF_VER})

endif()


if(DYNAMIC_HDF)
    add_link_options(-ldl)
else()
    # We cannot rely on static zlib being available on OS X or Windows.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(ZLIB_USE_STATIC_LIBS ON)
    endif()
    set(HDF5_USE_STATIC_LIBRARIES ON)
endif()

find_package(ZLIB)
find_package(HDF5 COMPONENTS C CXX HL)
