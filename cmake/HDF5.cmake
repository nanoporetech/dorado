option(DYNAMIC_HDF "Link HDF as dynamic libs" OFF)

if((CMAKE_SYSTEM_NAME STREQUAL "Linux") AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"))
    # download the pacakge for arm, we want to package this due to hdf5's dependencies
    set(DYNAMIC_HDF ON)
    set(HDF_VER hdf5-1.10.0-aarch64)
    download_and_extract(
        ${DORADO_CDN_URL}/${HDF_VER}.zip
        ${HDF_VER}
        "872f75e05d98d6d7b05c91a1e6ce0d82cbc890177a642ea4f43abd572164959d"
    )
    list(PREPEND CMAKE_PREFIX_PATH ${DORADO_3RD_PARTY_DOWNLOAD}/${HDF_VER}/${HDF_VER})

elseif(WIN32)
    set(HDF_VER hdf5-1.12.1-3)
    set(ZLIB_VER 1.3.1)

    # On windows, we need to build HDF5
    set(HDF5_ZLIB_INSTALL_DIR ${CMAKE_BINARY_DIR}/zlib-${ZLIB_VER}/install)
    if(EXISTS ${DORADO_3RD_PARTY_DOWNLOAD}/${HDF_VER} AND EXISTS ${HDF5_ZLIB_INSTALL_DIR})
        message(STATUS "Found HDF=${HDF_VER} and ZLIB=${ZLIB_VER}")
    else()
        # Need a zlib build for HDF to use
        download_and_extract(
            https://github.com/madler/zlib/archive/refs/tags/v${ZLIB_VER}.tar.gz
            zlib-${ZLIB_VER}
            "17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c"
        )
        set(HDF5_ZLIB_BUILD_DIR ${CMAKE_BINARY_DIR}/zlib-${ZLIB_VER}/build)
        execute_process(
            COMMAND
                cmake
                    -S ${DORADO_3RD_PARTY_DOWNLOAD}/zlib-${ZLIB_VER}/zlib-${ZLIB_VER}
                    -B ${HDF5_ZLIB_BUILD_DIR}
                    -A x64
                    -D CMAKE_INSTALL_PREFIX=${HDF5_ZLIB_INSTALL_DIR}
                    -D CMAKE_CONFIGURATION_TYPES=Release
                    -G ${CMAKE_GENERATOR}
            COMMAND_ERROR_IS_FATAL ANY
        )
        execute_process(
            COMMAND cmake --build ${HDF5_ZLIB_BUILD_DIR} --config Release --target install
            COMMAND_ERROR_IS_FATAL ANY
        )

        # HDF5 itself
        download_and_extract(
            ${DORADO_CDN_URL}/${HDF_VER}-win.zip
            ${HDF_VER}
            "6bf77c3154fff7f0e9b4ec9c42b8783b0a786cc2b658c0c15968148fefce6268"
        )
    endif()

    list(APPEND CMAKE_PREFIX_PATH ${HDF5_ZLIB_INSTALL_DIR})
    list(APPEND CMAKE_PREFIX_PATH ${DORADO_3RD_PARTY_DOWNLOAD}/${HDF_VER}/${HDF_VER})

    install(FILES ${HDF5_ZLIB_INSTALL_DIR}/bin/zlib.dll DESTINATION bin)

elseif (APPLE)
    set(HDF_VER 1.14.3)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(HDF_ARCH "x86_64")
        set(HDF_HASH "0b95a50b824eba64aeede416e68243bbc708653ffdf17d7635b074058e38797a")
    else()
        set(HDF_ARCH "armv8")
        set(HDF_HASH "1f4dab9ac68129968cc5777fd62836686c2496af02f13cdbec6b82e030092466")
    endif()
    download_and_extract(
        ${DORADO_CDN_URL}/hdf5-${HDF_VER}-${HDF_ARCH}.zip
        hdf5-${HDF_VER}-${HDF_ARCH}
        ${HDF_HASH}
    )
    list(PREPEND CMAKE_PREFIX_PATH ${DORADO_3RD_PARTY_DOWNLOAD}/hdf5-${HDF_VER}-${HDF_ARCH}/${HDF_VER}_${HDF_ARCH})

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

find_package(ZLIB REQUIRED)
find_package(HDF5 COMPONENTS C REQUIRED)
