if(WIN32)
    set(ZLIB_VER 1.3.1)
    set(ZLIB_HASH "17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c")

    # On windows, we need to build Zlib
    set(ZLIB_INSTALL_DIR ${CMAKE_BINARY_DIR}/zlib-${ZLIB_VER}/install)
    if(EXISTS ${ZLIB_INSTALL_DIR})
        message(STATUS "Found ZLIB=${ZLIB_VER}")
    else()
        # Need a zlib build
        download_and_extract(
            https://github.com/madler/zlib/archive/refs/tags/v${ZLIB_VER}.tar.gz
            zlib-${ZLIB_VER}
            "${ZLIB_HASH}"
        )
        set(ZLIB_BUILD_DIR ${CMAKE_BINARY_DIR}/zlib-${ZLIB_VER}/build)
        execute_process(
            COMMAND
                cmake
                    -S ${DORADO_3RD_PARTY_DOWNLOAD}/zlib-${ZLIB_VER}/zlib-${ZLIB_VER}
                    -B ${ZLIB_BUILD_DIR}
                    -A x64
                    -D CMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_DIR}
                    -D CMAKE_CONFIGURATION_TYPES=Release
                    -G ${CMAKE_GENERATOR}
            COMMAND_ERROR_IS_FATAL ANY
        )
        execute_process(
            COMMAND cmake --build ${ZLIB_BUILD_DIR} --config Release --target install
            COMMAND_ERROR_IS_FATAL ANY
        )

    endif()

    list(APPEND CMAKE_PREFIX_PATH ${ZLIB_INSTALL_DIR})

    install(FILES ${ZLIB_INSTALL_DIR}/bin/zlib.dll DESTINATION bin)

endif()

find_package(ZLIB REQUIRED)