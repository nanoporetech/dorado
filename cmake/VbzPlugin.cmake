function(add_hdf_vbz_plugin)
    set(ENABLE_CONAN OFF)
    set(ENABLE_PERF_TESTING OFF)
    set(ENABLE_PYTHON OFF)
    set(ENABLE_PACKAGING OFF)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)

    if(WIN32)
        # On windows we need to build a static lib for zstd as there's no prebuilt distro
        set(ZSTD_BUILD_DIR ${CMAKE_BINARY_DIR}/cmake-build-zstd)
        execute_process(COMMAND
            cmake
                -S ${DORADO_3RD_PARTY_SOURCE}/zstd/build/cmake
                -B ${ZSTD_BUILD_DIR}
                -A x64
                -D CMAKE_CONFIGURATION_TYPES=Release
            COMMAND_ERROR_IS_FATAL ANY
        )
        execute_process(
            COMMAND cmake --build ${ZSTD_BUILD_DIR} --config Release
            COMMAND_ERROR_IS_FATAL ANY
        )

        # On windows we need to tell hdf_plugins where we put the built zstd lib
        set(CONAN_INCLUDE_DIRS_RELEASE ${DORADO_3RD_PARTY_SOURCE}/zstd/lib)
        set(CONAN_INCLUDE_DIRS_DEBUG ${DORADO_3RD_PARTY_SOURCE}/zstd/lib)
        set(CONAN_LIB_DIRS_RELEASE ${ZSTD_BUILD_DIR}/lib/Release)
        set(CONAN_LIB_DIRS_DEBUG ${ZSTD_BUILD_DIR}/lib/Debug)

        install(FILES ${ZSTD_BUILD_DIR}/lib/Release/zstd.dll DESTINATION bin)

    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        # The GCC8 CI build needs a newer version of zstd than provided by the docker image.
        set(ZSTD_BUILD_DIR ${CMAKE_BINARY_DIR}/cmake-build-zstd)
        set(ZSTD_INSTALL_DIR ${CMAKE_BINARY_DIR}/cmake-install-zstd)
        execute_process(COMMAND
            cmake
                -S ${DORADO_3RD_PARTY_SOURCE}/zstd/build/cmake
                -B ${ZSTD_BUILD_DIR}
                -D CMAKE_INSTALL_PREFIX=${ZSTD_INSTALL_DIR}
            COMMAND_ERROR_IS_FATAL ANY
        )
        execute_process(
            COMMAND cmake --build ${ZSTD_BUILD_DIR} --target install
            COMMAND_ERROR_IS_FATAL ANY
        )

        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dorado/3rdparty/hdf_plugins/cmake")
        # Findzstd.cmake uses conan variables to determine where things are.
        set(CONAN_INCLUDE_DIRS_RELEASE ${ZSTD_INSTALL_DIR}/include)
        set(CONAN_LIB_DIRS_RELEASE ${ZSTD_INSTALL_DIR}/lib)
        find_package(zstd 1.3.6 REQUIRED)

    else()
        # Some platforms need the Findzstd.cmake from hdf_plugins, but append it in case we already
        # have one in the path.
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dorado/3rdparty/hdf_plugins/cmake")
        # hdf_plugins looks for 1.3.1 but the libarrow in POD5 requires 1.3.6 minimum.
        find_package(zstd 1.3.6 REQUIRED)

    endif()

    add_subdirectory(dorado/3rdparty/hdf_plugins EXCLUDE_FROM_ALL)
endfunction()
