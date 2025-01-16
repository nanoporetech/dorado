function(add_hdf_vbz_plugin)
    # cmake policy CMP0077 allows options to be overridden (which is
    # exactly what we're trying to do here) -- without it set to NEW
    # some of these option overrides won't take effect.
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    # hdf_plugins is a pretty general-purpose library and we don't
    # want to compile it with several of its features. We can hide
    # them in a function to avoid polluting state
    set(ENABLE_CONAN OFF)
    set(ENABLE_PERF_TESTING OFF)
    set(ENABLE_PYTHON OFF)
    set(ENABLE_PACKAGING OFF)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)

    # Grab the project root in a way that works for projects that include us.
    set(_dorado_root ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..)

    if(WIN32)
        # On windows we need to build a static lib for zstd as there's no prebuilt distro
        set(ZSTD_BUILD_DIR ${CMAKE_BINARY_DIR}/cmake-build-zstd)
        execute_process(COMMAND
            cmake
                -S ${DORADO_3RD_PARTY_SOURCE}/zstd/build/cmake
                -B ${ZSTD_BUILD_DIR}
                -A x64
                -D CMAKE_CONFIGURATION_TYPES=Release
                -D ZSTD_BUILD_SHARED=OFF
                -G ${CMAKE_GENERATOR}
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

    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        # The GCC8 CI build needs a newer version of zstd than provided by the docker image.
        set(ZSTD_BUILD_DIR ${CMAKE_BINARY_DIR}/cmake-build-zstd)
        set(ZSTD_INSTALL_DIR ${CMAKE_BINARY_DIR}/cmake-install-zstd)
        execute_process(COMMAND
            cmake
                -S ${DORADO_3RD_PARTY_SOURCE}/zstd/build/cmake
                -B ${ZSTD_BUILD_DIR}
                -D CMAKE_INSTALL_PREFIX=${ZSTD_INSTALL_DIR}
                -D ZSTD_BUILD_SHARED=OFF
            COMMAND_ERROR_IS_FATAL ANY
        )
        execute_process(
            COMMAND cmake --build ${ZSTD_BUILD_DIR} --target install
            COMMAND_ERROR_IS_FATAL ANY
        )

        list(PREPEND CMAKE_MODULE_PATH "${_dorado_root}/dorado/3rdparty/hdf_plugins/cmake")
        # Findzstd.cmake uses conan variables to determine where things are.
        set(CONAN_INCLUDE_DIRS_RELEASE ${ZSTD_INSTALL_DIR}/include)
        set(CONAN_LIB_DIRS_RELEASE ${ZSTD_INSTALL_DIR}/lib)
        find_package(zstd 1.3.6 REQUIRED)

    else()
        # Some platforms need the Findzstd.cmake from hdf_plugins, but prepend it in case we already
        # have one in the path.
        list(PREPEND CMAKE_MODULE_PATH "${_dorado_root}/dorado/3rdparty/hdf_plugins/cmake")
        # hdf_plugins looks for 1.3.1 but the libarrow in POD5 requires 1.3.6 minimum.
        find_package(zstd 1.3.6 REQUIRED)

    endif()

    add_subdirectory(${_dorado_root}/dorado/3rdparty/hdf_plugins EXCLUDE_FROM_ALL)
endfunction()
