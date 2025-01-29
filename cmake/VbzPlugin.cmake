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

    # Add any extra flags to the cmake configure step.
    if (WIN32)
        set(_extra_zstd_cmake_flags -A x64)
    else()
        set(_extra_zstd_cmake_flags)
    endif()

    # All builds should use our own build of zstd.
    set(ZSTD_BUILD_DIR ${CMAKE_BINARY_DIR}/cmake-build-zstd)
    set(ZSTD_INSTALL_DIR ${CMAKE_BINARY_DIR}/cmake-install-zstd)
    if (NOT EXISTS ${ZSTD_INSTALL_DIR})
        execute_process(COMMAND
            cmake
                -S ${DORADO_3RD_PARTY_SOURCE}/zstd/build/cmake
                -B ${ZSTD_BUILD_DIR}
                -D CMAKE_CONFIGURATION_TYPES=Release
                -D CMAKE_INSTALL_PREFIX=${ZSTD_INSTALL_DIR}
                -D ZSTD_BUILD_SHARED=OFF
                -G ${CMAKE_GENERATOR}
                ${_extra_zstd_cmake_flags}
            COMMAND_ERROR_IS_FATAL ANY
        )
        execute_process(COMMAND
            cmake
                --build ${ZSTD_BUILD_DIR}
                --config Release
                --target install
                --parallel
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()

    # Add the install folder to cmake's search paths so that it gets picked over the system.
    list(PREPEND CMAKE_PREFIX_PATH ${ZSTD_INSTALL_DIR})

    # Go find it.
    find_package(zstd 1.4.8 REQUIRED)

    # Grab the project root in a way that works for projects that include us.
    set(_dorado_root ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..)
    add_subdirectory(${_dorado_root}/dorado/3rdparty/hdf_plugins EXCLUDE_FROM_ALL)
endfunction()
