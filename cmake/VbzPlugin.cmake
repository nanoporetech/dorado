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
        )
        execute_process(COMMAND cmake --build ${ZSTD_BUILD_DIR} --config Release)

        # On windows we need to tell hdf_plugins where we put the built zstd lib
        set(CONAN_INCLUDE_DIRS_RELEASE ${DORADO_3RD_PARTY_SOURCE}/zstd/lib)
        set(CONAN_INCLUDE_DIRS_DEBUG ${DORADO_3RD_PARTY_SOURCE}/zstd/lib)
        set(CONAN_LIB_DIRS_RELEASE ${ZSTD_BUILD_DIR}/lib/Release)
        set(CONAN_LIB_DIRS_DEBUG ${ZSTD_BUILD_DIR}/lib/Debug)

        install(FILES ${ZSTD_BUILD_DIR}/lib/Release/zstd.dll DESTINATION bin)
    else()
        # Some platforms need the Findzstd.cmake from hdf_plugins, but append it in case we already
        # have one in the path.
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dorado/3rdparty/hdf_plugins/cmake")
        # hdf_plugins looks for 1.3.1 but the libarrow in POD5 requires 1.3.6 minimum.
        find_package(zstd 1.3.6 REQUIRED)
    endif()

    add_subdirectory(dorado/3rdparty/hdf_plugins EXCLUDE_FROM_ALL)
endfunction()
