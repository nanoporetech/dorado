function(add_hdf_vbz_plugin)
    set(ENABLE_CONAN OFF)
    set(ENABLE_PERF_TESTING OFF)
    set(ENABLE_PYTHON OFF)
    set(ENABLE_PACKAGING OFF)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)

    if(WIN32)
        # On windows we need to build a static lib for zstd as there's no prebuilt distro
        execute_process(COMMAND cmake -S ${DORADO_3RD_PARTY}/zstd/build/cmake -B ${DORADO_3RD_PARTY}/cmake-build-zstd -A x64)
        execute_process(COMMAND cmake --build ${DORADO_3RD_PARTY}/cmake-build-zstd --config Release)

        # On windows we need to tell hdf_plugins where we put the built zstd lib
        set(CONAN_INCLUDE_DIRS_RELEASE ${DORADO_3RD_PARTY}/zstd/lib)
        set(CONAN_INCLUDE_DIRS_DEBUG ${DORADO_3RD_PARTY}/zstd/lib)
        set(CONAN_LIB_DIRS_RELEASE ${DORADO_3RD_PARTY}/cmake-build-zstd/lib/Release)
        set(CONAN_LIB_DIRS_DEBUG ${DORADO_3RD_PARTY}/cmake-build-zstd/lib/Debug)

        install(FILES ${DORADO_3RD_PARTY}/cmake-build-zstd/lib/Release/zstd.dll DESTINATION bin)
    endif()
   
    add_subdirectory(dorado/3rdparty/hdf_plugins EXCLUDE_FROM_ALL)
endfunction()
