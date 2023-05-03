if(NOT TARGET htslib) # lazy include guard
    if(WIN32)
        message(STATUS "Fetching htslib")
        download_and_extract(https://cdn.oxfordnanoportal.com/software/analysis/htslib-win.tar.gz htslib-win)
        set(HTSLIB_DIR ${DORADO_3RD_PARTY}/htslib-win CACHE STRING
                    "Path to htslib repo")
        add_library(htslib SHARED IMPORTED)
        set_property(TARGET htslib APPEND PROPERTY IMPORTED_IMPLIB ${HTSLIB_DIR}/hts-3.lib)        
        set_property(TARGET htslib APPEND PROPERTY IMPORTED_LOCATION ${HTSLIB_DIR}/hts-3.dll)        
        target_link_directories(htslib INTERFACE ${HTSLIB_DIR})
    else()
        message(STATUS "Building htslib")
        set(HTSLIB_DIR ${DORADO_3RD_PARTY}/htslib CACHE STRING
                    "Path to htslib repo")
        set(MAKE_COMMAND make)
        set(AUTOCONF_COMMAND autoconf)
        execute_process(COMMAND bash -c "autoconf -V | sed 's/.* //; q'"
                OUTPUT_VARIABLE AUTOCONF_VERS)
        if(AUTOCONF_VERS VERSION_GREATER_EQUAL 2.70)
            set(AUTOCONF_COMMAND autoreconf --install)
        endif()
        set(htslib_PREFIX ${CMAKE_BINARY_DIR}/3rdparty/htslib)

        # We need cross-compilation mode for iOS builds.
        if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
            set(CONFIGURE_FLAGS "--host=aarch64-apple-darwin")
        endif()

        include(ExternalProject)
        ExternalProject_Add(htslib_project
                PREFIX ${htslib_PREFIX}
                SOURCE_DIR ${HTSLIB_DIR}
                BUILD_IN_SOURCE 1
                CONFIGURE_COMMAND autoheader
                COMMAND ${AUTOCONF_COMMAND}
                COMMAND ./configure --disable-bz2 --disable-lzma --disable-libcurl --disable-s3 --disable-gcs ${CONFIGURE_FLAGS}
                BUILD_COMMAND ${MAKE_COMMAND} install prefix=${htslib_PREFIX}
                INSTALL_COMMAND ""
                BUILD_BYPRODUCTS ${htslib_PREFIX}/lib/libhts.a
                LOG_CONFIGURE 0
                LOG_BUILD 0
                LOG_TEST 0
                LOG_INSTALL 0
                )

        include_directories(${htslib_PREFIX}/include/htslib)
        add_library(htslib STATIC IMPORTED)
        set_property(TARGET htslib APPEND PROPERTY IMPORTED_LOCATION ${htslib_PREFIX}/lib/libhts.a)
        message(STATUS "Done Building htslib")
    endif()
endif()
