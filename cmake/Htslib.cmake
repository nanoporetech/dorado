if(NOT TARGET htslib) # lazy include guard
    if(WIN32)
        message(STATUS "Fetching htslib")
        download_and_extract(
            ${DORADO_CDN_URL}/htslib-win.tar.gz
            htslib-win
            "7b1719da1ae3d2ea059bb1e7f02e5e3aac57aa41e4fc38d3ab0c20fd68143d08"
        )
        set(HTSLIB_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/htslib-win CACHE STRING
                    "Path to htslib repo")
        add_library(htslib SHARED IMPORTED)
        set_target_properties(htslib PROPERTIES 
            "IMPORTED_IMPLIB" ${HTSLIB_DIR}/hts-3.lib
            "IMPORTED_LOCATION" ${HTSLIB_DIR}/hts-3.dll
            "INTERFACE_INCLUDE_DIRECTORIES" ${HTSLIB_DIR})
        target_link_directories(htslib INTERFACE ${HTSLIB_DIR})
    else()
        message(STATUS "Setting up htslib build")
        set(HTSLIB_DIR ${DORADO_3RD_PARTY_SOURCE}/htslib CACHE STRING "Path to htslib repo")
        set(htslib_PREFIX ${CMAKE_BINARY_DIR}/3rdparty/htslib)

        find_program(MAKE_COMMAND make REQUIRED)
        find_program(AUTOCONF_COMMAND autoconf REQUIRED)
        find_program(AUTOHEADER_COMMAND autoheader REQUIRED)
        execute_process(
            COMMAND bash -c "${AUTOCONF_COMMAND} -V | sed 's/.* //; q'"
            OUTPUT_VARIABLE AUTOCONF_VERS
            COMMAND_ERROR_IS_FATAL ANY
        )
        if (AUTOCONF_VERS VERSION_GREATER_EQUAL 2.70 AND NOT CMAKE_GENERATOR STREQUAL "Xcode")
            set(AUTOCONF_COMMAND autoreconf --install)
        endif()

        # htslib will only try to build the .so as PIC, but ont_core needs it all as PIC.
        unset(hts_cflags)
        if (LINUX)
            set(hts_cflags "-fPIC")
        endif()

        # If we're building with sanitizers then we need to build htslib the same way.
        if (ECM_ENABLE_SANITIZERS MATCHES "address")
            set(hts_cflags "${hts_cflags} -g -fsanitize=address")
        endif()
        if (ECM_ENABLE_SANITIZERS MATCHES "undefined")
            set(hts_cflags "${hts_cflags} -g -fsanitize=undefined")
        endif()
        if (ECM_ENABLE_SANITIZERS MATCHES "thread")
            set(hts_cflags "${hts_cflags} -g -fsanitize=thread")
        endif()

        unset(hts_configure_flags)
        if (hts_cflags)
            set(hts_configure_flags "CPPFLAGS=${hts_cflags}" "LDFLAGS=${hts_cflags}")
        endif()

        # Parallelise the htslib build too, though not enough to compete with the main build.
        include(ProcessorCount)
        ProcessorCount(nproc)
        math(EXPR htslib_build_jobs "${nproc} / 4")
        if (htslib_build_jobs EQUAL 0)
            set(htslib_build_jobs 1)
        endif()

        # The Htslib build apparently requires BUILD_IN_SOURCE=1, which is a problem when
        # switching between build targets because htscodecs object files aren't regenerated.
        # To avoid this, copy the source tree to a build-specific directory and do the build there.
        set(HTSLIB_BUILD ${CMAKE_BINARY_DIR}/htslib_build)
        file(COPY ${HTSLIB_DIR} DESTINATION ${HTSLIB_BUILD})

        include(ExternalProject)
        ExternalProject_Add(htslib_project
            PREFIX ${HTSLIB_BUILD}
            SOURCE_DIR ${HTSLIB_BUILD}/htslib
            BUILD_IN_SOURCE 1
            CONFIGURE_COMMAND ${AUTOHEADER_COMMAND}
            COMMAND ${AUTOCONF_COMMAND}
            COMMAND ./configure --disable-bz2 --disable-lzma --disable-libcurl --disable-s3 --disable-gcs --without-libdeflate ${hts_configure_flags}
            BUILD_COMMAND ${MAKE_COMMAND} -j${htslib_build_jobs} install prefix=${htslib_PREFIX}
            COMMAND ${INSTALL_NAME}
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS ${htslib_PREFIX}/lib/libhts.a
            LOG_CONFIGURE 0
            LOG_BUILD 0
            LOG_TEST 0
            LOG_INSTALL 0
        )

        add_library(htslib STATIC IMPORTED)
        # Need to ensure this directory exists before we can add it to INTERFACE_INCLUDE_DIRECTORIES
        file(MAKE_DIRECTORY ${htslib_PREFIX}/include)
        set_target_properties(htslib 
            PROPERTIES 
                "IMPORTED_LOCATION" ${htslib_PREFIX}/lib/libhts.a
                "INTERFACE_INCLUDE_DIRECTORIES" ${htslib_PREFIX}/include)
        message(STATUS "Done configuring htslib")

        # Make sure that the project is built before any targets try to use it.
        add_dependencies(htslib htslib_project)
    endif()
endif()
