if(WIN32)
    message(STATUS "Fetching htslib from Box")
    download_and_extract(https://nanoporetech.box.com/shared/static/9dnctbjw86d20qq8l8tw3dk93hu1nrul.gz htslib-win)
    set(HTSLIB_DIR ${PROJECT_SOURCE_DIR}/dorado/3rdparty/htslib-win CACHE STRING
                "Path to htslib repo")
    set(HTSLIB_LIBRARIES hts-3)
    link_directories(${HTSLIB_DIR})
else()
    message(STATUS "Building htslib")
    set(HTSLIB_DIR ${PROJECT_SOURCE_DIR}/dorado/3rdparty/htslib CACHE STRING
                "Path to htslib repo")
    set(MAKE_COMMAND make)
    set(HTSLIB_INSTALL ${MAKE_COMMAND} install prefix=${CMAKE_BINARY_DIR}/3rdparty/htslib)
    set(htslib_PREFIX ${CMAKE_BINARY_DIR}/3rdparty/htslib)
    include(ExternalProject)
    ExternalProject_Add(htslib_project
            PREFIX ${htslib_PREFIX}
            SOURCE_DIR ${HTSLIB_DIR}
            BUILD_IN_SOURCE 1
            CONFIGURE_COMMAND autoheader && autoconf && ${HTSLIB_DIR}/configure --disable-bz2 --disable-lzma --disable-libcurl --disable-s3 --disable-gcs
            BUILD_COMMAND "${HTSLIB_INSTALL}"
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS ${htslib_PREFIX}/lib/libhts.a
            LOG_CONFIGURE 0
            LOG_BUILD 0
            LOG_TEST 0
            LOG_INSTALL 0
            )

    include_directories(${htslib_PREFIX}/include/htslib)
    set(HTSLIB_LIBRARIES htslib)
    add_library(htslib STATIC IMPORTED)
    set_property(TARGET htslib APPEND PROPERTY IMPORTED_LOCATION ${htslib_PREFIX}/lib/libhts.a)
    message(STATUS "Done Building htslib")
endif()
