message(STATUS "Building htslib")
set(HTSLIB_DIR ${PROJECT_SOURCE_DIR}/dorado/3rdparty/htslib/ CACHE STRING
            "Path to htslib repo")
set(MAKE_COMMAND make)
set(HTSLIB_INSTALL ${MAKE_COMMAND} install prefix=${CMAKE_BINARY_DIR}/3rdparty/htslib)
set(htslib_PREFIX ${CMAKE_BINARY_DIR}/3rdparty/htslib)
include(ExternalProject)
ExternalProject_Add(htslib_project
        PREFIX ${htslib_PREFIX}
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/dorado/3rdparty/htslib
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND autoheader && autoconf && ${HTSLIB_DIR}configure --disable-bz2 --disable-lzma --disable-libcurl --disable-s3 --disable-gcs
        BUILD_COMMAND "${HTSLIB_INSTALL}"
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/3rdparty/htslib/lib/libhts.a
        LOG_CONFIGURE 0
        LOG_BUILD 0
        LOG_TEST 0
        LOG_INSTALL 0
        )

include_directories(${CMAKE_BINARY_DIR}/3rdparty/htslib/include/htslib)
add_library(htslib STATIC IMPORTED)
set_property(TARGET htslib APPEND PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/3rdparty/htslib/lib/libhts.a)
message(STATUS "Done Building htslib")
