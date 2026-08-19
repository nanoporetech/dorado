find_package(ZLIB QUIET)
if(WIN32 AND NOT TARGET ZLIB::ZLIB)
    set(ZLIB_VER 1.3.1)
    set(ZLIB_HASH "17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c")

    # On windows, we need to build Zlib
    include(FetchContent)
    FetchContent_Declare(
        zlib
        URL https://github.com/madler/zlib/archive/refs/tags/v${ZLIB_VER}.tar.gz
        URL_HASH SHA256=${ZLIB_HASH}
        SOURCE_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/zlib-${ZLIB_VER}/zlib-${ZLIB_VER}
        BINARY_DIR ${CMAKE_BINARY_DIR}/zlib-${ZLIB_VER}/build
    )
    FetchContent_MakeAvailable(zlib)

    if(TARGET zlibstatic)
        set(ZLIB_TARGET zlibstatic)
    elseif(TARGET zlib)
        set(ZLIB_TARGET zlib)
    else()
        message(FATAL_ERROR "ZLIB=${ZLIB_VER} was fetched, but no zlib target was created")
    endif()

    add_library(ZLIB::ZLIB INTERFACE IMPORTED)
    set_target_properties(ZLIB::ZLIB PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${zlib_BINARY_DIR};${zlib_SOURCE_DIR}"
        INTERFACE_LINK_LIBRARIES ${ZLIB_TARGET}
    )

    set(ZLIB_FOUND TRUE)
    set(ZLIB_INCLUDE_DIR ${zlib_SOURCE_DIR})
    set(ZLIB_INCLUDE_DIRS ${zlib_BINARY_DIR} ${zlib_SOURCE_DIR})
    set(ZLIB_LIBRARY ZLIB::ZLIB)
    set(ZLIB_LIBRARIES ZLIB::ZLIB)
    set(ZLIB_VERSION ${ZLIB_VER})
    set(ZLIB_VERSION_STRING ${ZLIB_VER})
endif()

if(NOT TARGET ZLIB::ZLIB)
    find_package(ZLIB REQUIRED)
endif()
