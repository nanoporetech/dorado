# CMake errors out if we try to create the pod5_libs target twice, which happens in ont_core.
include_guard(GLOBAL)

set(POD5_VERSION 0.3.39)
set(POD5_DIR pod5-${POD5_VERSION}-${CMAKE_SYSTEM_NAME})
set(POD5_REPO "https://github.com/nanoporetech/pod5-file-format")

if(DEFINED DORADO_POD5_DIR)
    # Use the existing pod5 we have been pointed at, in the same way as DORADO_LIBTORCH_DIR.
    message(STATUS "Using existing pod5 at ${DORADO_POD5_DIR}")
    set(POD5_ROOT ${DORADO_POD5_DIR})
else()
    # Otherwise download a new one.
    set(POD5_ROOT ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR})
endif()

# If we're building with ASAN enabled then we don't want to statically link to POD5 since that
# also forces us to link to libarrow which causes issues with std::vector due to parts of the
# implementation being inlined with different instrumentation settings.
# See https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow#false-positives
if(ECM_ENABLE_SANITIZERS MATCHES "address")
    set(POD5_STATIC FALSE)
else()
    set(POD5_STATIC TRUE)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LIB_DIR "lib64")
    if(DEFINED DORADO_POD5_DIR AND NOT EXISTS "${POD5_ROOT}/${LIB_DIR}")
        # Installs that aren't from the release archives may use a different libdir.
        set(LIB_DIR "lib")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
        set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-linux-arm64.tar.gz")
        set(POD5_HASH "04902f353a9a8d642ba6d58b1c063e002dfd481c35f47e7f189035bb9ecc9baa")
    else()
        set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-linux-x64.tar.gz")
        set(POD5_HASH "7fcf1d9f1da6f1bd13167782e900462fbadb0db57c666911bdba9bbdfd4e7e06")
    endif()
    if(POD5_STATIC)
        set(POD5_LIBRARIES
            ${POD5_ROOT}/${LIB_DIR}/libpod5_format.a
            ${POD5_ROOT}/${LIB_DIR}/libarrow.a
            ${POD5_ROOT}/${LIB_DIR}/libjemalloc_pic.a
            ${POD5_ROOT}/${LIB_DIR}/libzstd.a
        )
    else()
        set(POD5_LIBRARIES ${POD5_ROOT}/lib/libpod5_format.so)
    endif()
elseif(APPLE)
    set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-osx-14.0-arm64.tar.gz")
    set(POD5_HASH "710db33a00ca041094588b7c7b5355ad45611e6160e2e3ec8763e8be34f4ac91")
    if(POD5_STATIC)
        set(POD5_LIBRARIES
            ${POD5_ROOT}/lib/libpod5_format.a
            ${POD5_ROOT}/lib/libarrow.a
            ${POD5_ROOT}/lib/libzstd.a
        )
    else()
        set(POD5_LIBRARIES ${POD5_ROOT}/lib/libpod5_format.dylib)
    endif()
elseif(WIN32)
    set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-win-x64.tar.gz")
    set(POD5_HASH "e034cb86e10fa64ce23969c47f84d06ecd13da07992593a92c743bc82c09bb5a")
    set(POD5_LIBRARIES
        ${POD5_ROOT}/lib/pod5_format.lib
        ${POD5_ROOT}/lib/arrow_static.lib
        ${POD5_ROOT}/lib/zstd_static.lib
        bcrypt.lib
    )
endif()

if(NOT DEFINED DORADO_POD5_DIR)
    download_and_extract(${POD5_URL} ${POD5_DIR} ${POD5_HASH})
endif()

# Create the target which other libraries can link to.
add_library(pod5_libs INTERFACE)
target_link_libraries(pod5_libs INTERFACE ${POD5_LIBRARIES})
target_include_directories(pod5_libs SYSTEM INTERFACE ${POD5_ROOT}/include)

# pod5 makes use of threads and jemalloc requires the dl* symbols, so make sure to link to them.
find_package(Threads REQUIRED)
add_library(pod5_deps INTERFACE)
target_link_libraries(pod5_deps INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
target_link_libraries(pod5_libs INTERFACE pod5_deps)

# We need the libraries installed if they're not static.
if (NOT POD5_STATIC)
    install(
        FILES ${POD5_LIBRARIES}
        DESTINATION lib
    )
endif()
