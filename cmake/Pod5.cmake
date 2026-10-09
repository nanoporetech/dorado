# CMake errors out if we try to create the pod5_libs target twice, which happens in ont_core.
include_guard(GLOBAL)

set(POD5_VERSION 0.3.49)
set(POD5_DIR pod5-${POD5_VERSION}-${CMAKE_SYSTEM_NAME})
set(POD5_REPO "https://github.com/nanoporetech/pod5-file-format")

option(POD5_STATIC "Statically link the pod5 library" ON)

# If we're building with ASAN enabled then we don't want to statically link to POD5 since that
# also forces us to link to libarrow which causes issues with std::vector due to parts of the
# implementation being inlined with different instrumentation settings.
# See https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow#false-positives
if(ECM_ENABLE_SANITIZERS MATCHES "address")
    set(POD5_STATIC OFF)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LIB_DIR "lib")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
        set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-linux-arm64.tar.gz")
        set(POD5_HASH "c895a963c5b58e92c805cf8183e7c4584dda6705c11ea555f8dd8635a847f25c")
    else()
        set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-linux-x64.tar.gz")
        set(POD5_HASH "7d9eda9d2e37a08be1b4763f310ab5bf9dc3ab7d4c0bd49b6010602d74aab759")
    endif()
    if(POD5_STATIC)
        set(POD5_LIBRARIES
            ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/${LIB_DIR}/libpod5_format.a
            ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/${LIB_DIR}/libarrow.a
            ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/${LIB_DIR}/libjemalloc_pic.a
            ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/${LIB_DIR}/libzstd.a
        )
    else()
        set(POD5_LIBRARIES ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/libpod5_format.so)
    endif()
elseif(APPLE)
    set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-osx-14.0-arm64.tar.gz")
    set(POD5_HASH "2b17ab8f0cfe5226a9c84cfbfe01ea64a52cd6e240cdf21940c20bc9eddf42c2")
    if(POD5_STATIC)
        set(POD5_LIBRARIES
            ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/libpod5_format.a
            ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/libarrow.a
            ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/libzstd.a
        )
    else()
        set(POD5_LIBRARIES ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/libpod5_format.dylib)
    endif()
elseif(WIN32)
    set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-win-x64.tar.gz")
    set(POD5_HASH "31840a62c9d34211c98785c6055ef6efd3af22b05e281fac40d7f7372ab2e9fe")
    if(POD5_STATIC)
        add_library(pod5_format STATIC IMPORTED GLOBAL)
        set_target_properties(pod5_format
            PROPERTIES
                IMPORTED_LOCATION ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/pod5_format.lib
        )
    else()
        add_library(pod5_format SHARED IMPORTED GLOBAL)
        set_target_properties(pod5_format
            PROPERTIES
                IMPORTED_IMPLIB ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/pod5_format.lib
                IMPORTED_LOCATION ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/bin/pod5_format.dll
        )
    endif()
    set(POD5_LIBRARIES
        pod5_format
        ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/arrow_static.lib
        ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/zstd_static.lib
        bcrypt.lib
    )
endif()

download_and_extract(${POD5_URL} ${POD5_DIR} ${POD5_HASH})

# Create the target which other libraries can link to.
add_library(pod5_libs INTERFACE)
target_link_libraries(pod5_libs INTERFACE ${POD5_LIBRARIES})
target_include_directories(pod5_libs SYSTEM INTERFACE ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/include)

# pod5 makes use of threads and jemalloc requires the dl* symbols, so make sure to link to them.
find_package(Threads REQUIRED)
add_library(pod5_deps INTERFACE)
target_link_libraries(pod5_deps INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
target_link_libraries(pod5_libs INTERFACE pod5_deps)
