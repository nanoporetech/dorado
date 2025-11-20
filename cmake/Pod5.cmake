# CMake errors out if we try to create the pod5_libs target twice, which happens in ont_core.
include_guard(GLOBAL)

set(POD5_VERSION 0.3.36)
set(POD5_DIR pod5-${POD5_VERSION}-${CMAKE_SYSTEM_NAME})
set(POD5_REPO "https://github.com/nanoporetech/pod5-file-format")

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
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
        set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-linux-arm64.tar.gz")
        set(POD5_HASH "47ed06bad56882e4f4f71be9903153421f1cf6fcbecf01e47fee358c77c5b7ac")
    else()
        set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-linux-x64.tar.gz")
        set(POD5_HASH "96fa832bdf93e8e6898aba25deae8a27b6a0aa8403613f205ea3616a47d11bd5")
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
    set(POD5_HASH "777b83d8585a304b518d1d89a77998bcb4cfe583d0a92915ccd51e626dcd7006")
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
    set(POD5_HASH "186cc2e0b137a4a8f80cdb000d9e127730efa15ca6a4b2dc38460b35778f7c8e")
    set(POD5_LIBRARIES
        ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/pod5_format.lib
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

# We need the libraries installed if they're not static.
if (NOT POD5_STATIC)
    install(
        FILES ${POD5_LIBRARIES}
        DESTINATION lib
    )
endif()
