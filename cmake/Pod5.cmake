# CMake errors out if we try to create the pod5_libs target twice, which happens in ont_core.
include_guard(GLOBAL)

set(POD5_VERSION 0.3.27)
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
      set(POD5_HASH "fc6b331f1c27651518894ad56403853f2aaffc3b50ebdbf0adafc9b00628bb70")
    else()
      set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-linux-x64.tar.gz")
      set(POD5_HASH "c610e70851d92644cc1589a8f2dfdb14c708949f2a06bbbf003945e39882a324")
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
    set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-osx-11.0-arm64.tar.gz")
    set(POD5_HASH "f8ae07a182ec8898b9b4433f8c55465bd6767b709b5c66f82dad2580872acaa0")
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
    set(POD5_HASH "9be398d65a7ea76c55754c465f9612f8a94347ebcffd73b267c657732463bb31")
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
