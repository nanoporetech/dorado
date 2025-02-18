# CMake errors out if we try to create the pod5_libs target twice, which happens in ont_core.
include_guard(GLOBAL)

set(POD5_VERSION 0.3.23)
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
      if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
        set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-linux-gcc8-arm64.tar.gz")
        set(POD5_HASH "0f980f367d37ee5ae5d6d27647160d54d5fbb84b119feb55a8c905d8a192046a")
        set(LIB_DIR "lib")
      else()
        set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-linux-arm64.tar.gz")
        set(POD5_HASH "d6586a3c5a44f5683318ad7d2e4f6b020ad1e947b05898a82fb0e611f7bff486")
      endif()
    else()
      set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-linux-x64.tar.gz")
      set(POD5_HASH "3cb319ab31ce931dd7c87532b6985167efa29d3a386be1883622e62ebed6c507")
    endif()
    if(POD5_STATIC)
      set(POD5_LIBRARIES
        ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/${LIB_DIR}/libpod5_format.a
        ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/${LIB_DIR}/libarrow.a
        ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/${LIB_DIR}/libjemalloc_pic.a
      )
    else()
      set(POD5_LIBRARIES ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/libpod5_format.so)
    endif()
elseif(APPLE)
    set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-osx-11.0-arm64.tar.gz")
    set(POD5_HASH "688043462efd6f9422df33707ad5ae090460e66e85f6bc7c763cb8db4f1e2e9c")
    if(POD5_STATIC)
      set(POD5_LIBRARIES
        ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/libpod5_format.a
        ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/libarrow.a
      )
    else()
      set(POD5_LIBRARIES ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/libpod5_format.dylib)
    endif()
elseif(WIN32)
    set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/lib_pod5-${POD5_VERSION}-win-x64.tar.gz")
    set(POD5_HASH "282226df35a2857b3eca01c5f70f8e68b1af0acf5ae94a3f08f73d2857170fd0")
    set(POD5_LIBRARIES
      ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/pod5_format.lib
      ${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}/lib/arrow_static.lib
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
