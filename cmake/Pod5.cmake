set(POD5_VERSION 0.0.41)
set(POD5_DIR pod5-${POD5_VERSION}-${CMAKE_SYSTEM_NAME})
set(POD5_REPO "https://github.com/nanoporetech/pod5-file-format")
set(POD5_INCLUDE dorado/3rdparty/${POD5_DIR}/include)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/pod5-file-format-${POD5_VERSION}-linux-x64.tar.gz")
    set(POD5_LIBRARIES
      ${DORADO_3RD_PARTY}/${POD5_DIR}/lib64/libpod5_format.a
      ${DORADO_3RD_PARTY}/${POD5_DIR}/lib64/libarrow.a
      ${DORADO_3RD_PARTY}/${POD5_DIR}/lib64/libjemalloc_pic.a
    )
elseif(APPLE)
    set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/pod5-file-format-${POD5_VERSION}-osx-11.0-arm64.tar.gz")
    set(POD5_LIBRARIES
      ${DORADO_3RD_PARTY}/${POD5_DIR}/lib/libpod5_format.a
      ${DORADO_3RD_PARTY}/${POD5_DIR}/lib/libarrow.a
    )
elseif(WIN32)
    set(POD5_URL "${POD5_REPO}/releases/download/${POD5_VERSION}/pod5-file-format-${POD5_VERSION}-win-x64.tar.gz")
    set(POD5_LIBRARIES
      ${DORADO_3RD_PARTY}/${POD5_DIR}/lib/pod5_format.lib
      ${DORADO_3RD_PARTY}/${POD5_DIR}/lib/arrow_static.lib
      bcrypt.lib
    )
endif()

download_and_extract(${POD5_URL} ${POD5_DIR})
