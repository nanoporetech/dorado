if(TARGET OpenSSL::SSL)
  return()
endif()

if(ECM_ENABLE_SANITIZERS)
  set(OPENSSL_USE_STATIC_LIBS FALSE)
else()
  set(OPENSSL_USE_STATIC_LIBS TRUE)
endif()

if(NOT DEFINED OPENSSL_ROOT_DIR)
    set(OPENSSL_VERSION 3.5.7)
    if(APPLE)
        download_and_extract(
            ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-macos-aarch64.zip
            openssl-${OPENSSL_VERSION}-macos-aarch64
            "a445b60441f86e19211b2b743d34e22c3fc2128e0a02911118b8f871309f35ac"
        )
        set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-macos-aarch64)
    elseif(WIN32)
        download_and_extract(
            ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-win.zip
            openssl-${OPENSSL_VERSION}-win
            "15b0221b56cdcab0f6b842c063d7f154e6b1e17f6ceb7a7ab8f777411a05eab3"
        )
        set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-win)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            download_and_extract(
                ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-linux-x86_64.zip
                openssl-${OPENSSL_VERSION}-Linux-x86_64
                "9c195f46bb8ad5c4ab336c089108d1cfb5f71b7e541a010154a1773fa2867188"
            )
            set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-Linux-x86_64)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
            download_and_extract(
                ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-linux-aarch64.zip
                openssl-${OPENSSL_VERSION}-Linux-aarch64
                "9e4b215fa73350ea6657f95a97b7d86987c205e70938086029a6fc4abdcd7b86"
            )
            set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-Linux-aarch64)
        endif()
    endif()
else()
    message(STATUS "Using existing OpenSSL at ${OPENSSL_ROOT_DIR}")
endif()

set(CMAKE_PREFIX_PATH ${OPENSSL_ROOT_DIR} ${CMAKE_PREFIX_PATH}) # put the selected openssl path before any older imported one.

find_package(OpenSSL REQUIRED QUIET)
