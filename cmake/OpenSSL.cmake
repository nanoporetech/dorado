if(ECM_ENABLE_SANITIZERS)
  set(OPENSSL_USE_STATIC_LIBS FALSE)
else()
  set(OPENSSL_USE_STATIC_LIBS TRUE)
endif()

if(NOT DEFINED OPENSSL_ROOT_DIR)
    set(OPENSSL_VERSION 3.5.1)
    if(APPLE)
        download_and_extract(
            ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-macos-aarch64.zip
            openssl-${OPENSSL_VERSION}-macos-aarch64
            "d3a7c9dd9eaf09e6667c5ecfd213133fe3235b10a82c36246387e4d38b29ecd3"
        )
        set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-macos-aarch64)
    elseif(WIN32)
        download_and_extract(
            ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-win.zip
            openssl-${OPENSSL_VERSION}-win
            "eeb4300919d6a44c896b30cf9e3afb851980b61b7a9b655c5cf1da9cc78aeb73"
        )
        set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-win)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            download_and_extract(
                ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-linux-x86_64.zip
                openssl-${OPENSSL_VERSION}-Linux-x86_64
                "1f5e232fd4fb6dc441a0ce51eec631125a8585447454bc1e6a4d25ff32869b3a"
            )
            set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-Linux-x86_64)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
            download_and_extract(
                ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-linux-aarch64.zip
                openssl-${OPENSSL_VERSION}-Linux-aarch64
                "38111138e7c22f74d59760ac4144f998ad62ed7e6671a792f952349c3ace13fd"
            )
            set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-Linux-aarch64)
        endif()
    endif()
else()
    message(STATUS "Using existing OpenSSL at ${OPENSSL_ROOT_DIR}")
endif()

set(CMAKE_PREFIX_PATH ${OPENSSL_ROOT_DIR} ${CMAKE_PREFIX_PATH}) # put the selected openssl path before any older imported one.

find_package(OpenSSL REQUIRED QUIET)
