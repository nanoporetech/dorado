if(ECM_ENABLE_SANITIZERS)
  set(OPENSSL_USE_STATIC_LIBS FALSE)
else()
  set(OPENSSL_USE_STATIC_LIBS TRUE)
endif()

if(NOT DEFINED OPENSSL_ROOT_DIR)
    if(APPLE)
        set(OPENSSL_ROOT_DIR "/opt/homebrew/opt/openssl@3")
    elseif(WIN32)
        download_and_extract(
            ${DORADO_CDN_URL}/openssl3-win.zip
            openssl3-win
            "2b4dcaff48250428b48e28eb12f32138843af82eba5542fc8c9cbbd364a9451c"
        )
        set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl3-win)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            download_and_extract(
                ${DORADO_CDN_URL}/openssl3-linux-x86_64.zip
                openssl3-Linux-x86_64
                "cc5790f3b58437e81cf0b523665de0cf1fb34e67c2a23181c11d1a051c18a995"
            )
            set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl3-Linux-x86_64)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
            if(${CUDAToolkit_VERSION} VERSION_LESS 11.0)
                download_and_extract(
                    ${DORADO_CDN_URL}/openssl3-linux-aarch64-gcc7.zip
                    openssl3-Linux-aarch64
                    "d9788e098b830612678780c623071fc1ffde7944639693970a295cef7a56b9b1"
                )
            else()
                download_and_extract(
                    ${DORADO_CDN_URL}/openssl3-linux-aarch64.zip
                    openssl3-Linux-aarch64
                    "f69511b0f1702e425fa0ec2f915c848a7ce451a30eff6492bdfa12bdbd860a93"
                )
            endif()
            set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl3-Linux-aarch64)
        endif()
    endif()
else()
    message(STATUS "Using existing OpenSSL at ${OPENSSL_ROOT_DIR}")
endif()

set(CMAKE_PREFIX_PATH ${OPENSSL_ROOT_DIR} ${CMAKE_PREFIX_PATH}) # put the selected openssl path before any older imported one.

find_package(OpenSSL REQUIRED QUIET)
