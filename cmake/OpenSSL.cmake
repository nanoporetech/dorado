set(OPENSSL_USE_STATIC_LIBS TRUE)

if(APPLE)
    if(NOT DEFINED OPENSSL_ROOT_DIR)
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            set(OPENSSL_ROOT_DIR "/usr/local/opt/openssl@1.1")
        else()
            set(OPENSSL_ROOT_DIR "/opt/homebrew/opt/openssl@1.1")
        endif()
    endif()
elseif(WIN32)
    if(NOT DEFINED OPENSSL_ROOT_DIR)
        download_and_extract(https://nanoporetech.box.com/shared/static/paqqcwfpdjo3eqaghu2denk9vgnis2ph.zip openssl-win)
	    set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY}/openssl-win)
    endif()
endif()

set(CMAKE_PREFIX_PATH ${OPENSSL_ROOT_DIR} ${CMAKE_PREFIX_PATH}) # put the selected openssl path before any older imported one.

find_package(OpenSSL REQUIRED QUIET)
