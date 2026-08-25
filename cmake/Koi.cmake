function(get_best_compatible_koi_version KOI_CUDA)
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        # Koi provides binaries for these cuda versions when targeting aarch64
        set(SUPPORTED_VERSIONS 13.0 12.6)
    else()
        set(SUPPORTED_VERSIONS 13.0 12.8)
    endif()

    list(SORT SUPPORTED_VERSIONS COMPARE NATURAL ORDER DESCENDING)
    foreach(SUPPORTED_VERSION IN LISTS SUPPORTED_VERSIONS)
        if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL ${SUPPORTED_VERSION})
            set(${KOI_CUDA} ${SUPPORTED_VERSION} PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "Unsupported CUDA toolkit version: ${CUDAToolkit_VERSION}")
endfunction()

function(get_koi_download_hash KOI_DIR KOI_HASH)
    # List of valid hashes.
    set(hash__libkoi__0_7_5__Linux__aarch64__cuda__12_6 "bc3ddab1b0f76f43c357666538f8348209dbd4cf67d9dd9912b1e0a7002d0bf8")
    set(hash__libkoi__0_7_5__Linux__aarch64__cuda__13_0 "384e9b239fd3691054732b0ccde3fd30bbd40a1f59c55fe71128ed60eaa5b06c")
    set(hash__libkoi__0_7_5__Linux__x86_64__cuda__12_8 "bfb3bea6f1bb7f7e6e602d4899db14f422a46519ce69e2d1b645050eba307aad")
    set(hash__libkoi__0_7_5__Linux__x86_64__cuda__13_0 "0600a138761b17467bd3a1f8933c6a6a02e17415a9da377f22bc14357c3101ca")
    set(hash__libkoi__0_7_5__Windows__AMD64__cuda__12_8 "d694b9d71bab92d08a682a354ed74ae8e96c88f1de6007af709efd6ae3951321")
    set(hash__libkoi__0_7_5__Windows__AMD64__cuda__13_0 "f73e45733d619e01878bebaa941254f00e01ff48df7a3d1ac5163bc6bfb91754")

    # Do the lookup.
    string(REPLACE "." "_" hash_key ${KOI_DIR})
    string(REPLACE "-" "__" hash_key ${hash_key})
    set(hash_key "hash__${hash_key}")
    if (NOT DEFINED ${hash_key})
        message(FATAL_ERROR "Missing hash for ${KOI_DIR}")
    endif()
    set(${KOI_HASH} ${${hash_key}} PARENT_SCOPE)
endfunction()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR WIN32)

    set(KOI_VERSION 0.7.5)
    if(DEFINED KOI_DIR)
        message(STATUS "Building Koi from source: ${KOI_DIR}")
        if(NOT EXISTS ${KOI_DIR})
            if(DEFINED GITLAB_CI_TOKEN)
                message("Cloning Koi using CI token")
                set(KOI_REPO https://gitlab-ci-token:${GITLAB_CI_TOKEN}@git.oxfordnanolabs.local/machine-learning/koi.git)
            else()
                message("Cloning Koi using ssh")
                set(KOI_REPO git@git.oxfordnanolabs.local:machine-learning/koi.git)
            endif()
            execute_process(
                COMMAND
                    git clone
                        -b v${KOI_VERSION}
                        # TODO: once we drop centos support we can use these instead of a separate submodule update
                        #--depth 1
                        #--recurse-submodules
                        #--shallow-submodules
                        ${KOI_REPO}
                        ${KOI_DIR}
                COMMAND_ERROR_IS_FATAL ANY
            )
            execute_process(
                COMMAND git submodule update --init --checkout
                WORKING_DIRECTORY ${KOI_DIR}
                COMMAND_ERROR_IS_FATAL ANY
            )
        endif()
        # force the koi build into the binary directory
        # this allows KOI_DIR to be out-of-tree
        add_subdirectory(${KOI_DIR} ${CMAKE_BINARY_DIR}/koi)
    else()
        find_package(CUDAToolkit REQUIRED)
        get_best_compatible_koi_version(KOI_CUDA)
        set(KOI_DIR libkoi-${KOI_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-cuda-${KOI_CUDA})
        get_koi_download_hash(${KOI_DIR} KOI_HASH)

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            download_and_extract(${DORADO_CDN_URL}/${KOI_DIR}.tar.gz ${KOI_DIR} ${KOI_HASH})
            set(KOI_LIBRARY ${DORADO_3RD_PARTY_DOWNLOAD}/${KOI_DIR}/${KOI_DIR}/lib/libkoi.a)
        elseif(WIN32)
            download_and_extract(${DORADO_CDN_URL}/${KOI_DIR}.zip ${KOI_DIR} ${KOI_HASH})
            set(KOI_LIBRARY ${DORADO_3RD_PARTY_DOWNLOAD}/${KOI_DIR}/${KOI_DIR}/lib/koi.lib)
        endif()
        set(KOI_INCLUDE ${DORADO_3RD_PARTY_DOWNLOAD}/${KOI_DIR}/${KOI_DIR}/include)

        add_library(koi STATIC IMPORTED)
        set_target_properties(koi
            PROPERTIES
                IMPORTED_LOCATION ${KOI_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES ${KOI_INCLUDE}
        )

    endif()
endif()
