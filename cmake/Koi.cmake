OPTION(BUILD_KOI_FROM_SOURCE OFF)

function(get_best_compatible_koi_version KOI_CUDA)
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        # Koi provides binaries for these cuda versions when targeting aarch64
        set(SUPPORTED_VERSIONS 12.6 11.4 10.2)
    else()
        set(SUPPORTED_VERSIONS 12.8 12.4 12.0 11.8)
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
    set(hash__libkoi__0_5_5__Linux__aarch64__cuda__10_2 "d6b71458fb0dde91d410963083d322cd6d49b955e3c21a4fdaaef578a30a82e5")
    set(hash__libkoi__0_5_5__Linux__aarch64__cuda__12_6 "18d868bd5d33d6a00f9696a912bd039a5ee9a4690fdb4cf567fd4d323b2f39c0")
    set(hash__libkoi__0_5_5__Linux__x86_64__cuda__11_8 "766ad12be34543d7bb4d79129873095edf44b397448a7c3b43adc0fa67a51a83")
    set(hash__libkoi__0_5_5__Linux__x86_64__cuda__12_0 "99c8e3cf29f69e2fb2487b4525c2d1381c17b3962d50fa0a5be2d536042e787d")
    set(hash__libkoi__0_5_5__Linux__x86_64__cuda__12_4 "d482e89773be220cd04820cc267c52a53e41a262e06fd3273952d6f6081c2a09")
    set(hash__libkoi__0_5_5__Linux__x86_64__cuda__12_8 "9478932daa5b85c35f4bd1d870c2d6717d552a17a0ba931eb8e2d8d2fb4a0d97")
    set(hash__libkoi__0_5_5__Windows__AMD64__cuda__11_8 "2753c546b372b5801a1ed495a79b0203603dd0d8b10240bb36cae0da51d12eca")
    set(hash__libkoi__0_5_5__Windows__AMD64__cuda__12_4 "bc2eb837bdf235c9e2bcd91e97ec7dbc7ddbd81c24cef86a1d2fa10e83c8d2e7")
    set(hash__libkoi__0_5_5__Windows__AMD64__cuda__12_8 "bed39b9fed9e207376e4ffa5bc1e94f004b81d66ab7bc75382364b09a6a49552")

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

    set(KOI_VERSION 0.5.5)
    if(BUILD_KOI_FROM_SOURCE)
        set(KOI_DIR "${DORADO_3RD_PARTY_SOURCE}/koi")
        if(NOT EXISTS ${KOI_DIR})
            set(KOI_DIR "${DORADO_3RD_PARTY_DOWNLOAD}/koi")
        endif()
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
        add_subdirectory(${KOI_DIR}/koi/lib)

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
