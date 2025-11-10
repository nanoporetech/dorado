OPTION(BUILD_KOI_FROM_SOURCE OFF)

function(get_best_compatible_koi_version KOI_CUDA)
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        # Koi provides binaries for these cuda versions when targeting aarch64
        set(SUPPORTED_VERSIONS 13.0 12.6)
    else()
        set(SUPPORTED_VERSIONS 13.0 12.8 11.8)
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
    set(hash__libkoi__0_6_6__Linux__aarch64__cuda__12_6 "39ebffbdebcf8e3bd2a1e87c1f80ee45f1fcb2006eecd2ac2fe3d09f464f2e98")
    set(hash__libkoi__0_6_6__Linux__aarch64__cuda__13_0 "a360d120ae54ac17d171112310796e6f818ce00e2f6f6514fb392cdb099ccb91")
    set(hash__libkoi__0_6_6__Linux__x86_64__cuda__11_8 "32d002f7e020ed164abe9115ae03e96a0f4ac550e6535220c0658c7c52040849")
    set(hash__libkoi__0_6_6__Linux__x86_64__cuda__12_8 "4422dc04205d5e8766cb1b2842c990d0eb360b6037c055018918105fc0d4b8d7")
    set(hash__libkoi__0_6_6__Linux__x86_64__cuda__13_0 "abf5b45d158821f4e4a0beb81e93ffd5aa22a35fa7598c0dd69369a458cfb474")
    set(hash__libkoi__0_6_6__Windows__AMD64__cuda__12_8 "cde31f9467f967f1f47e213c3af662c18c16d5e7a0c459dd77d0730d5222331c")
    set(hash__libkoi__0_6_6__Windows__AMD64__cuda__13_0 "748fdcd80a659b43fb054ef63d807753160b0f5f89a491484b004b65d572e462")

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

    set(KOI_VERSION 0.6.6)
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
        add_subdirectory(${KOI_DIR})

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
