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
    set(hash__libkoi__0_5_4__Linux__aarch64__cuda__10_2 "3026e8be1f93801e8439a35bbd955a76c0f228446fc188d38083549f3980e523")
    set(hash__libkoi__0_5_4__Linux__aarch64__cuda__12_6 "64900160651c3a6389b3bfa1392b8341006a7511213d68d34848ce3d6812b3ba")
    set(hash__libkoi__0_5_4__Linux__x86_64__cuda__11_8 "9630a45e72b2a3ddc47519dc9c37679ff9486f6968ae1e982d41530b7521f51a")
    set(hash__libkoi__0_5_4__Linux__x86_64__cuda__12_0 "1c568a2823fb15a92a782eabe04989c2a149425b49e43be885e2fecfeb809abd")
    set(hash__libkoi__0_5_4__Linux__x86_64__cuda__12_4 "5bdb57114d73857184626fa26fa59863213f2f54c9be5a2d664982655515e50b")
    set(hash__libkoi__0_5_4__Linux__x86_64__cuda__12_8 "dac6791a254f8edf582b8617718f30a6f648dc0259f9286ed9571df0cf83d800")
    set(hash__libkoi__0_5_4__Windows__AMD64__cuda__11_8 "a2e9d0d3a462bc35d02789eeacfc465da69fc4c71d3469f66a7b0f9039420f70")
    set(hash__libkoi__0_5_4__Windows__AMD64__cuda__12_0 "6633797074f527ec5707c0dc950fae02d4dc9014fd663b3300cf80b42b14e68f")
    set(hash__libkoi__0_5_4__Windows__AMD64__cuda__12_4 "73c959f33780a2678a74168c88b32b0e632cb66ae52d6b6f44384d6d6b648879")
    set(hash__libkoi__0_5_4__Windows__AMD64__cuda__12_8 "8ae8dfc6902de981e71098279f73419fbb4b8ff4171cb44eedba82244ac76552")

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

    set(KOI_VERSION 0.5.4)
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
