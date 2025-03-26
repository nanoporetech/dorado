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
    set(hash__libkoi__0_5_3__Linux__aarch64__cuda__10_2 "4a3c495e73a1880fb7154bdc79a4150e00d00e8c765ea87909a121f3b29c5c54")
    set(hash__libkoi__0_5_3__Linux__aarch64__cuda__11_4 "6f5a8c3f683f46a414b77565eccc0e04c0824830ee491c5320c1699324b02cc3")
    set(hash__libkoi__0_5_3__Linux__aarch64__cuda__12_6 "7821b8021d26b0e985fac6fd09f7dc4db70031b7dc918038a110d2154a670f28")
    set(hash__libkoi__0_5_3__Linux__x86_64__cuda__11_8 "b6da7003d61a0cab9c30a3114b145a7866f983e4b5a0c36a086b4bf882125326")
    set(hash__libkoi__0_5_3__Linux__x86_64__cuda__12_0 "202a64e27706929f3ccf1c2efd0fdb98950deb98f39014d6c38d518e8012ccf8")
    set(hash__libkoi__0_5_3__Linux__x86_64__cuda__12_4 "6cad7dbb730927d10987fd34844901d535c4f323248ca488081e24053e0ea23b")
    set(hash__libkoi__0_5_3__Linux__x86_64__cuda__12_8 "14628f03a46435cb57b708fca5461fb3265a54d412dab21ef1249831cdb217bb")
    set(hash__libkoi__0_5_3__Windows__AMD64__cuda__11_8 "a4a2c0b0eb8567b5ba0f1cd9903e49f8ef1fc4a776af53ab27f9fd09f5953c64")
    set(hash__libkoi__0_5_3__Windows__AMD64__cuda__12_0 "74f73b26474d3a754da5ad3716ee1884aa3f9056055393161f7512ed102a0102")
    set(hash__libkoi__0_5_3__Windows__AMD64__cuda__12_4 "c0c7858056bc681f3b872a01ca9431f139491385be36d76352db0cdd5cd63b20")

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
                        -b ${KOI_VERSION}
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
