if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # CUDA toolkit DLLs we depend on:
    set(VERSIONED_CUDA_LIBS
        libcublas*.so*
        libcudart*.so*
        libnvrtc*.so*
        libnvToolsExt*.so*
    )

    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        list(APPEND VERSIONED_CUDA_LIBS
            "*cusparse.so*"
            "*curand.so*"
            "*cusolver.so*"
            "*cufft.so*"
            "*cupti.so*"
        )
    endif()

    foreach(LIB IN LISTS VERSIONED_CUDA_LIBS)
        # torch may bundle it's own specific copy of the cuda libs. if it does, we want everything to point at them
        file(GLOB TORCH_CUDA_LIBS "${TORCH_LIB}/lib/${LIB}")
        if(TORCH_CUDA_LIBS)
            # Sort the list so that we process in order: libX.so -> libX.so.1 -> libX.so.1.1.1
            list(SORT TORCH_CUDA_LIBS)
            foreach(TORCH_CUDA_LIB IN LISTS TORCH_CUDA_LIBS)
                # create links to the torch bundled libs with hashes in the name
                # e.g. libcublas.so.11 => libcublas-3b81d170.so.11
                set(target ${TORCH_CUDA_LIB})
                string(REGEX REPLACE "-[0-9a-f]+[.]" "." link ${target})
                get_filename_component(target_name ${target} NAME)
                get_filename_component(link_name ${link} NAME)
                if (NOT target_name STREQUAL link_name)
                    execute_process(
                        COMMAND ln -rfs ${target_name} ${link_name}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        COMMAND_ERROR_IS_FATAL ANY
                    )
                    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${link_name} DESTINATION lib COMPONENT redist_libs)

                    # create links to the versioned links above
                    # e.g. libcublas.so => libcublas.so.11
                    string(REGEX REPLACE "[.]so[.0-9]*$" ".so" base_link ${link_name})
                    if (NOT base_link STREQUAL link_name)
                        execute_process(
                            COMMAND ln -rfs ${link_name} ${base_link}
                            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                            COMMAND_ERROR_IS_FATAL ANY
                        )
                        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${base_link} DESTINATION lib COMPONENT redist_libs)
                    endif()
                endif()
            endforeach()
        else()
            # bundle the libraries from the cuda toolkit
            file(GLOB NATIVE_CUDA_LIBS "${CUDAToolkit_TARGET_DIR}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib/${LIB}")
            install(FILES ${NATIVE_CUDA_LIBS} DESTINATION lib COMPONENT redist_libs)
        endif()
    endforeach()

    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.so*")
    install(FILES ${TORCH_DLLS} DESTINATION lib COMPONENT redist_libs)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*" AND ${TORCH_VERSION} VERSION_GREATER_EQUAL 2.0)
        find_library(NUMA_DLL numa REQUIRED)
        get_filename_component(NUMA_DLL_PATH ${NUMA_DLL} DIRECTORY)
        file(GLOB NUMA_DLLS "${NUMA_DLL_PATH}/libnuma.so*")
        install(FILES ${NUMA_DLLS} DESTINATION lib COMPONENT redist_libs)
    endif()

elseif(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dll")
    install(FILES ${TORCH_DLLS} DESTINATION bin COMPONENT redist_libs)
    file(GLOB HTSLIB_DLLS "${HTSLIB_DIR}/*.dll")
    install(FILES ${HTSLIB_DLLS} DESTINATION bin COMPONENT redist_libs)

elseif(APPLE)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dylib")
    install(FILES ${TORCH_DLLS} DESTINATION lib COMPONENT redist_libs)

endif()
