if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # CUDA toolkit DLLs we depend on:
    set(REQUIRED_CUDA_LIBS
        libcublasLt.so*
        libcublas.so*
        libcudart*.so*
        libnvrtc*.so*
        libnvToolsExt*.so*
        libnvJitLink*.so*
        libcusolver.so*
        libcusparse.so*
        libcufft.so*
    )

    # bundle the libraries from the cuda toolkit
    foreach(LIB IN LISTS REQUIRED_CUDA_LIBS)
        file(GLOB NATIVE_CUDA_LIBS "${CUDAToolkit_TARGET_DIR}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib/${LIB}")
        foreach(COMPONENT IN LISTS DORADO_RUNTIME_COMPONENTS)
            install(FILES ${NATIVE_CUDA_LIBS} DESTINATION lib COMPONENT ${COMPONENT})
        endforeach()
    endforeach()

    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.so*")
    foreach(COMPONENT IN LISTS DORADO_RUNTIME_COMPONENTS)
        install(FILES ${TORCH_DLLS} DESTINATION lib COMPONENT ${COMPONENT})
    endforeach()

elseif(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dll")
    foreach(COMPONENT IN LISTS DORADO_RUNTIME_COMPONENTS)
        install(FILES ${TORCH_DLLS} DESTINATION bin COMPONENT ${COMPONENT})
    endforeach()

elseif(APPLE)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dylib")
    foreach(COMPONENT IN LISTS DORADO_RUNTIME_COMPONENTS)
        install(FILES ${TORCH_DLLS} DESTINATION lib COMPONENT ${COMPONENT})
    endforeach()

endif()
