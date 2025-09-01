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
        install(FILES ${NATIVE_CUDA_LIBS} DESTINATION lib COMPONENT redist_libs)
    endforeach()

    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.so*")
    install(FILES ${TORCH_DLLS} DESTINATION lib COMPONENT redist_libs)

elseif(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dll")
    install(FILES ${TORCH_DLLS} DESTINATION bin COMPONENT redist_libs)
    file(GLOB HTSLIB_DLLS "${HTSLIB_DIR}/*.dll")
    install(FILES ${HTSLIB_DLLS} DESTINATION bin COMPONENT redist_libs)

elseif(APPLE)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dylib")
    install(FILES ${TORCH_DLLS} DESTINATION lib COMPONENT redist_libs)

endif()
