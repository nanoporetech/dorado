set(TORCH_VERSION 1.10.2)
set(CUDNN_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/dorado/3rdparty/fake_cudnn)
set(CUDNN_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/dorado/3rdparty/fake_cudnn)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(TORCH_URL https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu113.zip)
    set(TORCH_LIB "${CMAKE_SOURCE_DIR}/dorado/3rdparty/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/libtorch")
elseif(APPLE)
    set(TORCH_URL https://files.pythonhosted.org/packages/7b/91/89bbe2316b93671b6bccec094df6bc66109cf6d21a364cd2f1becd11ba3c/torch-${TORCH_VERSION}-cp39-none-macosx_11_0_arm64.whl)
    set(TORCH_LIB "${CMAKE_SOURCE_DIR}/dorado/3rdparty/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/torch")
elseif(WIN32)
    set(TORCH_URL https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-${TORCH_VERSION}%2Bcu113.zip)
    set(TORCH_LIB "${CMAKE_SOURCE_DIR}/dorado/3rdparty/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/libtorch")
    add_compile_options(
        # Note we need to use the generator expression to avoid setting this for CUDA.
        $<$<COMPILE_LANGUAGE:CXX>:/wd4624> # from libtorch: destructor was implicitly defined as deleted 
    )
endif()

if(DEFINED DORADO_LIBTORCH_DIR)
    # Use the existing libtorch we have been pointed at
    list(APPEND CMAKE_PREFIX_PATH ${DORADO_LIBTORCH_DIR})
    message(STATUS "Using existing libtorch at ${DORADO_LIBTORCH_DIR}")
    set(TORCH_LIB ${DORADO_LIBTORCH_DIR})
else()
    # Get libtorch (if we don't already have it)
    download_and_extract(${TORCH_URL} torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME})
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_LIB}")
endif()

find_package(Torch REQUIRED)
