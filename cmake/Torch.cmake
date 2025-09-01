# Guard against double-inclusion errors. See https://github.com/pytorch/pytorch/issues/25004
include_guard(GLOBAL)

set(TORCH_VERSION 2.7.1)
unset(TORCH_PATCH_SUFFIX)

if(ECM_ENABLE_SANITIZERS)
    set(TRY_USING_STATIC_TORCH_LIB FALSE)
else()
    set(TRY_USING_STATIC_TORCH_LIB TRUE)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR WIN32)
    find_package(CUDAToolkit REQUIRED)
    # the torch cuda.cmake will set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}") [2]
    # so we need to make CUDA_TOOLKIT_ROOT_DIR is set correctly as per [1]
    # 1. https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
    # 2. https://github.com/pytorch/pytorch/blob/5fa71207222620b4efb78989849525d4ee6032e8/cmake/public/cuda.cmake#L40
    if(DEFINED CUDAToolkit_ROOT)
      set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_ROOT})
    else()
        # Bodge for Torch, since static linking assumes find_package(CUDA) has already been called
        find_package(CUDA REQUIRED)
    endif()
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
      if(DEFINED CUDAToolkit_ROOT)
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
      else()
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
      endif()
    endif()

    # use python3 to compute shorthash for libnvrtc.so
    # https://github.com/pytorch/pytorch/blob/7289d22d6749465d3bae2cb5a6ce04729318f55b/cmake/public/cuda.cmake#L173
    find_package(Python3 COMPONENTS "Interpreter" REQUIRED)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

    set(CUDNN_LIBRARY_PATH ${DORADO_3RD_PARTY_SOURCE}/fake_cudnn/libcudnn.a)
    set(CUDNN_INCLUDE_PATH ${DORADO_3RD_PARTY_SOURCE}/fake_cudnn)

    set(CMAKE_CUDA_ARCHITECTURES 62 70 72 75)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.3)
        list(APPEND CMAKE_CUDA_ARCHITECTURES 80 86)
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.4)
      list(APPEND CMAKE_CUDA_ARCHITECTURES 87)
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
    endif()

    # Versions of nvcc before CUDA 12.x don't support CUDA C++20 as a standard.
    if (CUDAToolkit_VERSION VERSION_LESS 12.0)
        set(CMAKE_CUDA_STANDARD 17)
    endif()
endif()

if(DEFINED DORADO_LIBTORCH_DIR)
    # Use the existing libtorch we have been pointed at
    message(STATUS "Using existing libtorch at ${DORADO_LIBTORCH_DIR}")
    set(TORCH_LIB ${DORADO_LIBTORCH_DIR})
else()
    # Otherwise download a new one
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
            set(TORCH_URL ${DORADO_CDN_URL}/torch-2.7.1-linux-aarch64-ont.zip)
            set(TORCH_PATCH_SUFFIX -ont)
            set(TORCH_HASH "7e741501d7c8b050d3de853c31f79e91f6eb7ba370694431029f3c7dbba69ad3")
            set(TORCH_LIB_SUFFIX "/libtorch")
            set(USING_STATIC_TORCH_LIB TRUE)
        else()
            set(TORCH_URL https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu126.zip)
            set(TORCH_PATCH_SUFFIX -cxx11-abi)
            set(TORCH_HASH "15708d647d720eb703994f022488bca9ae29a07cf19e76e8b218d0a07be2a943")
            set(TORCH_LIB_SUFFIX "/libtorch")
        endif()

    elseif(APPLE)
        # Taken from https://pypi.org/project/torch/#files
        set(TORCH_URL https://files.pythonhosted.org/packages/b3/17/41f681b87290a1d2f1394f943e470f8b0b3c2987b7df8dc078d8831fce5b/torch-${TORCH_VERSION}-cp39-none-macosx_11_0_arm64.whl)
        set(TORCH_HASH "265f70de5fd45b864d924b64be1797f86e76c8e48a02c2a3a6fc7ec247d2226c")
        set(TORCH_LIB_SUFFIX "/torch")
    elseif(WIN32)
        set(TORCH_URL https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-${TORCH_VERSION}%2Bcu126.zip)
        set(TORCH_HASH "89ed2ae468555487ad153bf6f1b0bcce17814da314ba14996c4d63602e94c8c9")
        set(TORCH_LIB_SUFFIX "/libtorch")
    endif()

    # Get libtorch (if we don't already have it)
    set(TORCH_LIB_DIR torch-${TORCH_VERSION}${TORCH_PATCH_SUFFIX}-${CMAKE_SYSTEM_NAME})
    download_and_extract(${TORCH_URL} ${TORCH_LIB_DIR} ${TORCH_HASH})
    set(TORCH_LIB "${DORADO_3RD_PARTY_DOWNLOAD}/${TORCH_LIB_DIR}/${TORCH_LIB_SUFFIX}")
endif()

# Our libtorch should be chosen over any others on the system
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_LIB}")

find_package(Torch REQUIRED)

if(APPLE)
    set(TORCH_BUILD_VERSION ${TORCH_VERSION})
else()
    if(EXISTS "${TORCH_LIB}/build-version")
        file(STRINGS "${TORCH_LIB}/build-version" TORCH_BUILD_VERSION)
    else()
        set(PYTORCH_BUILD_VERSION "import torch; print('%s+cu%s' % (torch.__version__, torch.version.cuda.replace('.', '')), end='')")
        execute_process(
            COMMAND python3 -c "${PYTORCH_BUILD_VERSION}"
            OUTPUT_VARIABLE TORCH_BUILD_VERSION
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()
    message(STATUS "TORCH_BUILD_VERSION: ${TORCH_BUILD_VERSION}")
endif()

if (NOT TORCH_BUILD_VERSION VERSION_EQUAL TORCH_VERSION)
  message(WARNING "expected ${TORCH_VERSION} but found ${TORCH_BUILD_VERSION}")
endif()

if(WIN32 AND DEFINED MKL_ROOT)
    link_directories(${MKL_ROOT}/lib/intel64)
endif()

# Add missing frameworks
if (APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(MPSG_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
    list(APPEND TORCH_LIBRARIES
        ${ACCELERATE_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${METAL_FRAMEWORK}
        ${MPS_FRAMEWORK}
        ${MPSG_FRAMEWORK}
    )
endif()

if (TORCH_VERSION VERSION_EQUAL 2.7.1 AND LINUX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    # For some reason cublas is missing in non-static Linux builds (ie sanitized), so do that here.
    list(APPEND TORCH_LIBRARIES CUDA::cublas)
endif()

# Create the target which other libraries can link to
add_library(torch_lib INTERFACE)
target_link_libraries(torch_lib INTERFACE ${TORCH_LIBRARIES})
target_include_directories(torch_lib SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
