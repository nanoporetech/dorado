# Guard against double-inclusion errors. See https://github.com/pytorch/pytorch/issues/25004
include_guard(GLOBAL)

set(TORCH_VERSION 2.7.1)
unset(TORCH_PATCH_SUFFIX)

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
            set(TORCH_URL ${DORADO_CDN_URL}/torch-2.7.1.0-ont-CUDA-12.6-linux-aarch64.zip)
            set(TORCH_PATCH_SUFFIX .0-ont-CUDA-12.6)
            set(TORCH_HASH "26fbe0e0b876a68d5f0e78bfee84fb9ec6f57bc69bbcc72d7212edfadef91d31")
            set(TORCH_LIB_SUFFIX "libtorch")
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
            set(TORCH_URL ${DORADO_CDN_URL}/torch-2.7.1.0-ont-CUDA-12.8-linux-x64-cxx11-abi.zip)
            set(TORCH_PATCH_SUFFIX .0-ont-CUDA-12.8)
            set(TORCH_HASH "4bfb80621057614cde1a28daeda70b50f84a2b4a5cca5d39340911612abdf96b")
            set(TORCH_LIB_SUFFIX "libtorch")
        else()
            set(TORCH_URL ${DORADO_CDN_URL}/torch-2.7.1.0-ont-CUDA-11.8-linux-x64-cxx11-abi.zip)
            set(TORCH_PATCH_SUFFIX .0-ont-CUDA-11.8)
            set(TORCH_HASH "6fbb32feb5311ffbb8d540118a099bae61ec9693b994d24687a9748b8d658e78")
            set(TORCH_LIB_SUFFIX "libtorch")  
        endif()
    elseif(APPLE)
        set(TORCH_URL ${DORADO_CDN_URL}/torch-2.7.1.0-ont-macos-m1.zip)
        set(TORCH_PATCH_SUFFIX .0-ont)
        set(TORCH_HASH "819251d91b47b22f368461dceeafbfc1ea0f651eb5f8719d74beab585d876d52")
        set(TORCH_LIB_SUFFIX "libtorch")
    elseif(WIN32)
        set(TORCH_URL ${DORADO_CDN_URL}/torch-2.7.1.1-ont-windows.zip)
        set(TORCH_PATCH_SUFFIX .1-ont)
        set(TORCH_HASH "40929f175c34b1282b8700ccf88cf4499bc7797b499da7ff602637b67bb9ee20")
        set(TORCH_LIB_SUFFIX "libtorch")
    endif()

    # Get libtorch (if we don't already have it)
    set(TORCH_LIB_DIR torch-${TORCH_VERSION}${TORCH_PATCH_SUFFIX}-${CMAKE_SYSTEM_NAME})
    download_and_extract(${TORCH_URL} ${TORCH_LIB_DIR} ${TORCH_HASH})
    set(TORCH_LIB "${DORADO_3RD_PARTY_DOWNLOAD}/${TORCH_LIB_DIR}/${TORCH_LIB_SUFFIX}")
endif()

# Our libtorch should be chosen over any others on the system
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_LIB}")

set(CAFFE2_STATIC_LINK_CUDA ON)
find_package(Torch REQUIRED)

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

if (TORCH_VERSION VERSION_EQUAL 2.7.1 AND (LINUX OR WIN32))
    # For some reason cublas is missing in non-static Linux builds, so do that here.
    list(APPEND TORCH_LIBRARIES CUDA::cublas)
endif()

# Create the target which other libraries can link to
add_library(torch_lib INTERFACE)
target_link_libraries(torch_lib INTERFACE ${TORCH_LIBRARIES})
target_include_directories(torch_lib SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})

if (WIN32)
    # Note we need to use the generator expression to avoid setting this for CUDA.
    target_compile_options(torch_lib INTERFACE
        # from libtorch: destructor was implicitly defined as deleted
        $<$<COMPILE_LANGUAGE:CXX>:/wd4624>
        # from libtorch: structure was padded due to alignment specifier
        $<$<COMPILE_LANGUAGE:CXX>:/wd4324>
        # from libtorch: possible loss of data
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        # from libtorch: 'initializing': conversion from 'X' to 'Y', possible loss of data
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        # Unreachable code warnings are emitted from Torch's Optional class, even though they should be disabled by the
        # MSVC /external:W0 setting.  This is a limitation of /external: for some C47XX backend warnings.  See:
        # https://learn.microsoft.com/en-us/cpp/build/reference/external-external-headers-diagnostics?view=msvc-170#limitations
        $<$<COMPILE_LANGUAGE:CXX>:/wd4702>
    )
    target_compile_definitions(torch_lib INTERFACE
        # Torch Windows static build includes some win headers without defining NOMINMAX.
        NOMINMAX
    )
endif()
