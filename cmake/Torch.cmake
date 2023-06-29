# Guard against double-inclusion errors. See https://github.com/pytorch/pytorch/issues/25004
include_guard(GLOBAL)

set(TORCH_VERSION 2.0.0)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR WIN32)
    find_package(CUDAToolkit REQUIRED)
    # the torch cuda.cmake will set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}") [2]
    # so we need to make CUDA_TOOLKIT_ROOT_DIR is set correctly as per [1]
    # 1. https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
    # 2. https://github.com/pytorch/pytorch/blob/5fa71207222620b4efb78989849525d4ee6032e8/cmake/public/cuda.cmake#L40
    if(DEFINED CUDAToolkit_ROOT)
      set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_ROOT})
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

    set(CUDNN_LIBRARY_PATH ${DORADO_3RD_PARTY}/fake_cudnn/libcudnn.a)
    set(CUDNN_INCLUDE_PATH ${DORADO_3RD_PARTY}/fake_cudnn)

    set(CMAKE_CUDA_ARCHITECTURES 62 70 72 75)
    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 11.3)
        list(APPEND CMAKE_CUDA_ARCHITECTURES 80 86)
    endif()
    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 11.4)
      list(APPEND CMAKE_CUDA_ARCHITECTURES 87)
    endif()
    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 11.8)
      list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
    endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
        if(${CUDAToolkit_VERSION} VERSION_LESS 11.0)
            set(TORCH_VERSION 1.10.2)
            set(TORCH_URL https://cdn.oxfordnanoportal.com/software/analysis/torch-1.10.2-Linux-aarch64.zip)
            set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/torch")
        else()
            set(TORCH_URL https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.09-cp38-cp38-linux_aarch64.whl)
            set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/torch")
        endif()
    else()
        if(DORADO_USING_OLD_CPP_ABI)
            set(TORCH_URL https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip)

            set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-no-cxx11-abi-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/libtorch")
        else()
            set(TORCH_URL https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip)

            set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/libtorch")
        endif()
    endif()

elseif(APPLE)
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(TORCH_URL https://nanoporetech.box.com/shared/static/nzdq2wk45pzbwi2zex92j28dt3s5k9vt.tgz)
        set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(TORCH_URL https://download.pytorch.org/whl/cpu/torch-${TORCH_VERSION}-cp39-none-macosx_10_9_x86_64.whl)
        set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/torch")
    else()
        set(TORCH_URL https://files.pythonhosted.org/packages/4d/80/760f3edcf0179c3111fae496b97ee3fa9171116b4bccae6e073efe928e72/torch-${TORCH_VERSION}-cp39-none-macosx_11_0_arm64.whl)
        set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/torch")
    endif()
elseif(WIN32)
    set(TORCH_URL https://cdn.oxfordnanoportal.com/software/analysis/torch-2.0.0.2-Windows-ont.zip)
    set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/libtorch")
    add_compile_options(
        # Note we need to use the generator expression to avoid setting this for CUDA.
        $<$<COMPILE_LANGUAGE:CXX>:/wd4624> # from libtorch: destructor was implicitly defined as deleted 
    )

    # Torch Windows static build includes some win headers without defining NOMINMAX.
    add_compile_options(-DNOMINMAX)
endif()

if(DEFINED DORADO_LIBTORCH_DIR)
    # Use the existing libtorch we have been pointed at
    list(APPEND CMAKE_PREFIX_PATH ${DORADO_LIBTORCH_DIR})
    message(STATUS "Using existing libtorch at ${DORADO_LIBTORCH_DIR}")
    set(TORCH_LIB ${DORADO_LIBTORCH_DIR})
else()
    # Get libtorch (if we don't already have it)
    if(DORADO_USING_OLD_CPP_ABI AND NOT WIN32 AND NOT APPLE)
        download_and_extract(${TORCH_URL} torch-no-cxx11-abi-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME})
    else()
        download_and_extract(${TORCH_URL} torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME})
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_LIB}")
endif()

if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    # For some reason the RPATHs of the dylibs are pointing to the libiomp5.dylib in functools rather
    # than the one that's next to them, so correct that here before we import the package.
    file(GLOB TORCH_DYLIBS "${TORCH_LIB}/lib/*.dylib")
    foreach(TORCH_DYLIB IN LISTS TORCH_DYLIBS)
        execute_process(
            COMMAND
                ${CMAKE_INSTALL_NAME_TOOL}
                -change "@loader_path/../../functorch/.dylibs/libiomp5.dylib" "@loader_path/libiomp5.dylib"
                ${TORCH_DYLIB}
            RESULT_VARIABLE RETVAL
            OUTPUT_VARIABLE OUTPUT
            ERROR_VARIABLE ERRORS
        )
        if (NOT RETVAL EQUAL 0)
            message(FATAL_ERROR "Error running ${CMAKE_INSTALL_NAME_TOOL}: ${RETVAL}\nOUTPUT=${OUTPUT}\nERRORS=${ERRORS}")
        endif()
    endforeach()
endif()

find_package(Torch REQUIRED)

if(APPLE)
    set(TORCH_BUILD_VERSION ${TORCH_VERSION})
else()
    if(EXISTS "${TORCH_LIB}/build-version")
        file(STRINGS "${TORCH_LIB}/build-version" TORCH_BUILD_VERSION)
    else()
        set(PYTORCH_BUILD_VERSION "import torch; print('%s+cu%s' % (torch.__version__, torch.version.cuda.replace('.', '')), end='')")
        execute_process(COMMAND python3 -c "${PYTORCH_BUILD_VERSION}" OUTPUT_VARIABLE TORCH_BUILD_VERSION WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()
    message(STATUS "TORCH_BUILD_VERSION: ${TORCH_BUILD_VERSION}")
endif()

if (NOT TORCH_BUILD_VERSION VERSION_EQUAL TORCH_VERSION)
  message(WARNING "expected ${TORCH_VERSION} but found ${TORCH_BUILD_VERSION}")
endif()

if(WIN32 AND DEFINED MKL_ROOT)
    link_directories(${MKL_ROOT}/lib/intel64)
endif()
