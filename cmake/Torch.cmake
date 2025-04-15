# Guard against double-inclusion errors. See https://github.com/pytorch/pytorch/issues/25004
include_guard(GLOBAL)

set(TORCH_VERSION 2.6.0)
unset(TORCH_PATCH_SUFFIX)

if (NOT DEFINED TRY_USING_STATIC_TORCH_LIB)
    # If we're building with sanitizers then we want Torch to be dynamic since we don't build the
    # static lib with instrumentation and hence get some false-positives.
    if(ECM_ENABLE_SANITIZERS)
        set(TRY_USING_STATIC_TORCH_LIB FALSE)
    else()
        set(TRY_USING_STATIC_TORCH_LIB TRUE)
    endif()
endif()
set(USING_STATIC_TORCH_LIB FALSE)

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

if(DEFINED DORADO_LIBTORCH_DIR)
    # Use the existing libtorch we have been pointed at
    message(STATUS "Using existing libtorch at ${DORADO_LIBTORCH_DIR}")
    set(TORCH_LIB ${DORADO_LIBTORCH_DIR})
else()
    # Otherwise download a new one
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
            if(${CUDAToolkit_VERSION} VERSION_LESS 11.0)
                set(TORCH_VERSION 1.10.2)
                if (TRY_USING_STATIC_TORCH_LIB)
                    set(TORCH_URL ${DORADO_CDN_URL}/torch-1.10.2.1-linux-aarch64-ont-static.zip)
                    set(TORCH_PATCH_SUFFIX -ont.1)
                    set(TORCH_HASH "a5b2927dcadf4dc0a258aa7138ca52a9c120dd16322191b748151c5c2987b52c")
                    set(TORCH_LIB_SUFFIX "/libtorch")
                    set(USING_STATIC_TORCH_LIB TRUE)
                else()
                    set(TORCH_URL ${DORADO_CDN_URL}/torch-1.10.2-Linux-aarch64.zip)
                    set(TORCH_PATCH_SUFFIX -ont)
                    set(TORCH_HASH "9b5b111986dd54b7d25f9f9382e56b06423bb18ae3e2c0a19de6d74949d9c06a")
                    set(TORCH_LIB_SUFFIX "/torch")
                endif()
            elseif(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 12.0)
                if (TRY_USING_STATIC_TORCH_LIB)
                    set(TORCH_URL ${DORADO_CDN_URL}/torch-2.6.0-linux-aarch64-ont.zip)
                    set(TORCH_PATCH_SUFFIX -ont)
                    set(TORCH_HASH "7e741501d7c8b050d3de853c31f79e91f6eb7ba370694431029f3c7dbba69ad3")
                    set(TORCH_LIB_SUFFIX "/libtorch")
                    set(USING_STATIC_TORCH_LIB TRUE)
                else()
                    message(FATAL_ERROR "CUDA-12 builds on aarch64 only support static torch")
                endif()
            else()
                if (TRY_USING_STATIC_TORCH_LIB)
                    set(TORCH_URL ${DORADO_CDN_URL}/torch-2.0.0.2-linux-aarch64-ont.zip)
                    set(TORCH_PATCH_SUFFIX -ont.2)
                    set(TORCH_HASH "90128c2921e96d8fa8c2ef9853f329d810421ef0fbeb04fcbdab9047bac0f440")
                    set(TORCH_LIB_SUFFIX "/libtorch")
                    set(USING_STATIC_TORCH_LIB TRUE)
                else()
                    # Grab from NVidia rather than pytorch so that it has the magic NVidia sauce
                    set(TORCH_VERSION 1.13.0)
                    set(TORCH_URL https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.09-cp38-cp38-linux_aarch64.whl)
                    set(TORCH_HASH "ddbdf57c089cd0036289e3e43fb7126ce98980721fd14be7a09240948a5f1fa9")
                    set(TORCH_LIB_SUFFIX "/torch")
                endif()
            endif()
        else()
            if (TRY_USING_STATIC_TORCH_LIB)
                if(DORADO_USING_OLD_CPP_ABI)
                    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 12.8)
                        set(TORCH_URL ${DORADO_CDN_URL}/torch-${TORCH_VERSION}.2-ont-CUDA-12.8-linux-x64-pre-cxx11.zip)
                        set(TORCH_HASH "ce1b9fc2f829b5bf4e4761d69f2cdf58fc64c5df604cd80311e68e3e9cef7df6")
                    else()
                        set(TORCH_URL ${DORADO_CDN_URL}/torch-${TORCH_VERSION}.2-ont-CUDA-11.8-linux-x64-pre-cxx11.zip)
                        set(TORCH_HASH "9104e8c8ed8cf313fcd97b038311dba74c77e3a508262949a5d83c7971d71dbe")
                    endif()
                    set(TORCH_PATCH_SUFFIX -ont-pre-cxx11)
                else()
                    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 12.8)
                        set(TORCH_URL ${DORADO_CDN_URL}/torch-${TORCH_VERSION}.2-ont-CUDA-12.8-linux-x64-cxx11-abi.zip)
                        set(TORCH_HASH "75bb7dd8c6e9d96261338b53ead12aa492d8d43f9abb724bd484a1150956db9f")
                    else()
                        set(TORCH_URL ${DORADO_CDN_URL}/torch-${TORCH_VERSION}.2-ont-CUDA-11.8-linux-x64-cxx11-abi.zip)
                        set(TORCH_HASH "b66e93595b20587d63245c6a42d207eccf933fb48bae82c53eee6eff247bb1b9")
                    endif()
                    set(TORCH_PATCH_SUFFIX -ont-cxx11-abi)
                endif()
                set(USING_STATIC_TORCH_LIB TRUE)
            else()
                if(DORADO_USING_OLD_CPP_ABI)
                    set(TORCH_URL https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip)
                    set(TORCH_PATCH_SUFFIX -pre-cxx11)
                    set(TORCH_HASH "b2af1a32e7fa8bc39c24bdcdf374c3e8cc3439efc6ec9319fcfb34c395a30501")
                else()
                    set(TORCH_URL https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip)
                    set(TORCH_PATCH_SUFFIX -cxx11-abi)
                    set(TORCH_HASH "36835d6c6315d741ad687632516f7bcd8efb6de3b57b61ca66b96f98e5ea30e8")
                endif()
            endif()
            set(TORCH_LIB_SUFFIX "/libtorch")
        endif()

    elseif(APPLE)
        if (TRY_USING_STATIC_TORCH_LIB)
            set(TORCH_URL ${DORADO_CDN_URL}/torch-${TORCH_VERSION}.1-macos-m1-ont.zip)
            set(TORCH_PATCH_SUFFIX -ont.1)
            set(TORCH_HASH "37dd00a4d1137ca890f50afb0d9a930e3153fc68df0cbec42ba043cc202ad089")
            set(TORCH_LIB_SUFFIX "/libtorch")
            set(USING_STATIC_TORCH_LIB TRUE)
        else()
            # Taken from https://pypi.org/project/torch/#files
            set(TORCH_URL https://files.pythonhosted.org/packages/b3/17/41f681b87290a1d2f1394f943e470f8b0b3c2987b7df8dc078d8831fce5b/torch-${TORCH_VERSION}-cp39-none-macosx_11_0_arm64.whl)
            set(TORCH_HASH "265f70de5fd45b864d924b64be1797f86e76c8e48a02c2a3a6fc7ec247d2226c")
            set(TORCH_LIB_SUFFIX "/torch")
        endif()
    elseif(WIN32)
        if (TRY_USING_STATIC_TORCH_LIB)
            set(TORCH_URL ${DORADO_CDN_URL}/torch-${TORCH_VERSION}.2-ont-windows.zip)
            set(TORCH_PATCH_SUFFIX -ont.2)
            set(TORCH_HASH "4b2cebb0b7d195028d7f82e7a906201554ef6d3813909e976f6263526b0c537b")
            set(TORCH_LIB_SUFFIX "/libtorch")
            set(USING_STATIC_TORCH_LIB TRUE)
        else()
            set(TORCH_URL https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-${TORCH_VERSION}%2Bcu126.zip)
            set(TORCH_HASH "89ed2ae468555487ad153bf6f1b0bcce17814da314ba14996c4d63602e94c8c9")
            set(TORCH_LIB_SUFFIX "/libtorch")
        endif()
    endif()

    if (USING_STATIC_TORCH_LIB)
        set(TORCH_PATCH_SUFFIX "${TORCH_PATCH_SUFFIX}-static")
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

# Static builds require a few libs to be added
if (USING_STATIC_TORCH_LIB)
    if(WIN32)
        list(APPEND TORCH_LIBRARIES
            CUDA::cudart_static
            CUDA::cublas
            CUDA::cufft
            CUDA::cusolver
            CUDA::cusparse
        )

    elseif(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
        list(APPEND TORCH_LIBRARIES
            ${ACCELERATE_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
        )
        find_library(METAL_FRAMEWORK Metal REQUIRED)
        find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
        find_library(MPSG_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
        list(APPEND TORCH_LIBRARIES
            ${METAL_FRAMEWORK}
            ${MPS_FRAMEWORK}
            ${MPSG_FRAMEWORK}
        )

    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ${CUDAToolkit_VERSION} VERSION_LESS 11.0)
        list(APPEND TORCH_LIBRARIES
            # Missing libs that Torch forgets to link to
            ${TORCH_LIB}/lib/libbreakpad.a
            ${TORCH_LIB}/lib/libbreakpad_common.a

            # Some of the CUDA libs have inter-dependencies, so group them together
            $<LINK_GROUP:RESCAN,
                CUDA::cudart_static,
                CUDA::cublas_static,
                CUDA::cublasLt_static,
                # AFAICT Torch doesn't provide the symbol required for the callback, so use the nocallback variant
                CUDA::cufft_static_nocallback,
                CUDA::cusolver_static,
                # cusolver is missing this and I don't know why
                ${CUDAToolkit_TARGET_DIR}/lib64/liblapack_static.a,
                CUDA::cusparse_static,
                CUDA::cupti,
                CUDA::curand_static,
                CUDA::nvrtc,
                CUDA::culibos
            >
            # OMP implementation (i=Intel, g=GNU)
            ${TORCH_LIB}/lib/libgomp.so.1.0.0
            # BLAS rather than MKL
            ${TORCH_LIB}/lib/libopenblas.a
            ${TORCH_LIB}/lib/libgfortran.so.4.0.0
        )

    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Some CUDA lib symbols have internal linkage, so they must be part of the helper lib too
        set(ont_cuda_internal_linkage_libs CUDA::culibos CUDA::cudart_static)
        if (TARGET CUDA::cupti_static)
            list(APPEND ont_cuda_internal_linkage_libs CUDA::cupti_static)
        elseif(TARGET CUDA::cupti)
            # CUDA::cupti appears to be static if CUDA::cupti_static doesn't exist
            list(APPEND ont_cuda_internal_linkage_libs CUDA::cupti)
        elseif(EXISTS ${CUDAToolkit_TARGET_DIR}/extras/CUPTI/lib64/libcupti_static.a)
            # CMake sometimes can't find cupti for reasons which are not fully clear
            list(APPEND ont_cuda_internal_linkage_libs ${CUDAToolkit_TARGET_DIR}/extras/CUPTI/lib64/libcupti_static.a)
        else()
            message(FATAL_ERROR "Can't find CUPTI")
        endif()

        # Setup differences between platforms
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            if (TORCH_VERSION VERSION_LESS 2.6)
                list(APPEND TORCH_LIBRARIES
                    # These 2 libs depend on each other, but only libdnnl.a is added to Torch's install cmake, so we
                    # need to add it again after bringing in libdnnl_graph.a to fill in the missing symbols.
                    ${TORCH_LIB}/lib/libdnnl_graph.a
                    ${TORCH_LIB}/lib/libdnnl.a
                )
                set(ont_torch_extra_platform_libs
                    ${TORCH_LIB}/lib/libnccl_static.a
                    ${TORCH_LIB}/lib/libiomp5.so
                )
            else()
                list(APPEND TORCH_LIBRARIES
                    ${TORCH_LIB}/lib/libdnnl.a
                )
                set(ont_torch_extra_platform_libs
                    ${TORCH_LIB}/lib/libnccl_static.a
                    ${TORCH_LIB}/lib/libcusparseLt_static.a
                    ${TORCH_LIB}/lib/libiomp5.a
                )
            endif()
            set(ont_torch_extra_cuda_libs
                # I don't know why the MKL libs need to be part of the CUDA group, but having them in a
                # separate group causes missing symbol errors
                ${TORCH_LIB}/lib/libmkl_core.a
                ${TORCH_LIB}/lib/libmkl_intel_lp64.a
                ${TORCH_LIB}/lib/libmkl_intel_thread.a
            )
        else()
            set(ont_torch_extra_cuda_libs
                CUDA::curand_static
                CUDA::nvrtc
            )
            if(CUDAToolkit_VERSION VERSION_LESS 12.6)
                # cusolver is missing this and I don't know why
                list(APPEND ont_torch_extra_cuda_libs
                    ${CUDAToolkit_TARGET_DIR}/lib64/liblapack_static.a
                )
            endif()
            set(ont_torch_extra_platform_libs
                ${TORCH_LIB}/lib/libopenblas.a
                ${TORCH_LIB}/lib/libgfortran.so.5
                ${TORCH_LIB}/lib/libgomp.so.1.0.0
                numa
            )
        endif()

        # Link to the cuDNN libs
        if (TORCH_VERSION VERSION_GREATER_EQUAL 2.6)
            # a second helper library due to more relocation errors
            add_library(dorado_cudnn_lib SHARED
                ${CMAKE_CURRENT_LIST_DIR}/../dorado/cudnn_dummy.cpp
            )
            if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
                # These appear to be linked in elsewhere.
                set(caffe2_perfkernels)
            else()
                set(caffe2_perfkernels ${TORCH_LIB}/lib/libCaffe2_perfkernels_sve.a)
            endif()
            target_link_libraries(dorado_cudnn_lib PRIVATE
                # Note: libtorch is still setup to link to these dynamically (https://github.com/pytorch/pytorch/issues/81692)
                # though that shouldn't be a problem on Linux
                $<LINK_LIBRARY:WHOLE_ARCHIVE,
                    ${caffe2_perfkernels}
                    ${TORCH_LIB}/lib/libcudnn_adv_static_v9.a
                    ${TORCH_LIB}/lib/libcudnn_cnn_static_v9.a
                    ${TORCH_LIB}/lib/libcudnn_ops_static_v9.a
                    ${TORCH_LIB}/lib/libcudnn_graph_static_v9.a
                    ${TORCH_LIB}/lib/libcudnn_heuristic_static_v9.a
                    ${TORCH_LIB}/lib/libcudnn_engines_precompiled_static_v9.a
                    ${TORCH_LIB}/lib/libcudnn_engines_runtime_compiled_static_v9.a
                >
                CUDA::cudart_static
            )
            if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
                # CUDA::nvrtc_static depends on CUDA::nvrtc_builtins_static which depends on
                # CUDA::cuda_driver which we shouldn't be linking to. Depending on the OS and
                # version of LD, sometimes the linker is smart enough to realise that the link
                # to the driver is unnecessary and drops the dependency, but other times it
                # doesn't and we end up with builds that require CUDA to be installed.
                # Instead find the libs and manage the dependencies ourselves.
                # Workaround taken from here: https://discourse.cmake.org/t/cmake-incorrectly-links-to-nvrtc-builtins/12723
                find_library(ont_nvrtc_lib nvrtc_static PATHS "${CUDAToolkit_LIBRARY_DIR}" REQUIRED)
                find_library(ont_nvrtc_builtins_lib nvrtc-builtins_static PATHS "${CUDAToolkit_LIBRARY_DIR}" REQUIRED)
                find_library(ont_nvptxcompiler_lib nvptxcompiler_static PATHS "${CUDAToolkit_LIBRARY_DIR}" REQUIRED)
                target_link_libraries(dorado_cudnn_lib PRIVATE ${ont_nvrtc_lib} ${ont_nvrtc_builtins_lib} ${ont_nvptxcompiler_lib})
            else()
                target_link_libraries(dorado_cudnn_lib PRIVATE CUDA::nvrtc)
            endif()
            list(APPEND TORCH_LIBRARIES dorado_cudnn_lib)
            # Don't forget to install it
            install(TARGETS dorado_cudnn_lib LIBRARY)
        else()
            list(APPEND TORCH_LIBRARIES
                # Note: the order of the cuDNN libs matter
                # We aren't going to do any training, so these don't need to be whole-archived
                ${TORCH_LIB}/lib/libcudnn_adv_train_static.a
                ${TORCH_LIB}/lib/libcudnn_cnn_train_static.a
                ${TORCH_LIB}/lib/libcudnn_ops_train_static.a
                # I'm assuming we need this for https://github.com/pytorch/pytorch/issues/50153
                $<LINK_LIBRARY:WHOLE_ARCHIVE,
                    # Note: libtorch is still setup to link to these dynamically (https://github.com/pytorch/pytorch/issues/81692)
                    # though that shouldn't be a problem on Linux
                    ${TORCH_LIB}/lib/libcudnn_adv_infer_static.a
                    ${TORCH_LIB}/lib/libcudnn_cnn_infer_static.a
                    ${TORCH_LIB}/lib/libcudnn_ops_infer_static.a
                >
            )
        endif()

        # Currently we need to make use of a separate lib to avoid getting relocation errors at link time
        # because the final binary would end up too big.
        # See https://github.com/pytorch/pytorch/issues/39968
        set(USE_TORCH_HELPER_LIB TRUE)
        if (USE_TORCH_HELPER_LIB)
            add_library(dorado_torch_lib SHARED
                # We need to use listdir here so projects including us use the correct path
                ${CMAKE_CURRENT_LIST_DIR}/../dorado/torch_half.cpp
            )
            target_link_libraries(dorado_torch_lib PRIVATE
                ${TORCH_LIBRARIES}
                # Some of the CUDA libs have inter-dependencies, so group them together
                $<LINK_GROUP:RESCAN,
                    ${ont_cuda_internal_linkage_libs}
                >
            )
            target_include_directories(dorado_torch_lib PUBLIC
                ${TORCH_INCLUDE_DIRS}
            )
            # Replace the torch libs with the helper lib
            set(TORCH_LIBRARIES dorado_torch_lib)
            # Don't forget to install it
            install(TARGETS dorado_torch_lib LIBRARY)
        endif()

        list(APPEND TORCH_LIBRARIES
            # Some of the CUDA libs have inter-dependencies, so group them together
            $<LINK_GROUP:RESCAN,
                CUDA::cudart_static,
                CUDA::cublas_static,
                CUDA::cublasLt_static,
                CUDA::cufft_static_nocallback,
                CUDA::cusolver_static,
                CUDA::cusparse_static,
                ${ont_cuda_internal_linkage_libs},
                ${ont_torch_extra_cuda_libs}
            >
            ${ont_torch_extra_platform_libs}
        )

        if (${CMAKE_VERSION} VERSION_LESS 3.23.4 AND EXISTS ${CUDAToolkit_TARGET_DIR}/lib64/libcusolver_lapack_static.a)
            # CUDA::cusolver_static is missing the cusolver_lapack_static target+dependency in older versions of cmake
            list(APPEND TORCH_LIBRARIES
                ${CUDAToolkit_TARGET_DIR}/lib64/libcusolver_lapack_static.a
            )
        endif()
    endif()
elseif (TORCH_VERSION VERSION_EQUAL 2.6 AND LINUX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    # For some reason cublas is missing in non-static Linux builds (ie sanitized), so do that here.
    list(APPEND TORCH_LIBRARIES ${TORCH_LIB}/lib/libcublas-3b81d170.so.11)
endif()

# Create the target which other libraries can link to
add_library(torch_lib INTERFACE)
target_link_libraries(torch_lib INTERFACE ${TORCH_LIBRARIES})
target_include_directories(torch_lib SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})

if (WIN32 AND USING_STATIC_TORCH_LIB)
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
elseif (CMAKE_COMPILER_IS_GNUCXX AND TORCH_VERSION VERSION_EQUAL 2.6)
    # Don't allow the linker to relax addresses otherwise we get "failed to convert GOTPCREL relocation" errors.
    target_link_options(torch_lib INTERFACE "LINKER:--no-relax")
endif()
