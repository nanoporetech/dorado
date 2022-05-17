cmake_minimum_required(VERSION 3.20)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
project(dorado)
set(DORADO_VERSION "0.0.1a0")

set(MKR_VERSION 0.0.11)
set(TORCH_VERSION 1.10.2)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
option(DYNAMIC_HDF "Link HDF as dynamic libs" OFF)

if(WIN32)
    message(STATUS "toolset: " ${CMAKE_GENERATOR_TOOLSET})
    message(STATUS "platform: " ${CMAKE_GENERATOR_PLATFORM})
    message(STATUS "vs platform: " ${CMAKE_VS_PLATFORM_NAME})
endif()
enable_testing()

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")

        file(LOCK ${CMAKE_SOURCE_DIR} DIRECTORY)
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        file(LOCK ${CMAKE_SOURCE_DIR} DIRECTORY RELEASE)

        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()


# Helper function to extract the specified URL to the given 3rd party folder if it doesn't already exist
set(3RD_PARTY ${CMAKE_SOURCE_DIR}/dorado/3rdparty)
function(download_and_extract url name)
    file(LOCK ${CMAKE_SOURCE_DIR} DIRECTORY)

    if(EXISTS ${3RD_PARTY}/${name})
        message("-- Found ${name}")
    else()
        message("-- Downloading ${name}")
        file(DOWNLOAD ${url} ${3RD_PARTY}/${name}.zip)
        message("-- Downloading ${name} - done")
        message("-- Extracting ${name}")
        file(ARCHIVE_EXTRACT INPUT ${3RD_PARTY}/${name}.zip DESTINATION ${3RD_PARTY}/${name})
        file(REMOVE ${3RD_PARTY}/${name}.zip)
        message("-- Extracting ${name} - done")

    endif()

    file(LOCK ${CMAKE_SOURCE_DIR} DIRECTORY RELEASE)
endfunction()

function(add_hdf_vbz_plugin)
    set(ENABLE_CONAN OFF)
    set(ENABLE_PERF_TESTING OFF)
    set(ENABLE_PYTHON OFF)
    set(ENABLE_PACKAGING OFF)
    set(BUILD_SHARED_LIBS OFF)
    if(WIN32)
        # On windows we need to build a static lib for zstd as there's no prebuilt distro
        execute_process(COMMAND cmake -S dorado/3rdparty/zstd/build/cmake -B dorado/3rdparty/cmake-build-zstd -A x64)
        execute_process(COMMAND cmake --build dorado/3rdparty/cmake-build-zstd --config Release)
        # On windows we need to tell hdf_plugins where we put the built zstd lib
        set(CONAN_INCLUDE_DIRS_RELEASE ${CMAKE_SOURCE_DIR}/dorado/3rdparty/zstd/lib)
        set(CONAN_INCLUDE_DIRS_DEBUG ${CMAKE_SOURCE_DIR}/dorado/3rdparty/zstd/lib)
        set(CONAN_LIB_DIRS_RELEASE ${CMAKE_SOURCE_DIR}/dorado/3rdparty/cmake-build-zstd/lib/Release)
        set(CONAN_LIB_DIRS_DEBUG ${CMAKE_SOURCE_DIR}/dorado/3rdparty/cmake-build-zstd/lib/Debug)
    endif()
   
    add_subdirectory(dorado/3rdparty/hdf_plugins)
endfunction()


if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()


set(MKR_DIR mkr-${MKR_VERSION}-${CMAKE_SYSTEM_NAME})

set(CUDNN_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/dorado/3rdparty/fake_cudnn)
set(CUDNN_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/dorado/3rdparty/fake_cudnn)

if(LINUX)
    set(MKR_URL https://nanoporetech.box.com/shared/static/v589ievqqmq39dnbfuk14drq33xyomov.gz)
    set(MKR_LIBRARIES
      ${3RD_PARTY}/${MKR_DIR}/lib64/libmkr_format.a
      ${3RD_PARTY}/${MKR_DIR}/lib64/libboost_filesystem.a
      ${3RD_PARTY}/${MKR_DIR}/lib64/libarrow.a
    )
    # Note: We can set CUDA_TOOLKIT_ROOT_DIR externally, if it's not installed to the default location as
    #  per the readme instructions
    if(DEFINED CUDA_TOOLKIT_ROOT_DIR)
        message(STATUS "CUDA toolkit dir is ${CUDA_TOOLKIT_ROOT_DIR}")
    endif()
    enable_language(CUDA)
    set(TORCH_URL https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu113.zip)
    set(TORCH_LIB "${CMAKE_SOURCE_DIR}/dorado/3rdparty/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/libtorch")
elseif(APPLE)
    set(MKR_URL https://nanoporetech.box.com/shared/static/t5w6jl1tqazjlpoekbpi7r145qspw8xo.gz)
    set(MKR_LIBRARIES
      ${3RD_PARTY}/${MKR_DIR}/lib/libmkr_format.a
      ${3RD_PARTY}/${MKR_DIR}/lib/libboost_filesystem.a
      ${3RD_PARTY}/${MKR_DIR}/lib/libarrow.a
    )
    download_and_extract(https://developer.apple.com/metal/cpp/files/metal-cpp_macOS12_iOS15.zip metal-cpp)
    find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
    find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
    find_library(APPLE_FWK_METAL Metal REQUIRED)
    set(TORCH_URL https://files.pythonhosted.org/packages/7b/91/89bbe2316b93671b6bccec094df6bc66109cf6d21a364cd2f1becd11ba3c/torch-${TORCH_VERSION}-cp39-none-macosx_11_0_arm64.whl)
    set(TORCH_LIB "${CMAKE_SOURCE_DIR}/dorado/3rdparty/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/torch")

    set(AIR_FILES)
    set(METAL_SOURCES dorado/nn/metal/nn.metal)

    foreach(source ${METAL_SOURCES})
        get_filename_component(basename "${source}" NAME_WE)
        set(air_path "${CMAKE_BINARY_DIR}/${basename}.air")
        add_custom_command(
            OUTPUT "${air_path}"
            COMMAND xcrun -sdk macosx metal -ffast-math -c "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o "${air_path}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
            COMMENT "Compiling metal kernels"
        )
        list(APPEND AIR_FILES "${air_path}")
    endforeach()

    add_custom_command(
        OUTPUT default.metallib
        COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${CMAKE_BINARY_DIR}/lib/default.metallib
        DEPENDS ${AIR_FILES}
        COMMENT "Creating metallib"
    )
elseif(WIN32)
    set(MKR_URL https://nanoporetech.box.com/shared/static/wu75lj0f69lrs24fjvlnkpohkcch85c7.gz)
    set(MKR_LIBRARIES
      ${3RD_PARTY}/${MKR_DIR}/lib/mkr_format.lib
      ${3RD_PARTY}/${MKR_DIR}/lib/libboost_filesystem.lib
      ${3RD_PARTY}/${MKR_DIR}/lib/arrow_static.lib
      bcrypt.lib
    )
    enable_language(CUDA)
    set(TORCH_URL https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-${TORCH_VERSION}%2Bcu113.zip)
    set(TORCH_LIB "${CMAKE_SOURCE_DIR}/dorado/3rdparty/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/libtorch")

    # On windows, we need to build HDF5
    set(HDF_VER hdf5-1.12.1-3)
    if(EXISTS ${3RD_PARTY}/${HDF_VER})
        message("-- Found ${HDF_VER}")
    else()
        # Need a zlib build for HDF to use
        set(ZLIB_VER zlib-1.2.12)
        download_and_extract(https://zlib.net/${ZLIB_VER}.tar.gz ${ZLIB_VER})
        execute_process(COMMAND cmake -S ${3RD_PARTY}/${ZLIB_VER}/${ZLIB_VER} -B ${3RD_PARTY}/${ZLIB_VER}/${ZLIB_VER}/cmake-build -A x64
            -DCMAKE_INSTALL_PREFIX=${3RD_PARTY}/${ZLIB_VER}/install)
        execute_process(COMMAND cmake --build ${3RD_PARTY}/${ZLIB_VER}/${ZLIB_VER}/cmake-build --config Release --target install)
        list(APPEND CMAKE_PREFIX_PATH ${3RD_PARTY}/${ZLIB_VER}/install)

        # HDF5 itself
        download_and_extract(https://gamma.hdfgroup.org/ftp/pub/outgoing/hdf5/snapshots/v112/OLD/${HDF_VER}.tar.gz ${HDF_VER})
        execute_process(COMMAND cmake -S ${3RD_PARTY}/${HDF_VER}/${HDF_VER} -B ${3RD_PARTY}/${HDF_VER}/${HDF_VER}/cmake-build -A x64
            -DHDF5_ENABLE_Z_LIB_SUPPORT=1 -DCMAKE_PREFIX_PATH=${3RD_PARTY}/${ZLIB_VER}/install
            -DCMAKE_INSTALL_PREFIX=${3RD_PARTY}/${HDF_VER}/install)
        execute_process(COMMAND cmake --build ${3RD_PARTY}/${HDF_VER}/${HDF_VER}/cmake-build --config Release --target install)
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${3RD_PARTY}/${HDF_VER}/install)

    # Get prebuilt OpenSSL
    download_and_extract(https://nanoporetech.box.com/shared/static/paqqcwfpdjo3eqaghu2denk9vgnis2ph.zip openssl-win)
    set (OPENSSL_ROOT_DIR ${3RD_PARTY}/openssl-win)
endif() # WIN32

download_and_extract(${MKR_URL} ${MKR_DIR})
add_hdf_vbz_plugin()

if(DEFINED DORADO_LIBTORCH_DIR)
    # Use the existing libtorch we have been pointed at
    list(APPEND CMAKE_PREFIX_PATH ${DORADO_LIBTORCH_DIR})
    message(STATUS "Using existing libtorch at ${DORADO_LIBTORCH_DIR}")
else()
    # Get libtorch (if we don't already have it)
    download_and_extract(${TORCH_URL} torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME})
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_LIB}")
endif()
find_package(Torch REQUIRED)


if(DYNAMIC_HDF)
    add_link_options(-ldl)
else()
    set (HDF5_USE_STATIC_LIBRARIES On)
endif()

find_package(HDF5 COMPONENTS C CXX HL)

if(APPLE AND NOT DEFINED OPENSSL_ROOT_DIR)
    set(OPENSSL_ROOT_DIR "/opt/homebrew/opt/openssl@3")
endif()

set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)

if(WIN32)
    add_compile_options(
        # Note we need to use the generator expression to avoid setting this for CUDA.
        $<$<COMPILE_LANGUAGE:CXX>:/wd4624> # from libtorch: destructor was implicitly defined as deleted 
    )
endif()

configure_file(dorado/Version.h.in dorado/Version.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dorado)

include_directories(${HDF5_INCLUDE_DIRS}
        dorado/3rdparty/HighFive/include
        dorado/3rdparty/HDF5/apple/1.12.1/include
        dorado/3rdparty/catch2
        dorado/3rdparty/argparse
        dorado/3rdparty/toml11
        dorado/3rdparty/cpp-httplib
        dorado/3rdparty/hdf_plugins/vbz_plugin
        dorado/3rdparty/metal-cpp/metal-cpp
        dorado/3rdparty/${MKR_DIR}/include
        dorado)

# We need to force the use of the old ABI here as otherwise elzip builds with the libc++11 ABI
# and we can't link against it.  We're on the old ABI right now due to the version of pytorch 
# we are using.
add_compile_options(
        -D_GLIBCXX_USE_CXX11_ABI=0
)
        
add_subdirectory(dorado/3rdparty/elzip)

set(LIB_SOURCE_FILES
        dorado/cli/basecaller.cpp
        dorado/cli/download.cpp
        dorado/cli/cli.h
        dorado/models.h
        dorado/nn/CRFModel.h
        dorado/nn/CRFModel.cpp
        dorado/nn/ModelRunner.h
        dorado/read_pipeline/ReadPipeline.cpp
        dorado/read_pipeline/ReadPipeline.h
        dorado/read_pipeline/ScalerNode.cpp
        dorado/read_pipeline/ScalerNode.h
        dorado/read_pipeline/BasecallerNode.cpp
        dorado/read_pipeline/BasecallerNode.h
        dorado/read_pipeline/WriterNode.cpp
        dorado/read_pipeline/WriterNode.h
        dorado/data_loader/DataLoader.cpp
        dorado/data_loader/DataLoader.h
        dorado/decode/beam_search.cpp
        dorado/decode/fast_hash.cpp
        dorado/decode/fast_hash.h
        dorado/decode/beam_search.h
        dorado/decode/CPUDecoder.cpp
        dorado/decode/CPUDecoder.h
        dorado/decode/GPUDecoder.cpp
        dorado/decode/GPUDecoder.h
        dorado/decode/MTLDecoder.cpp
        dorado/decode/MTLDecoder.h
        dorado/utils/compat_utils.cpp
        dorado/utils/compat_utils.h
        dorado/utils/sequence_utils.cpp
        dorado/utils/sequence_utils.h
        dorado/utils/stitch.cpp
        dorado/utils/stitch.h
        dorado/utils/tensor_utils.cpp
        dorado/utils/tensor_utils.h)

if(APPLE)
    list(APPEND LIB_SOURCE_FILES
            dorado/nn/metal/nn.h
            dorado/nn/MetalCRFModel.h
            dorado/nn/MetalCRFModel.cpp
            dorado/utils/metal_utils.cpp
            dorado/utils/metal_utils.h)
endif()

add_library(dorado_lib ${LIB_SOURCE_FILES})
set_target_properties(dorado_lib
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_link_libraries(dorado_lib
    ${APPLE_FWK_FOUNDATION}
    ${APPLE_FWK_QUARTZ_CORE}
    ${APPLE_FWK_METAL}
    ${HDF5_C_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${MKR_LIBRARIES}
    vbz_hdf_plugin
    OpenSSL::SSL
    elzip
)
if(APPLE)
    add_custom_target(metal-lib DEPENDS default.metallib)
    add_dependencies(dorado_lib metal-lib)
endif()

add_executable(dorado
    dorado/main.cpp)
set_target_properties(dorado
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(dorado dorado_lib)
target_link_libraries(dorado elzip)

add_subdirectory(tests)

# Note that we can't cuurently do add_subdirectory(dorado) (or any subfolder thereof) as we're
# making an exe called 'dorado' right into the root here, which causes conflicts with cmake naming.

if(LINUX OR WIN32)
    if(BUILD_KOI_FROM_SOURCE)
        message("Building Koi from source")
        # Note that any source file which gets compiled multiple times with differing flags won't get added here. See below.
        set(KOI_SOURCE_FILES dorado/3rdparty/koi/koi/lib/ctc_simple.cu
                dorado/3rdparty/koi/koi/lib/cuda_beam_search_constants.h
                dorado/3rdparty/koi/koi/lib/decoder_lib.cu
                dorado/3rdparty/koi/koi/lib/lib.h
                )

        set(KOI_FLAGS -c
                --ptxas-options=-v
                --use_fast_math
                --restrict
                )

        set(CMAKE_CUDA_ARCHITECTURES 70)
        add_library(koi_lib ${KOI_SOURCE_FILES})
        target_compile_options(koi_lib
                PRIVATE
                ${KOI_FLAGS}
                )
        if(LINUX)
            target_compile_options(koi_lib
                    PRIVATE
                    -Xcompiler
                    '-fPIC'
                    )
        endif()

        set_source_files_properties(dorado/3rdparty/koi/koi/lib/ctc_simple.cu PROPERTIES COMPILE_FLAGS "-DFLOAT=float -DZERO=-1e38 -DSUM=logsumexp2 -DMUL=add")
        set_source_files_properties(dorado/3rdparty/koi/koi/lib/decoder_lib.cu PROPERTIES COMPILE_FLAGS "-D REMOVE_FIXED_BEAM_STAYS=1")

        # Here we need to compile the same source file more than once with different flags, so we're going to make a target for
        # each version of the source file we want to compile, then add that target to the koi lib as a dep. Note that the subtarget
        # WON'T be archived into koi_lib, but will be added as a dep to anything that depends on koi_lib

        #sparse_logZ
        add_library(koi_sparse_logz_max dorado/3rdparty/koi/koi/lib/sparse_logZ.cu)
        target_compile_options(koi_sparse_logz_max PRIVATE ${KOI_FLAGS} -DFLOAT=float -DZERO=-1e38 -DSUM=max_
                -DLOGZ_FWD_HOST_FN_NAME=logZ_fwd_host_max -DLOGZ_FWD_KERNEL_NAME=logz_fwd_max -DNZ=5 -DK=1 -DMUL=add -DSUFFIX=_max)
        target_link_libraries(koi_lib koi_sparse_logz_max)
        add_library(koi_sparse_logz_log dorado/3rdparty/koi/koi/lib/sparse_logZ.cu)
        target_compile_options(koi_sparse_logz_log PRIVATE ${KOI_FLAGS} -DFLOAT=float -DZERO=-1e38 -DSUM=logsumexp
                -DLOGZ_FWD_HOST_FN_NAME=logZ_fwd_host_log -DLOGZ_FWD_KERNEL_NAME=logz_fwd_log -DNZ=5 -DK=1 -DMUL=add -DSUFFIX=_log)
        target_link_libraries(koi_lib koi_sparse_logz_log)

        #lstm
        add_library(koi_lstm_fwd dorado/3rdparty/koi/koi/lib/lstm.cu)
        target_compile_options(koi_lstm_fwd PRIVATE ${KOI_FLAGS} -DREVERSE=0)
        target_link_libraries(koi_lib koi_lstm_fwd)
        add_library(koi_lstm_rev dorado/3rdparty/koi/koi/lib/lstm.cu)
        target_compile_options(koi_lstm_rev PRIVATE ${KOI_FLAGS} -DREVERSE=1)
        target_link_libraries(koi_lib koi_lstm_rev)

        #lstm_quantized
        add_library(koi_lstm_quantized_fwd_96 dorado/3rdparty/koi/koi/lib/lstm_quantized.cu)
        target_compile_options(koi_lstm_quantized_fwd_96 PRIVATE ${KOI_FLAGS} -DREVERSE=0 -DLAYER_OUT_SIZE=96)
        target_link_libraries(koi_lib koi_lstm_quantized_fwd_96)
        add_library(koi_lstm_quantized_rev_96 dorado/3rdparty/koi/koi/lib/lstm_quantized.cu)
        target_compile_options(koi_lstm_quantized_rev_96 PRIVATE ${KOI_FLAGS} -DREVERSE=1 -DLAYER_OUT_SIZE=96)
        target_link_libraries(koi_lib koi_lstm_quantized_rev_96)
        add_library(koi_lstm_quantized_fwd_128 dorado/3rdparty/koi/koi/lib/lstm_quantized.cu)
        target_compile_options(koi_lstm_quantized_fwd_128 PRIVATE ${KOI_FLAGS} -DREVERSE=0 -DLAYER_OUT_SIZE=128)
        target_link_libraries(koi_lib koi_lstm_quantized_fwd_128)
        add_library(koi_lstm_quantized_rev_128 dorado/3rdparty/koi/koi/lib/lstm_quantized.cu)
        target_compile_options(koi_lstm_quantized_rev_128 PRIVATE ${KOI_FLAGS} -DREVERSE=1 -DLAYER_OUT_SIZE=128)
        target_link_libraries(koi_lib koi_lstm_quantized_rev_128)

        # Finally add a dependency on koi_lib to the dorado project
        target_link_libraries(dorado_lib koi_lib)

    else() # end BUILD_KOI_FROM_SOURCE
        if(LINUX)
            download_and_extract(https://nanoporetech.box.com/shared/static/iow91rd925cbocxhxbss50wr9e2vahv4.gz koi_lib)
        elseif(WIN32)
            download_and_extract(https://nanoporetech.box.com/shared/static/psfgwpjfr0mxu3xc95lk1h3c1vj6wb6m.gz koi_lib)
        endif()
        # Build koi from prebuilt libs

        message ("Building Koi from prebuilt ${3RD_PARTY}/koi_lib")
        target_include_directories(dorado_lib PUBLIC
            ${3RD_PARTY}/koi_lib/koi_lib/include
        )
        target_link_directories(dorado_lib PUBLIC
            ${3RD_PARTY}/koi_lib/koi_lib/lib
        )

        #sparse_logZ
        target_link_libraries(dorado_lib koi_sparse_logz_max)
        target_link_libraries(dorado_lib koi_sparse_logz_log)
        #lstm
        target_link_libraries(dorado_lib koi_lstm_fwd)
        target_link_libraries(dorado_lib koi_lstm_rev)
        #lstm_quantized
        target_link_libraries(dorado_lib koi_lstm_quantized_fwd_96)
        target_link_libraries(dorado_lib koi_lstm_quantized_rev_96)
        target_link_libraries(dorado_lib koi_lstm_quantized_fwd_128)
        target_link_libraries(dorado_lib koi_lstm_quantized_rev_128)
        # koi_lib
        target_link_libraries(dorado_lib koi_lib)

    endif()
endif() # LINUX OR WIN32

