cmake_minimum_required(VERSION 3.20)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

project(dorado)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DORADO_3RD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/dorado/3rdparty)

include(cmake/DoradoVersion.cmake)
include(cmake/UpdateSubmodules.cmake)
include(cmake/DownloadAndExtract.cmake)
if(DORADO_USE_FAST5)
    include(cmake/HDF5.cmake)
    include(cmake/VbzPlugin.cmake)
endif()

if(DORADO_USE_POD5)
    add_definitions("USE_POD5")
    include(cmake/Pod5.cmake)
endif()
include(cmake/Koi.cmake)
include(cmake/Torch.cmake)
include(cmake/OpenSSL.cmake)
include(cmake/Htslib.cmake)


if(DORADO_USING_OLD_CPP_ABI)
    # We need to force the use of the old ABI here, if we are building in an old ABI context, as otherwise elzip builds
    # with the libc++11 ABI and we can't link against it.
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

add_subdirectory(dorado/3rdparty/elzip)

set(BUILD_TESTING OFF)
add_subdirectory(dorado/3rdparty/edlib EXCLUDE_FROM_ALL)
set(BUILD_TESTING ON)

enable_testing()

configure_file(dorado/Version.h.in dorado/Version.h)

if(WIN32)
    message(STATUS "toolset: ${CMAKE_GENERATOR_TOOLSET}")
    message(STATUS "platform: ${CMAKE_GENERATOR_PLATFORM}")
    message(STATUS "vs platform: ${CMAKE_VS_PLATFORM_NAME}")
endif()

if(DEFINED CUDA_TOOLKIT_ROOT_DIR)
    message(STATUS "CUDA toolkit dir is ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

set(LIB_SOURCE_FILES
        dorado/nn/CRFModel.h
        dorado/nn/CRFModel.cpp
        dorado/nn/ModelRunner.h
        dorado/nn/RemoraModel.cpp
        dorado/nn/RemoraModel.h
        dorado/read_pipeline/ReadPipeline.cpp
        dorado/read_pipeline/ReadPipeline.h
        dorado/read_pipeline/ScalerNode.cpp
        dorado/read_pipeline/ScalerNode.h
        dorado/read_pipeline/StereoDuplexEncoderNode.cpp
        dorado/read_pipeline/StereoDuplexEncoderNode.h
        dorado/read_pipeline/BasecallerNode.cpp
        dorado/read_pipeline/BasecallerNode.h
        dorado/read_pipeline/ModBaseCallerNode.cpp
        dorado/read_pipeline/ModBaseCallerNode.h
        dorado/read_pipeline/WriterNode.cpp
        dorado/read_pipeline/WriterNode.h
        dorado/read_pipeline/BaseSpaceDuplexCallerNode.cpp
        dorado/read_pipeline/BaseSpaceDuplexCallerNode.h
        dorado/data_loader/DataLoader.cpp
        dorado/data_loader/DataLoader.h
        dorado/decode/beam_search.cpp
        dorado/decode/fast_hash.cpp
        dorado/decode/fast_hash.h
        dorado/decode/beam_search.h
        dorado/decode/CPUDecoder.cpp
        dorado/decode/CPUDecoder.h
        dorado/modbase/remora_encoder.cpp
        dorado/modbase/remora_encoder.h
        dorado/modbase/remora_scaler.cpp
        dorado/modbase/remora_scaler.h
        dorado/modbase/remora_utils.cpp
        dorado/modbase/remora_utils.h
        dorado/utils/base_mod_utils.cpp
        dorado/utils/base_mod_utils.h
        dorado/utils/compat_utils.cpp
        dorado/utils/compat_utils.h
        dorado/utils/log_utils.h
        dorado/utils/log_utils.cpp
        dorado/utils/math_utils.h
        dorado/utils/models.cpp
        dorado/utils/models.h
        dorado/utils/module_utils.h
        dorado/utils/parameters.h
        dorado/utils/sequence_utils.cpp
        dorado/utils/sequence_utils.h
        dorado/utils/stitch.cpp
        dorado/utils/stitch.h
        dorado/utils/tensor_utils.cpp
        dorado/utils/tensor_utils.h
        dorado/utils/bam_utils.cpp
        dorado/utils/bam_utils.h
        dorado/utils/duplex_utils.h
        dorado/utils/duplex_utils.cpp
        dorado/data_loader/slow5_thread.h
        dorado/data_loader/slow5_thread.cpp)

function(add_slow5lib)
    include_directories(dorado/3rdparty/slow5lib/include)
    include_directories(dorado/3rdparty/slow5lib/src)
    add_subdirectory(dorado/3rdparty/slow5lib)
endfunction()

add_slow5lib()

if(APPLE)
    include(cmake/Metal.cmake)
    list(APPEND LIB_SOURCE_FILES
            dorado/nn/MetalCRFModel.h
            dorado/nn/MetalCRFModel.cpp
            dorado/utils/metal_utils.cpp
            dorado/utils/metal_utils.h
            )
else ()
    list(APPEND LIB_SOURCE_FILES
            dorado/decode/GPUDecoder.cpp
            dorado/decode/GPUDecoder.h
            dorado/nn/CudaCRFModel.h
            dorado/nn/CudaCRFModel.cpp
            dorado/utils/cuda_utils.cpp
            dorado/utils/cuda_utils.h
            )
endif()

add_library(dorado_lib ${LIB_SOURCE_FILES})

set_target_properties(dorado_lib
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        )

include_directories(${CMAKE_CURRENT_BINARY_DIR}/dorado
        ${HDF5_INCLUDE_DIRS}
        ${KOI_INCLUDE}
        ${POD5_INCLUDE}
        dorado/3rdparty/HighFive/include
        dorado/3rdparty/catch2
        dorado/3rdparty/argparse
        dorado/3rdparty/toml11
        dorado/3rdparty/cpp-httplib
        dorado/3rdparty/metal-cpp/metal-cpp
        dorado/3rdparty/hdf_plugins/vbz_plugin
        ${HTSLIB_DIR}
        dorado/3rdparty/cxxpool/src
        dorado/3rdparty/NVTX/c/include
        dorado
        dorado/3rdparty/spdlog/include
        )

enable_testing()

if(DORADO_USE_FAST5)
    # Can skip this if we're consuming the lib from a parent project that already has hdf_plugins
    add_hdf_vbz_plugin()
    target_link_libraries(dorado_lib
            ${HDF5_C_LIBRARIES}
            vbz_hdf_plugin
            )
endif()
if(DORADO_USE_POD5)
    target_link_libraries(dorado_lib
            ${POD5_LIBRARIES}
            )

endif()

target_link_libraries(dorado_lib
        ${APPLE_FWK_FOUNDATION}
        ${APPLE_FWK_QUARTZ_CORE}
        ${APPLE_FWK_METAL}
        ${IOKIT}
        ${TORCH_LIBRARIES}
        ${KOI_LIBRARIES}
        ${HTSLIB_LIBRARIES}
        slow5
        OpenSSL::SSL
        edlib
        elzip
        )

if(NOT WIN32)
    add_dependencies(dorado_lib htslib_project)
endif()

if(APPLE)
    add_custom_target(metal-lib DEPENDS default.metallib)
    add_dependencies(dorado_lib metal-lib)
endif()

if(NOT DORADO_LIB_ONLY)
    if(NOT WIN32)
        # Set up RPATHs so we can find dependencies
        set(CMAKE_SKIP_RPATH FALSE)
        # Note: we don't need the relative lib dir if everything is in
        if(APPLE)
            set(CMAKE_INSTALL_RPATH "@executable_path/;@executable_path/../lib")
        else()
            set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib/;$ORIGIN")
        endif()
        set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    endif()

    add_executable(dorado
            dorado/main.cpp
            dorado/cli/duplex.cpp
            dorado/cli/basecaller.cpp
            dorado/cli/benchmark.cpp
            dorado/cli/download.cpp
            dorado/cli/cli.h
            )

    set_target_properties(dorado
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            )

    target_link_libraries(dorado dorado_lib)

    if (DEFINED DORADO_INSTALL_PATH)
        set(CMAKE_INSTALL_PREFIX ${DORADO_INSTALL_PATH})
    else()
        set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/../dist)
    endif()

    install(TARGETS dorado RUNTIME)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.so*")
    install(FILES ${TORCH_DLLS} DESTINATION lib COMPONENT redist_libs)
    if(DYNAMIC_HDF)
        file(GLOB HDF_DLLS "${HDF5_CXX_LIBRARY_DIRS}/*hdf5*.so*")
        install(FILES ${HDF_DLLS} DESTINATION lib COMPONENT redist_libs)
    endif()
    file(GLOB SZ_DLLS "${HDF5_CXX_LIBRARY_DIRS}/libsz.so*")
    install(FILES ${SZ_DLLS} DESTINATION lib COMPONENT redist_libs)
    file(GLOB AEC_DLLS "${HDF5_CXX_LIBRARY_DIRS}/libaec.so*")
    list(LENGTH AEC_DLLS AEC_DLL_LIST_LEN)
    if(AEC_DLL_LIST_LEN)
        install(FILES ${AEC_DLLS} DESTINATION lib COMPONENT redist_libs)
    endif()

    # CUDA system DLLs we depend on:
    file(GLOB CUDART_DLLS "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib/*cudart.so*")
    file(GLOB CUBLAS_DLLS "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib/*cublas*.so*")
    file(GLOB NVTOOLS_DLLS "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib/*nvToolsExt.so*")
    install(FILES ${CUDART_DLLS} DESTINATION lib COMPONENT redist_libs)
    install(FILES ${CUBLAS_DLLS} DESTINATION lib COMPONENT redist_libs)
    install(FILES ${NVTOOLS_DLLS} DESTINATION lib COMPONENT redist_libs)

    # If zstd has been dynamically linked, add the .so to the package
#    get_filename_component(ZSTD_LIBRARY_PATH ${ZSTD_LIBRARY_RELEASE} DIRECTORY)
#    file(GLOB ZSTD_DLLS "${ZSTD_LIBRARY_PATH}/*zstd.so*")
#    install(FILES ${ZSTD_DLLS} DESTINATION lib COMPONENT redist_libs)

elseif(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dll")
    install(FILES ${TORCH_DLLS} DESTINATION bin COMPONENT redist_libs)
    file(GLOB HTSLIB_DLLS "${HTSLIB_DIR}/*.dll")
    install(FILES ${HTSLIB_DLLS} DESTINATION bin COMPONENT redist_libs)
elseif(APPLE)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dylib")
    install(FILES ${TORCH_DLLS} DESTINATION lib COMPONENT redist_libs)
endif()

#add_subdirectory(tests)

if(NOT DORADO_LIB_ONLY)
    include(cmake/DoradoPackaging.cmake)
endif()
