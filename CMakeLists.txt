cmake_minimum_required(VERSION 3.20)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

project(dorado)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(WIN32)
  message(STATUS "toolset: ${CMAKE_GENERATOR_TOOLSET}")
  message(STATUS "platform: ${CMAKE_GENERATOR_PLATFORM}")
  message(STATUS "vs platform: ${CMAKE_VS_PLATFORM_NAME}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DORADO_3RD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/dorado/3rdparty)

find_package(CUDAToolkit QUIET)

if(${CUDAToolkit_FOUND})
  file(REAL_PATH ${CUDAToolkit_TARGET_DIR} CUDAToolkit_REAL_DIR)
  message(STATUS "Found CUDA ${CUDAToolkit_VERSION} (${CUDAToolkit_TARGET_DIR} -> ${CUDAToolkit_REAL_DIR})")
endif()

include(cmake/DoradoVersion.cmake)
include(cmake/UpdateSubmodules.cmake)
include(cmake/DownloadAndExtract.cmake)
include(cmake/SharedLibHelpers.cmake)

include(cmake/HDF5.cmake)
include(cmake/VbzPlugin.cmake)
include(cmake/Koi.cmake)
include(cmake/Pod5.cmake)
include(cmake/Torch.cmake)
include(cmake/OpenSSL.cmake)
include(cmake/Htslib.cmake)

if(DORADO_USING_OLD_CPP_ABI)
  # We need to force the use of the old ABI here, if we are building in an old ABI context, as otherwise elzip builds
  # with the libc++11 ABI and we can't link against it.
  add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

# We don't support GPU builds on macOS/x64
if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(DORADO_GPU_BUILD FALSE)
else()
    set(DORADO_GPU_BUILD TRUE)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if((CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*") AND (${CUDAToolkit_VERSION} VERSION_LESS 11.0))
        add_compile_definitions(DORADO_TX2)
    endif()
endif()

# ELZIP_DECOMPRESS_ONLY stops minizip from adding OpenSSL as a target, preventing use of three dylibs on osx.
set(ELZIP_DECOMPRESS_ONLY ON)
add_subdirectory(${DORADO_3RD_PARTY}/elzip)

set(ONT_MM2_EXE OFF)
add_subdirectory(${DORADO_3RD_PARTY}/ont-minimap2)

set(BUILD_TESTING OFF)
add_subdirectory(${DORADO_3RD_PARTY}/edlib EXCLUDE_FROM_ALL)
set(BUILD_TESTING ON)

enable_testing()

configure_file(dorado/Version.h.in dorado/Version.h)

set(LIB_SOURCE_FILES
    dorado/nn/CRFModel.h
    dorado/nn/CRFModel.cpp
    dorado/nn/ModelRunner.h
    dorado/nn/RemoraModel.cpp
    dorado/nn/RemoraModel.h
    dorado/read_pipeline/FakeDataLoader.cpp
    dorado/read_pipeline/FakeDataLoader.h
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
    dorado/read_pipeline/ReadFilterNode.cpp
    dorado/read_pipeline/ReadFilterNode.h
    dorado/read_pipeline/BaseSpaceDuplexCallerNode.cpp
    dorado/read_pipeline/BaseSpaceDuplexCallerNode.h
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
    dorado/utils/AsyncQueue.h
    dorado/utils/base_mod_utils.cpp
    dorado/utils/base_mod_utils.h
    dorado/utils/compat_utils.cpp
    dorado/utils/compat_utils.h
    dorado/utils/log_utils.h
    dorado/utils/log_utils.cpp
    dorado/utils/math_utils.h
    dorado/utils/module_utils.h
    dorado/utils/parameters.h
    dorado/utils/sequence_utils.cpp
    dorado/utils/sequence_utils.h
    dorado/utils/stitch.cpp
    dorado/utils/stitch.h
    dorado/utils/tensor_utils.cpp
    dorado/utils/tensor_utils.h
    dorado/utils/trim.cpp
    dorado/utils/trim.h
    dorado/utils/bam_utils.cpp
    dorado/utils/bam_utils.h
    dorado/utils/duplex_utils.h
    dorado/utils/duplex_utils.cpp
    dorado/utils/basecaller_utils.h
    dorado/utils/basecaller_utils.cpp)

if (DORADO_GPU_BUILD)
    if(APPLE)
        include(cmake/Metal.cmake)
        list(APPEND LIB_SOURCE_FILES
            dorado/nn/MetalCRFModel.h
            dorado/nn/MetalCRFModel.cpp
            dorado/utils/metal_utils.cpp
            dorado/utils/metal_utils.h
        )
    else()
        list(APPEND LIB_SOURCE_FILES
            dorado/decode/GPUDecoder.cpp
            dorado/decode/GPUDecoder.h
            dorado/nn/CudaCRFModel.h
            dorado/nn/CudaCRFModel.cpp
            dorado/utils/cuda_utils.cpp
            dorado/utils/cuda_utils.h
        )
    endif()
endif()

add_library(dorado_lib ${LIB_SOURCE_FILES})

set_target_properties(dorado_lib
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_compile_definitions(dorado_lib
    PUBLIC
        DORADO_GPU_BUILD=$<BOOL:${DORADO_GPU_BUILD}>
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/dorado
    ${TORCH_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${KOI_INCLUDE}
    ${POD5_INCLUDE}
    ${HTSLIB_DIR}
    ${DORADO_3RD_PARTY}/HighFive/include
    ${DORADO_3RD_PARTY}/catch2
    ${DORADO_3RD_PARTY}/argparse
    ${DORADO_3RD_PARTY}/toml11
    ${DORADO_3RD_PARTY}/cpp-httplib
    ${DORADO_3RD_PARTY}/metal-cpp/metal-cpp
    ${DORADO_3RD_PARTY}/hdf_plugins/vbz_plugin
    ${DORADO_3RD_PARTY}/cxxpool/src
    ${DORADO_3RD_PARTY}/NVTX/c/include
    ${DORADO_3RD_PARTY}/spdlog/include
    ${DORADO_3RD_PARTY}/spdlog/include
    ${DORADO_3RD_PARTY}/indicators/include
    ${DORADO_3RD_PARTY}/ont-minimap2/src/3rdparty/minimap2
    dorado)

enable_testing()

if(NOT SKIP_HDF_PLUGINS)
  # Can skip this if we're consuming the lib from a parent project that already has hdf_plugins
  add_hdf_vbz_plugin()
endif()

target_link_libraries(dorado_lib
    ${APPLE_FWK_FOUNDATION}
    ${APPLE_FWK_QUARTZ_CORE}
    ${APPLE_FWK_METAL}
    ${TORCH_LIBRARIES}
    ${KOI_LIBRARIES}
    htslib
    vbz_hdf_plugin
    OpenSSL::SSL
    edlib
)

if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    target_link_libraries(dorado_lib ${IOKIT})
endif()

if(NOT WIN32)
    add_dependencies(dorado_lib htslib_project)
endif()

if(APPLE AND DORADO_GPU_BUILD)
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
        dorado/cli/aligner.cpp
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

    target_link_libraries(dorado
        dorado_lib
        dorado_io_lib
        dorado_models_lib
        minimap2
    )

    if (DEFINED DORADO_INSTALL_PATH)
        set(CMAKE_INSTALL_PREFIX ${DORADO_INSTALL_PATH})
    else()
        set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/../dist)
    endif()

    install(TARGETS dorado RUNTIME)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")

    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        # CUDA toolkit DLLs we depend on:
        set(PATTERNS
            "*cusparse.so*"
            "*curand.so*"
            "*cusolver.so*"
            "*cufft.so*"
        )
        foreach(PATTERN ${PATTERNS})
            file(GLOB DEP_DLLS "${CUDAToolkit_ROOT}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib/${PATTERN}")
            install(FILES ${DEP_DLLS} DESTINATION lib COMPONENT redist_libs)
        endforeach()        
    endif()

    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.so*")
    install(FILES ${TORCH_DLLS} DESTINATION lib COMPONENT redist_libs)

    set(VERSIONED_CUDA_LIBS
        libcublas-*.so.11
        libcublasLt-*.so.11
        libcudart-*.so.11.0
        libnvrtc-*.so.11.2
        libnvrtc-builtins-*.so.11.8
        libnvToolsExt-*.so.1
    )

    foreach(LIB IN LISTS VERSIONED_CUDA_LIBS)
        file(GLOB VERSIONED_CUDA_LIB "${TORCH_LIB}/lib/${LIB}")
        if (VERSIONED_CUDA_LIB)
            set(target ${VERSIONED_CUDA_LIB})
            string(REGEX REPLACE "-[0-9a-f]+[.]" "." link ${target})
            get_filename_component(target_name ${target} NAME)
            get_filename_component(link_name ${link} NAME)
            execute_process(COMMAND ln -rfs ${target_name} ${link_name} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
            install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${link_name} DESTINATION lib COMPONENT redist_libs)
        endif()
    endforeach()

    if(DYNAMIC_HDF)
        string(REPLACE "."  "" SHARED_LIB_EXT "${CMAKE_SHARED_LIBRARY_SUFFIX}")
        FILTER_LIST("${HDF5_C_LIBRARIES}" DEBUG_LIBRARIES debug optimized ${SHARED_LIB_EXT})
        RESOLVE_SYMLINKS("${DEBUG_LIBRARIES}" NEW_HDF_DEBUG_LIBRARIES)
        foreach(HDF_LIB IN LISTS NEW_HDF_DEBUG_LIBRARIES)
	    if(${HDF_LIB} MATCHES "hdf5")
                install(FILES ${HDF_LIB} DESTINATION lib COMPONENT redist_libs CONFIGURATIONS Debug)
            endif()
        endforeach()
        FILTER_LIST("${HDF5_C_LIBRARIES}" RELEASE_LIBRARIES optimized debug ${SHARED_LIB_EXT})
        RESOLVE_SYMLINKS("${RELEASE_LIBRARIES}" NEW_HDF_RELEASE_LIBRARIES)
        foreach(HDF_LIB IN LISTS NEW_HDF_RELEASE_LIBRARIES)
	    if(${HDF_LIB} MATCHES "hdf5")
                install(FILES ${HDF_LIB} DESTINATION lib COMPONENT redist_libs CONFIGURATIONS Release ReleaseWithDebInfo)
            endif()
        endforeach()
    endif()

    find_library(SZ_DLL sz REQUIRED)
    get_filename_component(SZ_DLL_PATH ${SZ_DLL} DIRECTORY)
    file(GLOB SZ_DLLS "${SZ_DLL_PATH}/libsz.so*")
    install(FILES ${SZ_DLLS} DESTINATION lib COMPONENT redist_libs)

    find_library(AEC_DLL aec REQUIRED)
    get_filename_component(AEC_DLL_PATH ${AEC_DLL} DIRECTORY)
    file(GLOB AEC_DLLS "${AEC_DLL_PATH}/libaec.so*")
    install(FILES ${AEC_DLLS} DESTINATION lib COMPONENT redist_libs)

    # If zstd has been dynamically linked, add the .so to the package
    get_filename_component(ZSTD_LIBRARY_PATH ${ZSTD_LIBRARY_RELEASE} DIRECTORY)
    file(GLOB ZSTD_DLLS "${ZSTD_LIBRARY_PATH}/*zstd.so*")
    install(FILES ${ZSTD_DLLS} DESTINATION lib COMPONENT redist_libs)

elseif(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dll")
    install(FILES ${TORCH_DLLS} DESTINATION bin COMPONENT redist_libs)
    file(GLOB HTSLIB_DLLS "${HTSLIB_DIR}/*.dll")
    install(FILES ${HTSLIB_DLLS} DESTINATION bin COMPONENT redist_libs)
elseif(APPLE)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dylib")
    install(FILES ${TORCH_DLLS} DESTINATION lib COMPONENT redist_libs)
endif()

add_library(dorado_models_lib
    dorado/utils/models.cpp
    dorado/utils/models.h
)

target_link_libraries(dorado_models_lib
    OpenSSL::SSL
    elzip
)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    add_library(dorado_io_lib
        dorado/data_loader/DataLoader.cpp
        dorado/data_loader/DataLoader.h
    )

    if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        add_library(dorado_allocator_lib STATIC
            dorado/utils/allocator_override.cpp
        )
        target_link_libraries(dorado_allocator_lib
            ${POD5_LIBRARIES}
        )

        target_link_libraries(dorado_io_lib dorado_allocator_lib)
    endif()

    target_link_libraries(dorado_io_lib
       ${POD5_LIBRARIES}
       ${HDF5_C_LIBRARIES}
       ${CMAKE_DL_LIBS}
    )

    if(WIN32)
        target_link_libraries(dorado_io_lib
           ${ZLIB_LIBRARIES}
        )
    endif()

    set_target_properties(dorado_io_lib
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )

    add_subdirectory(tests)
endif()

if(NOT DORADO_LIB_ONLY)
    include(cmake/DoradoPackaging.cmake)
endif()
