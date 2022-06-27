cmake_minimum_required(VERSION 3.20)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

project(dorado)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DORADO_3RD_PARTY ${CMAKE_SOURCE_DIR}/dorado/3rdparty)

include(cmake/DoradoVersion.cmake)
include(cmake/UpdateSubmodules.cmake)
include(cmake/DownloadAndExtract.cmake)

include(cmake/HDF5.cmake)
include(cmake/VbzPlugin.cmake)
include(cmake/Koi.cmake)
include(cmake/Pod5.cmake)
include(cmake/Torch.cmake)
include(cmake/OpenSSL.cmake)

# We need to force the use of the old ABI here as otherwise elzip builds with the libc++11 ABI
# and we can't link against it.  We're on the old ABI right now due to the version of pytorch 
# we are using.
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
add_subdirectory(dorado/3rdparty/elzip)

enable_testing()
add_hdf_vbz_plugin()
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
    dorado/cli/basecaller.cpp
    dorado/cli/download.cpp
    dorado/cli/cli.h
    dorado/models.h
    dorado/nn/CRFModel.h
    dorado/nn/CRFModel.cpp
    dorado/nn/ModelRunner.h
    dorado/nn/RemoraModel.cpp
    dorado/nn/RemoraModel.h
    dorado/read_pipeline/ReadPipeline.cpp
    dorado/read_pipeline/ReadPipeline.h
    dorado/read_pipeline/ScalerNode.cpp
    dorado/read_pipeline/ScalerNode.h
    dorado/read_pipeline/BasecallerNode.cpp
    dorado/read_pipeline/BasecallerNode.h
    dorado/read_pipeline/ModBaseCallerNode.cpp
    dorado/read_pipeline/ModBaseCallerNode.h
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
    dorado/utils/base_mod_utils.cpp
    dorado/utils/base_mod_utils.h
    dorado/utils/base64_utils.h
    dorado/utils/compat_utils.cpp
    dorado/utils/compat_utils.h
    dorado/utils/math_utils.h
    dorado/utils/module_utils.h
    dorado/utils/sequence_utils.cpp
    dorado/utils/sequence_utils.h
    dorado/utils/stitch.cpp
    dorado/utils/stitch.h
    dorado/utils/tensor_utils.cpp
    dorado/utils/tensor_utils.h
)

if(APPLE)
    include(cmake/Metal.cmake)
    list(APPEND LIB_SOURCE_FILES
        dorado/nn/metal/nn.h
        dorado/nn/MetalCRFModel.h
        dorado/nn/MetalCRFModel.cpp
        dorado/utils/metal_utils.cpp
        dorado/utils/metal_utils.h
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
    dorado
)

target_link_libraries(dorado_lib
    ${APPLE_FWK_FOUNDATION}
    ${APPLE_FWK_QUARTZ_CORE}
    ${APPLE_FWK_METAL}
    ${HDF5_C_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${POD5_LIBRARIES}
    ${KOI_LIBRARIES}
    vbz_hdf_plugin
    OpenSSL::SSL
    elzip
)

if(APPLE)
    add_custom_target(metal-lib DEPENDS default.metallib)
    add_dependencies(dorado_lib metal-lib)
endif()

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

add_executable(dorado dorado/main.cpp)

set_target_properties(dorado
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(dorado elzip)
target_link_libraries(dorado dorado_lib)


set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/../dist)

install(TARGETS dorado RUNTIME)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.so*")
    install(FILES ${TORCH_DLLS} DESTINATION lib)
elseif(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dll")
    install(FILES ${TORCH_DLLS} DESTINATION bin)
elseif(APPLE)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.dylib")
    install(FILES ${TORCH_DLLS} DESTINATION lib)
endif()

add_subdirectory(tests)

include(cmake/DoradoPackaging.cmake)
