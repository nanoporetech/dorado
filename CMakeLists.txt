cmake_minimum_required(VERSION 3.25)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)

project(dorado LANGUAGES C CXX)
if(APPLE)
  enable_language(OBJC OBJCXX)
endif()

# BUILD_SHARED_LIBS is a global variable, so if we don't set it here at the start, we won't get
#  the same behaviour if we re-configure CMake compared to a clean configure, because it's
#  eventually set elsewhere
set(BUILD_SHARED_LIBS OFF)

# Allow targets with no side effects to be built in parallel when using Makefiles.
# Koi uses OBJECT targets which are (currently) incompatible with this.
# See https://gitlab.kitware.com/cmake/cmake/-/issues/24058.
if (NOT BUILD_KOI_FROM_SOURCE)
    set(CMAKE_OPTIMIZE_DEPENDENCIES ON)
endif()

get_cmake_property(MULTI_CONFIG GENERATOR_IS_MULTI_CONFIG)
if (NOT MULTI_CONFIG)
  if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# If no prefix is provided we install next to the binary directory.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/../dist CACHE PATH "" FORCE)
endif()

if(WIN32)
    # all third parties are release builds, so we must match the windows runtime
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    message(STATUS "toolset: ${CMAKE_GENERATOR_TOOLSET}")
    message(STATUS "platform: ${CMAKE_GENERATOR_PLATFORM}")
    message(STATUS "vs platform: ${CMAKE_VS_PLATFORM_NAME}")

    # MSVC won't parallelise individual targets unless you tell it to.
    add_compile_options("$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/MP${WIN_THREADS}>")
endif()

if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    message(FATAL_ERROR "MacOS x86_64 builds are no longer supported.")
endif()

if (APPLE AND NOT ECM_ENABLE_SANITIZERS)
    # Prefer static libs if they exist so that we don't run into issues
    # linking to dynamic libraries that live in brew.
    list(PREPEND CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set warnings and RPATH before we create any targets.
include(cmake/Warnings.cmake)
include(cmake/SetRPATH.cmake)

set(DORADO_3RD_PARTY_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/dorado/3rdparty)
set(DORADO_3RD_PARTY_DOWNLOAD ${CMAKE_CURRENT_BINARY_DIR}/download
    CACHE PATH "Location to download 3rdparty libraries into")

# All builds are now GPU builds
set(DORADO_GPU_BUILD TRUE)

find_package(CUDAToolkit QUIET)

if(${CUDAToolkit_FOUND})
  file(REAL_PATH ${CUDAToolkit_TARGET_DIR} CUDAToolkit_REAL_DIR)
  message(STATUS "Found CUDA ${CUDAToolkit_VERSION} (${CUDAToolkit_TARGET_DIR} -> ${CUDAToolkit_REAL_DIR})")
endif()

if(DEFINED ENV{DORADO_CDN_URL_OVERRIDE})
    set(DORADO_CDN_URL "$ENV{DORADO_CDN_URL_OVERRIDE}")
else()
    set(DORADO_CDN_URL "https://cdn.oxfordnanoportal.com/software/analysis")
endif()
message(STATUS "Using CDN URL ${DORADO_CDN_URL} for downloads")

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
if (DORADO_GPU_BUILD AND APPLE)
    include(cmake/Metal.cmake)
endif()

# Add sanitizer options to compilation flags
include(cmake/ECMEnableSanitizers.cmake)
if (ECM_ENABLE_SANITIZERS)
    # Always emit debug info to provide better stack traces
    add_compile_options(-g)
    # Add a way to detect if we're compiling with sanitizers enabled, since UBSan
    # detection isn't available in GCC
    foreach (SANITIZER ${ECM_ENABLE_SANITIZERS})
        string(TOUPPER ${SANITIZER} SANITIZER)
        add_compile_definitions(DORADO_SANITIZE_${SANITIZER})
    endforeach()
endif()

if(DORADO_USING_OLD_CPP_ABI)
  # We need to force the use of the old ABI here, if we are building in an old ABI context, as otherwise elzip builds
  # with the libc++11 ABI and we can't link against it.
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

# Use ccache for C and C++ if it's available
find_program(CCACHE_EXE ccache)
option(DORADO_DISABLE_CCACHE "Explicitly disable the use of ccache" OFF)
if (CCACHE_EXE AND NOT DORADO_DISABLE_CCACHE)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_EXE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXE})
    set(DORADO_ENABLE_PCH FALSE)
else()
    # Fallback to using a PCH if we don't have ccache support, since making them work together isn't simple.
    set(DORADO_ENABLE_PCH TRUE)
endif()

# Bring in 3rdparty libs
add_subdirectory(dorado/3rdparty)

enable_testing()

add_subdirectory(dorado/compat)
add_subdirectory(dorado/config)
add_subdirectory(dorado/utils)
add_subdirectory(dorado/torch_utils)
add_subdirectory(dorado/models)
add_subdirectory(dorado/model_downloader)
add_subdirectory(dorado/basecall)
add_subdirectory(dorado/modbase)
add_subdirectory(dorado/devtools)
add_subdirectory(dorado/secondary)

add_library(dorado_lib
    dorado/alignment/alignment_info.h
    dorado/alignment/alignment_processing_items.cpp
    dorado/alignment/alignment_processing_items.h
    dorado/alignment/bed_file.cpp
    dorado/alignment/bed_file.h
    dorado/alignment/BedFileAccess.cpp
    dorado/alignment/BedFileAccess.h
    dorado/alignment/IndexFileAccess.cpp
    dorado/alignment/IndexFileAccess.h
    dorado/alignment/minimap2_args.cpp
    dorado/alignment/minimap2_args.h
    dorado/alignment/minimap2_wrappers.h
    dorado/alignment/Minimap2Aligner.cpp
    dorado/alignment/Minimap2Aligner.h
    dorado/alignment/Minimap2Index.cpp
    dorado/alignment/Minimap2Index.h
    dorado/alignment/Minimap2IndexSupportTypes.h
    dorado/alignment/Minimap2Options.cpp
    dorado/alignment/Minimap2Options.h
    dorado/alignment/sam_utils.cpp
    dorado/alignment/sam_utils.h
    dorado/api/caller_creation.cpp
    dorado/api/caller_creation.h
    dorado/api/runner_creation.cpp
    dorado/api/runner_creation.h
    dorado/api/pipeline_creation.cpp
    dorado/api/pipeline_creation.h
    dorado/demux/adapter_primer_kits.cpp
    dorado/demux/adapter_primer_kits.h
    dorado/demux/adapter_info.h
    dorado/demux/AdapterDetector.cpp
    dorado/demux/AdapterDetector.h
    dorado/demux/AdapterDetectorSelector.h
    dorado/demux/AdapterDetectorSelector.cpp
    dorado/demux/BarcodeClassifier.cpp
    dorado/demux/BarcodeClassifier.h
    dorado/demux/BarcodeClassifierSelector.cpp
    dorado/demux/BarcodeClassifierSelector.h
    dorado/demux/barcoding_info.h
    dorado/demux/KitInfoProvider.cpp
    dorado/demux/KitInfoProvider.h
    dorado/demux/Trimmer.cpp
    dorado/demux/Trimmer.h
    dorado/demux/parse_custom_kit.cpp
    dorado/demux/parse_custom_kit.h
    dorado/demux/parse_custom_sequences.cpp
    dorado/demux/parse_custom_sequences.h
    dorado/read_pipeline/AdapterDetectorNode.cpp
    dorado/read_pipeline/AdapterDetectorNode.h
    dorado/read_pipeline/AlignerNode.cpp
    dorado/read_pipeline/AlignerNode.h
    dorado/read_pipeline/BarcodeClassifierNode.cpp
    dorado/read_pipeline/BarcodeClassifierNode.h
    dorado/read_pipeline/BarcodeDemuxerNode.cpp
    dorado/read_pipeline/BarcodeDemuxerNode.h
    dorado/read_pipeline/BasecallerNode.cpp
    dorado/read_pipeline/BasecallerNode.h
    dorado/read_pipeline/BaseSpaceDuplexCallerNode.cpp
    dorado/read_pipeline/BaseSpaceDuplexCallerNode.h
    dorado/read_pipeline/ClientInfo.h
    dorado/read_pipeline/context_container.h
    dorado/read_pipeline/CorrectionInferenceNode.cpp
    dorado/read_pipeline/CorrectionInferenceNode.h
    dorado/read_pipeline/CorrectionMapperNode.cpp
    dorado/read_pipeline/CorrectionMapperNode.h
    dorado/read_pipeline/CorrectionPafReaderNode.cpp
    dorado/read_pipeline/CorrectionPafReaderNode.h
    dorado/read_pipeline/CorrectionPafWriterNode.cpp
    dorado/read_pipeline/CorrectionPafWriterNode.h
    dorado/read_pipeline/DefaultClientInfo.h
    dorado/read_pipeline/DuplexReadTaggingNode.cpp
    dorado/read_pipeline/DuplexReadTaggingNode.h
    dorado/read_pipeline/FakeDataLoader.cpp
    dorado/read_pipeline/FakeDataLoader.h
    dorado/read_pipeline/HtsReader.cpp
    dorado/read_pipeline/HtsReader.h
    dorado/read_pipeline/HtsWriter.cpp
    dorado/read_pipeline/HtsWriter.h
    dorado/read_pipeline/MessageSink.cpp
    dorado/read_pipeline/MessageSink.h
    dorado/read_pipeline/ModBaseCallerNode.cpp
    dorado/read_pipeline/ModBaseCallerNode.h
    dorado/read_pipeline/ModBaseChunkCallerNode.cpp
    dorado/read_pipeline/ModBaseChunkCallerNode.h
    dorado/read_pipeline/NullNode.h
    dorado/read_pipeline/NullNode.cpp
    dorado/read_pipeline/PairingNode.cpp
    dorado/read_pipeline/PairingNode.h
    dorado/read_pipeline/PolyACalculatorNode.cpp
    dorado/read_pipeline/PolyACalculatorNode.h
    dorado/read_pipeline/ProgressTracker.cpp
    dorado/read_pipeline/ProgressTracker.h
    dorado/read_pipeline/ReadFilterNode.cpp
    dorado/read_pipeline/ReadFilterNode.h
    dorado/read_pipeline/ReadForwarderNode.cpp
    dorado/read_pipeline/ReadForwarderNode.h
    dorado/read_pipeline/ReadPipeline.cpp
    dorado/read_pipeline/ReadPipeline.h
    dorado/read_pipeline/ReadSplitNode.cpp
    dorado/read_pipeline/ReadSplitNode.h
    dorado/read_pipeline/ReadToBamTypeNode.cpp
    dorado/read_pipeline/ReadToBamTypeNode.h
    dorado/read_pipeline/ResumeLoader.cpp
    dorado/read_pipeline/ResumeLoader.h
    dorado/read_pipeline/ScalerNode.cpp
    dorado/read_pipeline/ScalerNode.h
    dorado/read_pipeline/StereoDuplexEncoderNode.cpp
    dorado/read_pipeline/StereoDuplexEncoderNode.h
    dorado/read_pipeline/SubreadTaggerNode.cpp
    dorado/read_pipeline/SubreadTaggerNode.h
    dorado/read_pipeline/TrimmerNode.cpp
    dorado/read_pipeline/TrimmerNode.h
    dorado/read_pipeline/messages.cpp
    dorado/read_pipeline/messages.h
    dorado/read_pipeline/flush_options.h
    dorado/read_pipeline/read_output_progress_stats.cpp
    dorado/read_pipeline/read_output_progress_stats.h
    dorado/read_pipeline/read_utils.cpp
    dorado/read_pipeline/read_utils.h
    dorado/read_pipeline/stereo_features.cpp
    dorado/read_pipeline/stereo_features.h
    dorado/read_pipeline/stitch.cpp
    dorado/read_pipeline/stitch.h
    dorado/splitter/DuplexReadSplitter.cpp
    dorado/splitter/DuplexReadSplitter.h
    dorado/splitter/RNAReadSplitter.cpp
    dorado/splitter/RNAReadSplitter.h
    dorado/splitter/myers.cpp
    dorado/splitter/myers.h
    dorado/splitter/splitter_utils.cpp
    dorado/splitter/splitter_utils.h
    dorado/poly_tail/dna_poly_tail_calculator.cpp
    dorado/poly_tail/dna_poly_tail_calculator.h
    dorado/poly_tail/plasmid_poly_tail_calculator.cpp
    dorado/poly_tail/plasmid_poly_tail_calculator.h
    dorado/poly_tail/poly_tail_calculator.cpp
    dorado/poly_tail/poly_tail_calculator.h
    dorado/poly_tail/poly_tail_calculator_selector.cpp
    dorado/poly_tail/poly_tail_calculator_selector.h
    dorado/poly_tail/poly_tail_config.cpp
    dorado/poly_tail/poly_tail_config.h
    dorado/poly_tail/rna_poly_tail_calculator.cpp
    dorado/poly_tail/rna_poly_tail_calculator.h
    dorado/summary/summary.cpp
    dorado/summary/summary.h
    dorado/hts_io/FastxRandomReader.cpp
    dorado/hts_io/FastxRandomReader.h
    dorado/correct/features.cpp
    dorado/correct/features.h
    dorado/correct/windows.cpp
    dorado/correct/windows.h
    dorado/correct/conversions.cpp
    dorado/correct/conversions.h
    dorado/correct/decode.cpp
    dorado/correct/decode.h
    dorado/correct/infer.cpp
    dorado/correct/infer.h
    dorado/correct/types.cpp
    dorado/correct/types.h
    dorado/correct/CorrectionProgressTracker.cpp
    dorado/correct/CorrectionProgressTracker.h
    dorado/polish/polish_impl.cpp
    dorado/polish/polish_impl.h
    dorado/polish/polish_progress_tracker.cpp
    dorado/polish/polish_progress_tracker.h
    dorado/polish/polish_stats.h
)

enable_warnings_as_errors(dorado_lib)

set_target_properties(dorado_lib
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_include_directories(dorado_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/dorado
)
# 3rdparty libs should be considered SYSTEM headers
target_include_directories(dorado_lib
    SYSTEM
    PUBLIC
        ${DORADO_3RD_PARTY_SOURCE}/hdf_plugins/vbz_plugin
        ${DORADO_3RD_PARTY_SOURCE}/cxxpool/src
        ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
        ${DORADO_3RD_PARTY_SOURCE}/indicators/include
        ${DORADO_3RD_PARTY_SOURCE}/intervaltree
)

enable_testing()

if(NOT SKIP_HDF_PLUGINS)
  # Can skip this if we're consuming the lib from a parent project that already has hdf_plugins
  add_hdf_vbz_plugin()
endif()

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_lib REUSE_FROM dorado_torch_utils)
endif()

target_link_libraries(dorado_lib
    PUBLIC
        torch_lib
        htslib
        edlib
        dorado_basecall
        dorado_modbase
        dorado_version
        dorado_secondary
    PRIVATE
        minimap2
        toml11::toml11
)

if(NOT WIN32)
    add_dependencies(dorado_lib htslib_project)
endif()

if(NOT DORADO_DISABLE_DORADO)
    add_executable(dorado
        dorado/main.cpp
        dorado/cli/aligner.cpp
        dorado/cli/basecall_output_args.cpp
        dorado/cli/basecall_output_args.h
        dorado/cli/demux.cpp
        dorado/cli/duplex.cpp
        dorado/cli/trim.cpp
        dorado/cli/basecaller.cpp
        dorado/cli/download.cpp
        dorado/cli/summary.cpp
        dorado/cli/correct.cpp
        dorado/cli/polish.cpp
        dorado/cli/cli.h
        dorado/cli/cli_utils.h
        dorado/cli/model_resolution.h
     )

    set_target_properties(dorado
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    if(MSVC)
        set_property(TARGET dorado APPEND PROPERTY LINK_OPTIONS "/ignore:4099")
    endif()

    enable_warnings_as_errors(dorado)
    check_linked_libs(dorado)

    if (DORADO_ENABLE_PCH)
        target_precompile_headers(dorado REUSE_FROM dorado_lib)
    endif()

    target_link_libraries(dorado
        dorado_lib
        dorado_io_lib
        dorado_file_info
        dorado_models_lib
        dorado_model_downloader_lib
        dorado_compat
        minimap2
    )

    install(TARGETS dorado RUNTIME)
endif()

add_subdirectory(dorado/data_loader)
add_subdirectory(dorado/file_info)
add_subdirectory(tests)

include(cmake/InstallRedistLibs.cmake)
if(NOT DORADO_DISABLE_PACKAGING)
    include(cmake/DoradoPackaging.cmake)
endif()

# For capturing test coverage.
if (GENERATE_TEST_COVERAGE)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" AND NOT DORADO_DISABLE_TESTS)
        include(cmake/CodeCoverage.cmake)
        foreach(COVERAGE_TARGET dorado_tests dorado_lib dorado_io_lib dorado_models_lib dorado_torch_utils dorado_utils dorado_secondary dorado_file_info)
            append_coverage_compiler_flags_to_target(${COVERAGE_TARGET})
        endforeach()
        setup_target_for_coverage_gcovr_html(
            NAME dorado_test_coverage
            EXECUTABLE ctest --test-dir ${CMAKE_BINARY_DIR} --verbose -R dorado_tests
            DEPENDENCIES dorado_tests
            BASE_DIRECTORY "${PROJECT_SOURCE_DIR}"
            EXCLUDE
                "${DORADO_3RD_PARTY_SOURCE}/*"
                "${DORADO_3RD_PARTY_DOWNLOAD}/*"
                "${PROJECT_SOURCE_DIR}/tests/*"
                "${CMAKE_BINARY_DIR}/3rdparty/*"
            )
    endif()
endif()

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(dorado/read_pipeline/stereo_features.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()
