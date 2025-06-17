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

# All static libraries should be position-independent, as they are liable to be compiled into shared
# libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set warnings and RPATH before we create any targets.
include(cmake/Warnings.cmake)
include(cmake/SetRPATH.cmake)

set(DORADO_3RD_PARTY_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/dorado/3rdparty)
set(DORADO_3RD_PARTY_DOWNLOAD ${CMAKE_CURRENT_BINARY_DIR}/download
    CACHE PATH "Location to download 3rdparty libraries into")

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

# Repo setup and helpers.
include(cmake/DoradoVersion.cmake)
include(cmake/UpdateSubmodules.cmake)
include(cmake/DownloadAndExtract.cmake)
include(cmake/SharedLibHelpers.cmake)
include(cmake/DoradoTarget.cmake)
include(cmake/DoradoCheckDeps.cmake)

# Add sanitizer options to compilation flags
include(cmake/ECMEnableSanitizers.cmake)
if (ECM_ENABLE_SANITIZERS)
    # Always emit debug info to provide better stack traces
    add_compile_options(-g)
    # Silence overeager GCC warnings that only show up when sanitizers are enabled.
    if ((CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0))
        add_compile_options(-Wno-error=array-bounds)
    endif()
endif()

if(DORADO_USING_OLD_CPP_ABI)
  # We need to force the use of the old ABI here, if we are building in an old ABI context, as otherwise elzip builds
  # with the libc++11 ABI and we can't link against it.
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

# For capturing test coverage.
option(GENERATE_TEST_COVERAGE "Build with code coverage enabled" OFF)
if (GENERATE_TEST_COVERAGE)
    include(cmake/CodeCoverage.cmake)
    if (NOT DORADO_DISABLE_TESTS)
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

set(CHOCO_CCACHE_HINT)
file(GLOB CHOCO_CACHE_LIST LIST_DIRECTORIES true "C:/ProgramData/chocolatey/lib/ccache/tools/ccache-*")
if (CHOCO_CACHE_LIST)
    list(SORT CHOCO_CACHE_LIST)
    list(GET CHOCO_CACHE_LIST -1 CHOCO_CCACHE_HINT)
endif()

# Use ccache for C and C++ if it's available
find_program(CCACHE_EXE ccache
    HINTS
        "C:/Program\ Files/ccache/"
        ${CHOCO_CCACHE_HINT}
)
option(DORADO_DISABLE_CCACHE "Explicitly disable the use of ccache" OFF)
if (CCACHE_EXE AND NOT DORADO_DISABLE_CCACHE)
    if(MSVC)
        # See https://github.com/ccache/ccache/wiki/MS-Visual-Studio#usage-with-cmake
        file(COPY_FILE
            ${CCACHE_EXE} ${CMAKE_BINARY_DIR}/cl.exe
            ONLY_IF_DIFFERENT
        )

        # By default Visual Studio generators will use /Zi which is not compatible
        # with ccache, so tell Visual Studio to use /Z7 instead.
        message(STATUS "Setting MSVC debug information format to 'Embedded'")
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")

        set(CMAKE_VS_GLOBALS
            "CLToolExe=cl.exe"
            "CLToolPath=${CMAKE_BINARY_DIR}"
            "UseMultiToolTask=true"
            "DebugInformationFormat=OldStyle"
        )
    else()
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_EXE})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXE})
    endif()
    message(STATUS "Using ccache at: ${CCACHE_EXE}")
    set(DORADO_ENABLE_PCH FALSE)
else()
    # Fallback to using a PCH if we don't have ccache support, since making them work together isn't simple.
    set(DORADO_ENABLE_PCH TRUE)
endif()

# By default, make per-read trace-logging a no-op. Turn this option on to enable it.
option(DORADO_PER_READ_TRACE "Enable per-read trace logging" OFF)

# Bring in 3rdparty libs
include(cmake/Zlib.cmake)
include(cmake/Koi.cmake)
include(cmake/Pod5.cmake)
include(cmake/Torch.cmake)
include(cmake/OpenSSL.cmake)
include(cmake/Htslib.cmake)
if (APPLE)
    include(cmake/Metal.cmake)
endif()
add_subdirectory(dorado/3rdparty)

enable_testing()

# CMake can't REUSE_FROM targets that it doesn't know about, so include
# torch_utils first for its PCH.
add_subdirectory(dorado/torch_utils)

add_subdirectory(dorado/aligner)
add_subdirectory(dorado/alignment)
add_subdirectory(dorado/api)
add_subdirectory(dorado/basecall)
add_subdirectory(dorado/compat)
add_subdirectory(dorado/config)
add_subdirectory(dorado/correct)
add_subdirectory(dorado/data_loader)
add_subdirectory(dorado/demux)
add_subdirectory(dorado/devtools)
add_subdirectory(dorado/file_info)
add_subdirectory(dorado/hts_io)
add_subdirectory(dorado/modbase)
add_subdirectory(dorado/model_downloader)
add_subdirectory(dorado/models)
add_subdirectory(dorado/nn)
add_subdirectory(dorado/polish)
add_subdirectory(dorado/poly_tail)
add_subdirectory(dorado/read_pipeline)
add_subdirectory(dorado/resume_loader)
add_subdirectory(dorado/secondary)
add_subdirectory(dorado/splitter)
add_subdirectory(dorado/summary)
add_subdirectory(dorado/utils)
add_subdirectory(tests)

if (NOT DORADO_DISABLE_DORADO)
    add_subdirectory(dorado/cli)
endif()

# Stopgap target to keep things linking.
# Eventually this will be removed and dependents should link only
# to what they need.
add_library(dorado_lib INTERFACE)
target_link_libraries(dorado_lib
    INTERFACE
        dorado_aligner
        dorado_alignment
        dorado_api
        dorado_basecall
        dorado_correct
        dorado_demux
        dorado_hts_io
        dorado_modbase
        dorado_poly_tail
        dorado_read_pipeline
        dorado_resume_loader
        dorado_secondary
        dorado_splitter
        dorado_version
)

# Packaging setup.
include(cmake/InstallRedistLibs.cmake)
if(NOT DORADO_DISABLE_PACKAGING)
    include(cmake/DoradoPackaging.cmake)
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0") )
    find_library(
        LIBSTDCPP_LINK_LOCATION
        NAMES "libstdc++.so.6"
        HINTS /usr/lib/x86_64-linux-gnu/
    )

    if (NOT LIBSTDCPP_LINK_LOCATION)
        message(FATAL_ERROR "Could not find libstdc++")
    endif()

    # Follow the softlink from LIBSTDCPP_LINK_LOCATION to the real library
    execute_process(
        COMMAND "readlink" "-e" "${LIBSTDCPP_LINK_LOCATION}"
        OUTPUT_VARIABLE LIBSTDCPP_LOCATION
        RESULT_VARIABLE LIBSTDCPP_READLINK_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT LIBSTDCPP_READLINK_RESULT EQUAL 0)
        message(
            FATAL_ERROR
            "Unable to follow ${LIBSTDCPP_LINK_LOCATION} to a real file, return-code ${LIBSTDCPP_READLINK_RESULT}"
        )
    endif ()

    install(
        FILES ${LIBSTDCPP_LOCATION} ${LIBSTDCPP_LINK_LOCATION}
        DESTINATION lib
    )
endif()
