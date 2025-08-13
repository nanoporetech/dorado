
# Helper to apply patches to 3rdparty libs.
function (apply_3rdparty_patch NAME)
    find_package(Git REQUIRED QUIET)
    set(PATCH_FILE ${CMAKE_CURRENT_LIST_DIR}/patches/${NAME}.patch)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${PATCH_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${NAME}
        RESULT_VARIABLE NOT_ALREADY_APPLIED
        ERROR_QUIET
    )
    if (NOT_ALREADY_APPLIED)
        message(STATUS "Applying patch ${PATCH_FILE}")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply ${PATCH_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${NAME}
            RESULT_VARIABLE PATCH_FAILED
        )
        if (PATCH_FAILED)
            message(FATAL_ERROR "Failed to apply patch ${PATCH_FILE}")
        endif()
    endif()
endfunction()



# argparse
add_subdirectory(argparse EXCLUDE_FROM_ALL)

# catch2
add_subdirectory(catch2 EXCLUDE_FROM_ALL)

# CxxUrl
add_subdirectory(CxxUrl EXCLUDE_FROM_ALL)

# cpp-httplib
add_library(httplib INTERFACE)
target_include_directories(httplib SYSTEM INTERFACE
    ${DORADO_3RD_PARTY_SOURCE}/cpp-httplib
)

# cxxpool
apply_3rdparty_patch(cxxpool)
add_library(cxxpool INTERFACE)
target_include_directories(cxxpool SYSTEM INTERFACE
    ${DORADO_3RD_PARTY_SOURCE}/cxxpool/src
)

# Bring in spdlog
add_subdirectory(spdlog EXCLUDE_FROM_ALL)
# Avoid namespace clashes with static torch.
target_compile_definitions(spdlog PUBLIC
    "FMT_BEGIN_NAMESPACE=namespace fmt { inline namespace ont {"
    "FMT_END_NAMESPACE=}}"
)

# ELZIP_DECOMPRESS_ONLY stops minizip from adding OpenSSL as a target, preventing use of three dylibs on osx.
set(ELZIP_DECOMPRESS_ONLY ON)
add_subdirectory(elzip EXCLUDE_FROM_ALL)

# indicators
add_library(indicators INTERFACE)
target_include_directories(indicators SYSTEM INTERFACE
    ${DORADO_3RD_PARTY_SOURCE}/indicators/include
)

# intervaltree
add_library(intervaltree INTERFACE)
target_include_directories(intervaltree SYSTEM INTERFACE
    ${DORADO_3RD_PARTY_SOURCE}/intervaltree
)

# minimap
option(ONT_MM2_EXE "Build minimap2 exe" OFF)
add_subdirectory(ont-minimap2 EXCLUDE_FROM_ALL)
message(STATUS "build minimap2 exe: ${ONT_MM2_EXE}")
message(STATUS "build minimap2 lib: ${ONT_MM2_LIB}")
if (ONT_MM2_EXE)
    install(TARGETS minimap2_exe)
    add_custom_target(testing_deps ALL DEPENDS minimap2_exe)
endif()
# Disable warnings from minimap source
disable_warnings(minimap2)

# NVTX
add_library(nvtx INTERFACE)
target_include_directories(nvtx SYSTEM INTERFACE
    ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
)

# edlib
apply_3rdparty_patch(edlib)
set(BUILD_TESTING OFF)
add_subdirectory(edlib EXCLUDE_FROM_ALL)
# Disable warnings from edlib source
disable_warnings(edlib)
set(BUILD_TESTING ON)

# date
add_subdirectory(date EXCLUDE_FROM_ALL)

# toml11
set(TOML11_PRECOMPILE ON)
add_subdirectory(toml11 EXCLUDE_FROM_ALL)

# kadayashi
set(BUILD_KADAYASHI_EXE OFF)
add_subdirectory(kadayashi EXCLUDE_FROM_ALL)
