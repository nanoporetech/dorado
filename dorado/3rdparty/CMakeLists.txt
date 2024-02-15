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

# edlib
set(BUILD_TESTING OFF)
add_subdirectory(edlib EXCLUDE_FROM_ALL)
# Disable warnings from edlib source
disable_warnings(edlib)
set(BUILD_TESTING ON)

# date
add_subdirectory(date EXCLUDE_FROM_ALL)
