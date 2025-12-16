# Setup metal-cpp target.
download_and_extract(
    https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
    metal-cpp_26
    "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4"
)
add_library(metal_cpp STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../dorado/metal-cpp-impl.cpp
)
target_include_directories(metal_cpp SYSTEM
    PUBLIC
        ${DORADO_3RD_PARTY_DOWNLOAD}/metal-cpp_26/metal-cpp
)



# Build metal sources.
set(AIR_FILES)
set(METAL_SOURCES dorado/nn/metal/nn.metal)

set(XCRUN_SDK macosx)
set(METAL_STD_VERSION "macos-metal2.3") # macOS 11.0
string(TOUPPER ${XCRUN_SDK} XCRUN_SDK_UPPER)

foreach(source ${METAL_SOURCES})
    get_filename_component(basename "${source}" NAME_WE)
    set(air_path "${CMAKE_BINARY_DIR}/${basename}.air")
    add_custom_command(
        OUTPUT "${air_path}"
        COMMAND
            ${CMAKE_COMMAND} -E env
                ${XCRUN_SDK_UPPER}_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            xcrun -sdk ${XCRUN_SDK} metal
                -Werror
                -Wall -Wextra -pedantic
                -Wno-c++17-extensions # [[maybe_unused]] is C++17
                -std=${METAL_STD_VERSION}
                -gline-tables-only -frecord-sources # embed source for trace analysis
                -O2 -ffast-math
                -c "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
                -o "${air_path}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
        COMMENT "Compiling metal kernels"
    )
    list(APPEND AIR_FILES "${air_path}")
endforeach()

add_custom_command(
    OUTPUT default.metallib
    COMMAND
        ${CMAKE_COMMAND} -E env
            ${XCRUN_SDK_UPPER}_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
        xcrun -sdk ${XCRUN_SDK} metallib
            ${AIR_FILES}
            -o ${CMAKE_BINARY_DIR}/lib/default.metallib
    DEPENDS ${AIR_FILES}
    COMMENT "Creating metallib"
)

add_custom_target(metal-lib DEPENDS default.metallib)

install(FILES ${CMAKE_BINARY_DIR}/lib/default.metallib DESTINATION lib)
