download_and_extract(https://developer.apple.com/metal/cpp/files/metal-cpp_macOS13_iOS16.zip metal-cpp)

set(AIR_FILES)
set(METAL_SOURCES dorado/basecall/metal/nn.metal)

if (IOS)
    set(XCRUN_SDK ${SDK_NAME})
else()
    set(XCRUN_SDK macosx)
endif()

foreach(source ${METAL_SOURCES})
    get_filename_component(basename "${source}" NAME_WE)
    set(air_path "${CMAKE_BINARY_DIR}/${basename}.air")
    add_custom_command(
        OUTPUT "${air_path}"
        COMMAND
            xcrun -sdk ${XCRUN_SDK} metal
                -Werror
                -Wall -Wextra -pedantic
                -Wno-c++17-attribute-extensions # [[maybe_unused]] is C++17
                -ffast-math
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
        xcrun -sdk ${XCRUN_SDK} metallib
            ${AIR_FILES}
            -o ${CMAKE_BINARY_DIR}/lib/default.metallib
    DEPENDS ${AIR_FILES}
    COMMENT "Creating metallib"
)

add_custom_target(metal-lib DEPENDS default.metallib)

install(FILES ${CMAKE_BINARY_DIR}/lib/default.metallib DESTINATION lib)
