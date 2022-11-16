set(SOURCE_FILES 
    main.cpp
    LinearTanh.cpp
    )

add_executable(dorado_metal_tests ${SOURCE_FILES})

target_link_libraries(dorado_metal_tests dorado_lib)

add_test(
    NAME dorado_metal_tests
    COMMAND dorado_metal_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
