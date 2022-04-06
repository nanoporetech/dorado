set(SOURCE_FILES main.cpp
                 Fast5DataLoaderTest.cpp)

add_executable(dorado_tests ${SOURCE_FILES})

target_link_libraries(dorado_tests dorado_lib)
