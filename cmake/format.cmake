# Formatting files in source directory

set(DORADO_SOURCE_FOLDER "dorado")

function(dorado_format_all_targets)
    find_program(CLANGFORMAT_EXECUTABLE NAMES clang-format)
    if(CLANGFORMAT_EXECUTABLE)
        set(format_list "")
        
        file(GLOB_RECURSE DORADO_FILE_SOURCES LIST_DIRECTORIES false "${DORADO_SOURCE_FOLDER}/*.h"
                                                                     "${DORADO_SOURCE_FOLDER}/*.cpp"
                                                                     "${DORADO_SOURCE_FOLDER}/*.cu"
                                                                     "${DORADO_SOURCE_FOLDER}/*.cuh")

        list(FILTER DORADO_FILE_SOURCES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/${DORADO_SOURCE_FOLDER}/3rdparty")
        
        foreach(source IN LISTS DORADO_FILE_SOURCES)
            file(RELATIVE_PATH target ${PROJECT_SOURCE_DIR} ${source})
            add_custom_command(OUTPUT clang-format/${target} COMMAND "${CLANGFORMAT_EXECUTABLE}" "-style=file" "-i" ${source})
            list(APPEND format_list clang-format/${target})
        endforeach()

        add_custom_target(format DEPENDS ${format_list} COMMENT "Finished formatting all source files")
    else()
        message(WARNING "> warning: skipping auto-formatting. clang-format not found.")
    endif()

endfunction()
