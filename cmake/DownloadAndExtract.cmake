# Helper function to extract the specified URL to the given 3rd party folder if it doesn't already exist

function(download_and_extract url name)

    file(LOCK ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

    if(EXISTS ${DORADO_3RD_PARTY}/${name})
        message(STATUS "Found ${name}")
    else()
        message(STATUS "Downloading ${name} from ${url}")
        file(DOWNLOAD ${url} ${DORADO_3RD_PARTY}/${name}.zip STATUS RESULT)
        list(GET RESULT 0 STATUS_CODE)
        list(GET RESULT 1 ERROR_MSG)
        if (NOT ${STATUS_CODE} EQUAL 0)
            message(FATAL_ERROR "Failed to download ${name}: ${ERROR_MSG}")
        endif()
        message(STATUS "Downloading ${name} - done")
        message(STATUS "Extracting ${name}")
        file(ARCHIVE_EXTRACT INPUT ${DORADO_3RD_PARTY}/${name}.zip DESTINATION ${DORADO_3RD_PARTY}/${name})
        file(REMOVE ${DORADO_3RD_PARTY}/${name}.zip)
        message(STATUS "Extracting ${name} - done")
    endif()

    file(LOCK ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY RELEASE)

endfunction()
