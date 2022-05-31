# Helper function to extract the specified URL to the given 3rd party folder if it doesn't already exist

function(download_and_extract url name)

    file(LOCK ${CMAKE_SOURCE_DIR} DIRECTORY)

    if(EXISTS ${DORADO_3RD_PARTY}/${name})
        message("-- Found ${name}")
    else()
        message("-- Downloading ${name}")
        file(DOWNLOAD ${url} ${DORADO_3RD_PARTY}/${name}.zip)
        message("-- Downloading ${name} - done")
        message("-- Extracting ${name}")
        file(ARCHIVE_EXTRACT INPUT ${DORADO_3RD_PARTY}/${name}.zip DESTINATION ${DORADO_3RD_PARTY}/${name})
        file(REMOVE ${DORADO_3RD_PARTY}/${name}.zip)
        message("-- Extracting ${name} - done")
    endif()

    file(LOCK ${CMAKE_SOURCE_DIR} DIRECTORY RELEASE)

endfunction()
