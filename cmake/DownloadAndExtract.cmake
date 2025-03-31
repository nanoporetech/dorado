# Helper function to extract the specified URL to the given 3rd party folder if it doesn't already exist

function(download_and_extract url name sha256)
    file(LOCK ${DORADO_3RD_PARTY_DOWNLOAD} DIRECTORY GUARD FUNCTION)

    if(EXISTS ${DORADO_3RD_PARTY_DOWNLOAD}/${name})
        message(STATUS "Found ${name}")
    else()
        message(STATUS "Downloading ${name} from ${url}")
        file(
            DOWNLOAD ${url} ${DORADO_3RD_PARTY_DOWNLOAD}/${name}.zip
            STATUS RESULT
            EXPECTED_HASH SHA256=${sha256}
        )
        list(GET RESULT 0 STATUS_CODE)
        list(GET RESULT 1 ERROR_MSG)
        if (NOT ${STATUS_CODE} EQUAL 0)
            message(FATAL_ERROR "Failed to download ${name}: ${ERROR_MSG}")
        endif()
        message(STATUS "Downloading ${name} - done")
        message(STATUS "Extracting ${name}")
        file(ARCHIVE_EXTRACT INPUT ${DORADO_3RD_PARTY_DOWNLOAD}/${name}.zip DESTINATION ${DORADO_3RD_PARTY_DOWNLOAD}/${name})
        file(REMOVE ${DORADO_3RD_PARTY_DOWNLOAD}/${name}.zip)
        message(STATUS "Extracting ${name} - done")
    endif()
endfunction()
