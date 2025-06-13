#
# Helpers to check that a target doesn't link to a specific one.
#

# Generic checker.
function(check_no_dependency_for_target TARGET CHECKER DEPENDENCY_PATH)
    # Keep track of how we got here.
    list(APPEND DEPENDENCY_PATH "${TARGET}")

    # Grab all the dependencies and apply the CHECKER function.
    get_target_property(all_deps "${TARGET}" LINK_LIBRARIES)
    foreach(lib_name IN LISTS all_deps)
        CHECKER("${lib_name}" "${DEPENDENCY_PATH}")

        # Check dependencies.
        if (TARGET "${lib_name}")
            check_no_dependency_for_target("${lib_name}" CHECKER "${DEPENDENCY_PATH}")
        endif()
    endforeach()
endfunction()

# Check that we don't link to torch.
function(check_no_dependency_on_torch TARGET)
    function(checker LIB_NAME DEPENDENCY_PATH)
        # If it has torch in the name then it's likely going to be or link to torch.
        string(TOLOWER "${LIB_NAME}" lib_lower)
        if (lib_lower MATCHES "torch")
            list(APPEND DEPENDENCY_PATH "${LIB_NAME}")
            list(JOIN DEPENDENCY_PATH " -> " dep_path)
            message(FATAL_ERROR "Target ${TARGET} has dependency on torch: ${dep_path}")
        endif()
    endfunction()

    check_no_dependency_for_target("${TARGET}" checker "")
endfunction()
