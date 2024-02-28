function(enable_warnings_as_errors TARGET_NAME)
    if(WIN32)
        # W4 - warning level 4.
        # WX - warnings as errors.
        # external:anglebrackets - treat <header> includes as external.
        # external:W0 - disable warnings for external code.
        target_compile_options(${TARGET_NAME} PRIVATE
          $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/W4 /WX /external:anglebrackets /external:W0>
        )
        target_compile_definitions(${TARGET_NAME} PRIVATE
            _CRT_SECURE_NO_WARNINGS
        )
    else
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Werror -Wundef)
    endif()
endfunction()

function(disable_warnings TARGET_NAME)
    if(WIN32)
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/W0>
        )
    endif()
endfunction()
