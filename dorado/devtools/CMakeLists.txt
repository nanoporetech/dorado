
# Note that compilation of this target doesn't automatically enable it. Targets
# that want to make use of this feature must also add scopes where relevant.
option(COMPILE_SLOW_INTERPOSE
    "Compile an interpose lib that slows down threading primitives" OFF)
if (COMPILE_SLOW_INTERPOSE)
    add_library(dorado_slow_interpose slow_interpose.cpp slow_interpose.h)
    target_link_libraries(dorado_slow_interpose PUBLIC ${CMAKE_DL_LIBS})
    enable_warnings_as_errors(dorado_slow_interpose)
    if (LINUX)
        target_link_options(dorado_slow_interpose
            PUBLIC
                -Wl,--wrap=_ZNSt18condition_variable10notify_oneEv
                -Wl,--wrap=_ZNSt18condition_variable10notify_allEv
                -Wl,--wrap=_ZNSt5mutex6unlockEv
        )
    endif()
endif()
