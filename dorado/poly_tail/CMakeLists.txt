add_library(dorado_poly_tail STATIC
    # Public
    include/poly_tail/poly_tail_calculator_selector.h
    include/poly_tail/poly_tail_calculator.h
    include/poly_tail/poly_tail_config.h

    # Private
    dna_poly_tail_calculator.cpp
    dna_poly_tail_calculator.h
    plasmid_poly_tail_calculator.cpp
    plasmid_poly_tail_calculator.h
    poly_tail_calculator_selector.cpp
    poly_tail_calculator.cpp
    poly_tail_config.cpp
    rna_poly_tail_calculator.cpp
    rna_poly_tail_calculator.h
)

target_link_libraries(dorado_poly_tail
    PUBLIC
        dorado_utils
    PRIVATE
        dorado_models_lib # TODO: remove this once pipeline target exists
        dorado_torch_utils
        edlib
        spdlog::spdlog
        toml11::toml11
)

target_include_directories(dorado_poly_tail
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/.. # TODO: remove this once pipeline target exists
)

enable_warnings_as_errors(dorado_poly_tail)
