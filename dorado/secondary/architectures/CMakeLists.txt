add_library(dorado_secondary_architectures
    # Public
    include/secondary/architectures/model_torch_base.h
    include/secondary/architectures/model_factory.h
    include/secondary/architectures/model_config.h

    # Private
    model_config.cpp
    model_factory.cpp
    model_gru.cpp
    model_gru.h
    model_latent_space_lstm.cpp
    model_latent_space_lstm.h
    model_slot_attention_consensus.cpp
    model_slot_attention_consensus.h
    model_torch_base.cpp
    model_torch_script.cpp
    model_torch_script.h
)

target_include_directories(dorado_secondary_architectures
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dorado_secondary_architectures
    PUBLIC
        dorado_torch_utils
        dorado_utils
    PRIVATE
        spdlog::spdlog
        toml11::toml11
)

enable_warnings_as_errors(dorado_secondary_architectures)
