add_library(dorado_secondary_architectures
    model_config.cpp
    model_config.h
    model_factory.cpp
    model_factory.h
    model_gru.cpp
    model_gru.h
    model_latent_space_lstm.cpp
    model_latent_space_lstm.h
    model_slot_attention_consensus.cpp
    model_slot_attention_consensus.h
    model_torch_base.cpp
    model_torch_base.h
    model_torch_script.cpp
    model_torch_script.h
)

target_include_directories(dorado_secondary_architectures
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/architectures
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
