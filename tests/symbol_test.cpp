// These are all of the headers that ont_core currently makes use of.
// TODO: once the public API exists this should only include the public API.
#include "alignment/IndexFileAccess.h"
#include "alignment/Minimap2Options.h"
#include "api/caller_creation.h"
#include "api/pipeline_creation.h"
#include "api/runner_creation.h"
#include "basecall/ModelRunner.h"
#include "basecall/crf_utils.h"
#include "config/BasecallModelConfig.h"
#include "config/ModBaseModelConfig.h"
#include "demux/parse_custom_kit.h"
#include "demux/parse_custom_sequences.h"
#include "modbase/ModBaseRunner.h"
#include "models/models.h"
#include "read_pipeline/AdapterDetectorNode.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/BarcodeClassifierNode.h"
#include "read_pipeline/ClientInfo.h"
#include "read_pipeline/ReadForwarderNode.h"
#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/SubreadTaggerNode.h"
#include "read_pipeline/read_utils.h"
#include "torch_utils/gpu_monitor.h"
#include "torch_utils/torch_utils.h"
#include "utils/barcode_kits.h"
#include "utils/parameters.h"
#include "utils/sequence_utils.h"
#include "utils/string_utils.h"
#include "utils/time_utils.h"
#include "utils/uuid_utils.h"

#if DORADO_METAL_BUILD
#include "torch_utils/metal_utils.h"
#elif DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"
#endif

#if defined(_MSC_VER)
#define DORADO_EXPORT __declspec(dllexport)
#else
#define DORADO_EXPORT __attribute__((visibility("default")))
#endif

namespace {

template <typename T>
void force_reference(T* sym) {
    // Storing to volatile ensures the value is read, which forces a reference.
    volatile auto ptr = sym;
    (void)ptr;
}

template <typename Obj, typename T>
void force_reference(T Obj::* sym) {
    // Storing to volatile ensures the value is read, which forces a reference.
    volatile auto ptr = sym;
    (void)ptr;
}

}  // namespace

DORADO_EXPORT void reference_all_public_functions();
DORADO_EXPORT void reference_all_public_functions() {
    // Reference a few functions in the public API so that we make sure that
    // their dependencies are also linked in. We'll get a linker error if we're
    // missing something in the dependency chain.
    //
    // TODO: Can we generate this file so that we capture a reference to every
    // TODO: function and constant in the public API?

    // alignment/IndexFileAccess.h
    force_reference(&dorado::alignment::validate_options);
    // alignment/Minimap2Options.h
    force_reference(&dorado::alignment::create_dflt_options);
    // api/caller_creation.h
    force_reference(&dorado::api::create_modbase_caller);
    // api/pipeline_creation.h
    force_reference(&dorado::api::create_simplex_pipeline);
    // api/runner_creation.h
    force_reference(&dorado::api::create_basecall_runners);
    // basecall/ModelRunner.h
    force_reference(&dorado::basecall::ModelRunner::accept_chunk);
    // config/BasecallModelConfig.h
    force_reference(&dorado::config::load_model_config);
    // demux/parse_custom_sequences.h
    force_reference(&dorado::demux::parse_custom_sequences);
    // modbase/ModBaseRunner.h
    force_reference(&dorado::modbase::ModBaseRunner::accept_chunk);
    // models/models.h
    force_reference(&dorado::models::find_model);
    // read_pipeline/AdapterDetectorNode.h
    force_reference(&dorado::AdapterDetectorNode::get_name);
    // read_pipeline/AlignerNode.h
    force_reference(&dorado::AlignerNode::get_name);
    // read_pipeline/BarcodeClassifierNode.h
    force_reference(&dorado::BarcodeClassifierNode::get_name);
    // read_pipeline/ClientInfo.h
    force_reference(&dorado::ClientInfo::is_disconnected);
    // read_pipeline/ReadForwarderNode.h
    force_reference(&dorado::ReadForwarderNode::get_name);
    // read_pipeline/ReadPipeline.h
    force_reference(&dorado::Pipeline::create);
    // read_pipeline/SubreadTaggerNode.h
    force_reference(&dorado::SubreadTaggerNode::get_name);
    // read_pipeline/read_utils.h
    force_reference(&dorado::utils::shallow_copy_read);
    // utils/barcode_kits.h
    force_reference(&dorado::barcode_kits::get_kit_infos);
    // demux/parse_custom_kit.h
    force_reference(&dorado::demux::parse_custom_arrangement);

#if DORADO_CUDA_BUILD
    // torch_utils/cuda_utils.h
    force_reference(&dorado::utils::acquire_gpu_lock);
#endif
    // torch_utils/gpu_monitor.h
    force_reference(&dorado::utils::gpu_monitor::get_device_count);
#if DORADO_METAL_BUILD
    // torch_utils/metal_utils.h
    force_reference(&dorado::utils::create_buffer);
#endif
    // utils/parameters.h
    force_reference(&dorado::utils::default_thread_allocations);
    // utils/sequence_utils.h
    force_reference(&dorado::utils::find_rna_polya);
    // utils/string_utils.h
    force_reference(&dorado::utils::split);
    // utils/time_utils.h
    force_reference(&dorado::utils::get_string_timestamp_from_unix_time);
    // torch_utils/torch_utils.h
    force_reference(&dorado::utils::make_torch_deterministic);
    // utils/uuid_utils.h
    force_reference(&dorado::utils::derive_uuid);
}
