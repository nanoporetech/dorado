#include "read_pipeline/BarcodeDemuxerNode.h"

#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/HtsReader.h"
#include "utils/SampleSheet.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <htslib/sam.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

#define TEST_GROUP "[barcode_demux]"

namespace fs = std::filesystem;

using namespace dorado;

namespace {
std::vector<BamPtr> create_bam_reader(const std::string& bc) {
    ReadCommon read_common;
    read_common.seq = "AAAA";
    read_common.qstring = "!!!!";
    read_common.read_id = bc;
    auto records = read_common.extract_sam_lines(false, std::nullopt, false);
    for (auto& rec : records) {
        bam_aux_append(rec.get(), "BC", 'Z', int(bc.length() + 1), (uint8_t*)bc.c_str());
    }
    return records;
}
}  // namespace

CATCH_TEST_CASE("BarcodeDemuxerNode: check correct output files are created", TEST_GROUP) {
    using Catch::Matchers::Contains;

    auto tmp_dir = make_temp_dir("dorado_demuxer");

    {
        // Creating local scope for the pipeline because on windows
        // the temporary directory is still being considered open unless
        // the pipeline object is closed. This needs to be looked at.
        // TODO: Address open file issue on windows.
        dorado::PipelineDescriptor pipeline_desc;
        auto demuxer = pipeline_desc.add_node<BarcodeDemuxerNode>({}, tmp_dir.m_path.string(), 8,
                                                                  false, nullptr, true);

        auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

        SamHdrPtr hdr(sam_hdr_init());
        sam_hdr_add_line(hdr.get(), "SQ", "ID", "foo", "LN", "100", "SN", "ref", NULL);

        auto& demux_writer_ref = pipeline->get_node_ref<BarcodeDemuxerNode>(demuxer);
        demux_writer_ref.set_header(hdr.get());

        auto client_info = std::make_shared<dorado::DefaultClientInfo>();
        for (auto bc : {"bc01", "bc02", "bc03"}) {
            auto records = create_bam_reader(bc);
            for (auto& rec : records) {
                pipeline->push_message(BamMessage{std::move(rec), client_info});
            }
        }

        pipeline->terminate(DefaultFlushOptions());

        demux_writer_ref.finalise_hts_files([](size_t) { /* noop */ });

        const std::unordered_set<std::string> expected_files = {
                "unknown_run_id_bc01.bam", "unknown_run_id_bc01.bam.bai",
                "unknown_run_id_bc02.bam", "unknown_run_id_bc02.bam.bai",
                "unknown_run_id_bc03.bam", "unknown_run_id_bc03.bam.bai",
        };

        std::unordered_set<std::string> actual_files;

        for (const auto& entry : fs::directory_iterator(tmp_dir.m_path)) {
            actual_files.insert(entry.path().filename().string());
        }

        for (const auto& expected : expected_files) {
            CATCH_CHECK(actual_files.find(expected) != actual_files.end());
        }
    }
}
