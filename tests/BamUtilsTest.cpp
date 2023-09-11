#include "TestUtils.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"

#include <catch2/catch.hpp>
#include <htslib/sam.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#define TEST_GROUP "[bam_utils]"

namespace fs = std::filesystem;

namespace {
class WrappedKString {
    kstring_t m_str = KS_INITIALIZE;

public:
    WrappedKString() {
        // On Windows |sam_hdr_find_tag_id| lives in a DLL and uses a different heap, but
        // |ks_free| is inline so when we call it we crash trying to free unknown memory. To
        // work around this we resize the kstring to a big value in our code so no resizing
        // happens inside the htslib library.
        ks_resize(&m_str, 1e6);
    }
    ~WrappedKString() { ks_free(&m_str); }

    kstring_t *get() { return &m_str; }
};
}  // namespace

TEST_CASE("BamUtilsTest: fetch keys from PG header", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("aligner_test"));
    auto sam = aligner_test_dir / "basecall.sam";

    auto keys = dorado::utils::extract_pg_keys_from_hdr(sam.string(), {"PN", "CL", "VN"});
    CHECK(keys["PN"] == "dorado");
    CHECK(keys["VN"] == "0.2.3+0f041c4+dirty");
    CHECK(keys["CL"] ==
          "dorado basecaller dna_r9.4.1_e8_hac@v3.3 ./tests/data/pod5 -x cpu --modified-bases "
          "5mCG");
}

TEST_CASE("BamUtilsTest: add_rg_hdr read group headers", TEST_GROUP) {
    auto has_read_group_header = [](sam_hdr_t *ptr, const char *id) {
        return sam_hdr_line_index(ptr, "RG", id) >= 0;
    };
    WrappedKString barcode_kstring;
    auto get_barcode_tag = [&barcode_kstring](sam_hdr_t *ptr,
                                              const char *id) -> std::optional<std::string> {
        if (sam_hdr_find_tag_id(ptr, "RG", "ID", id, "BC", barcode_kstring.get()) != 0) {
            return std::nullopt;
        }
        std::string tag(ks_str(barcode_kstring.get()), ks_len(barcode_kstring.get()));
        return tag;
    };

    SECTION("No read groups generate no headers") {
        dorado::SamHdrPtr sam_header(sam_hdr_init());
        CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == 0);
        dorado::utils::add_rg_hdr(sam_header.get(), {}, {});
        CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == 0);
    }

    const std::unordered_map<std::string, dorado::ReadGroup> read_groups{
            {"id_0",
             {"run_0", "basecalling_mod_0", "flowcell_0", "device_0", "exp_start_0", "sample_0"}},
            {"id_1",
             {"run_1", "basecalling_mod_1", "flowcell_1", "device_1", "exp_start_1", "sample_1"}},
    };

    SECTION("Read groups") {
        dorado::SamHdrPtr sam_header(sam_hdr_init());
        dorado::utils::add_rg_hdr(sam_header.get(), read_groups, {});

        // Check the IDs of the groups are all there.
        CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == read_groups.size());
        for (auto &&[id, read_group] : read_groups) {
            CHECK(has_read_group_header(sam_header.get(), id.c_str()));
            // None of the read groups should have a barcode.
            CHECK(get_barcode_tag(sam_header.get(), id.c_str()) == std::nullopt);
        }
    }

    // Pick some of the barcode kits (randomly chosen indices).
    const auto &kit_infos = dorado::barcode_kits::get_kit_infos();
    const std::vector<std::string> barcode_kits{
            std::next(kit_infos.begin(), 1)->first,
            std::next(kit_infos.begin(), 7)->first,
    };

    SECTION("Read groups with barcodes") {
        dorado::SamHdrPtr sam_header(sam_hdr_init());
        dorado::utils::add_rg_hdr(sam_header.get(), read_groups, barcode_kits);

        // Check the IDs of the groups are all there.
        size_t total_barcodes = 0;
        for (const auto &kit_name : barcode_kits) {
            total_barcodes += kit_infos.at(kit_name).barcodes.size();
        }
        const size_t total_groups = read_groups.size() * (total_barcodes + 1);
        CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == total_groups);

        // Check that the IDs match the expected format.
        const auto &barcode_seqs = dorado::barcode_kits::get_barcodes();
        for (auto &&[id, read_group] : read_groups) {
            CHECK(has_read_group_header(sam_header.get(), id.c_str()));
            CHECK(get_barcode_tag(sam_header.get(), id.c_str()) == std::nullopt);

            // The headers with barcodes should contain those barcodes.
            for (const auto &kit_name : barcode_kits) {
                const auto &kit_info = kit_infos.at(kit_name);
                for (const auto &barcode_name : kit_info.barcodes) {
                    const auto full_id = id + '_' + kit_name + '_' + barcode_name;
                    const auto &barcode_seq = barcode_seqs.at(barcode_name);
                    CHECK(has_read_group_header(sam_header.get(), full_id.c_str()));
                    CHECK(get_barcode_tag(sam_header.get(), full_id.c_str()) == barcode_seq);
                }
            }
        }
    }
}
