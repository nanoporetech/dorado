#include "TestUtils.h"
#include "hts_utils/FastxRandomReader.h"
#include "polish/polish_impl.h"
#include "secondary/common/interval.h"
#include "secondary/common/stats.h"
#include "secondary/common/variant.h"
#include "secondary/consensus/variant_calling_sample.h"
#include "secondary/features/decoder_base.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define TEST_GROUP "[PolishCallVariants]"

namespace dorado::polisher::tests {

CATCH_TEST_CASE("call_variants does not output variants with empty alts", TEST_GROUP) {
    /**
     * \brief The entire reference is deleted in both haplotypes, so the decoded variant has an
     *          empty ALT which cannot be padded by normalization (no flanking reference bases).
     *          general_decode_variants does not remove such variants, so call_variants has to
     *          either convert them into reference records or remove them before output.
     */
    const auto temp_dir = make_temp_dir("polish_call_variants_empty_alts");
    const auto reference_fasta = temp_dir.m_path / "reference.fa";

    {
        std::ofstream ref_out(reference_fasta);
        ref_out << ">chr1\nACG\n";
    }

    const secondary::DecoderBase decoder(secondary::LabelSchemeType::DIPLOID);

    // Logits of shape [num_pos x num_haplotypes x num_classes]. All positions are a deletion ('*' is
    // the symbol 0) in both haplotypes.
    at::Tensor logits = at::zeros({3, 2, 5}, at::kFloat);
    logits.index_put_({at::indexing::Slice(), at::indexing::Slice(), 0}, 1.0f);

    const std::vector<secondary::VariantCallingSample> vc_input_data{
            secondary::VariantCallingSample{
                    .seq_id = 0,
                    .positions_major = {0, 1, 2},
                    .positions_minor = {0, 0, 0},
                    .logits = std::move(logits),
                    .depth = at::full({3}, 10, at::kInt),
            },
    };

    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> draft_readers;
    draft_readers.emplace_back(std::make_unique<hts_io::FastxRandomReader>(reference_fasta));

    const std::vector<std::pair<std::string, int64_t>> draft_lens{
            {"chr1", 3},
    };

    std::atomic<bool> worker_terminate{false};
    secondary::Stats stats;

    const std::vector<secondary::Variant> variants = call_variants(
            worker_terminate, stats, secondary::Interval{0, 1}, vc_input_data, draft_readers,
            draft_lens, decoder, 3.0f, 0.0f, 0.0f, 0.0f, false, false, 1, false);

    for (const secondary::Variant& var : variants) {
        CATCH_INFO("var = " << secondary::to_string(var));
        CATCH_CHECK_FALSE(secondary::has_empty_alts(var));
        CATCH_CHECK((secondary::is_reference_record(var) || secondary::is_valid(var)));
    }
}

}  // namespace dorado::polisher::tests
