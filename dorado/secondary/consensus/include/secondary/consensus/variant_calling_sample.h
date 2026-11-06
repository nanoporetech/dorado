#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace dorado::secondary {

class DecoderBase;

/**
 * \brief Type which holds the input data for variant calling. This includes _all_
 *          samples for the current batch of draft sequences and the inference results (logits) for those samples.
 *          The variant calling process will be responsible to group samples by draft sequence ID, etc.
 */
struct VariantCallingSample {
    int32_t seq_id = -1;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
    at::Tensor logits;
    at::Tensor depth;
    std::vector<std::string> read_ids_left{};
    std::vector<std::string> read_ids_right{};
    at::Tensor basecalls{};

    void validate() const;
    int64_t start() const;
    int64_t end() const;
};

void debug_print(std::ostream& os, const secondary::VariantCallingSample& vc_sample, bool detailed);

std::string to_string(const secondary::VariantCallingSample& vc_sample, bool detailed);

std::ostream& operator<<(std::ostream& os, const VariantCallingSample& vc_sample);

bool operator==(const VariantCallingSample& lhs, const VariantCallingSample& rhs);

/**
 * \brief Strict weak ordering for reduction/trim paths.
 *          Samples are ordered by full starting coordinate (major, then minor), and for identical
 *          starts the longer sample comes first to match trimming expectations for containment.
 */
bool variant_calling_sample_less(const VariantCallingSample& lhs, const VariantCallingSample& rhs);

VariantCallingSample slice_vc_sample(const VariantCallingSample& vc_sample,
                                     int64_t idx_start,
                                     int64_t idx_end);

/**
 * \brief Slices the variant calling sample between specified reference coordinates.
 *          Throws if coordinates are not valid.
 * \param vc_sample Sample to trim.
 * \param ref_start Reference start coordinate, zero based. Will be looked up in positions_major.
 * \param ref_end   Reference end coordinate (exclusive). Will be looked up in positions_major.
 * \returns A new VariantCallingSample object sliced between the first column matching ref_start and
 *          last column matching (ref_end - 1).
 */
VariantCallingSample slice_vc_sample_in_ref_coords(const VariantCallingSample& vc_sample,
                                                   const int64_t ref_start,
                                                   const int64_t ref_end);

/**
 * \brief Given a vector of variant calling samples, it returns a new vector of VariantCallingSample objects
 *          where neighboring input samples are merged if their positions are adjacent and on the same sequence.
 * \param vc_samples Input variant calling samples. Should be sorted by the start position
 *                   (positions_major[0]). The samples are consumed while constructing the result.
 * \return A new vector of VariantCallingSample where any adjacent samples from the input are merged into a single sample.
 */
std::vector<VariantCallingSample> merge_vc_samples(std::vector<VariantCallingSample> vc_samples);

// Consumes vc_samples so intermediate tensor storage can be released as samples are joined.
std::vector<VariantCallingSample> join_samples(std::vector<VariantCallingSample> vc_samples);

/**
 * \brief Splits samples at the given reference boundaries (e.g. hemizygous region edges) so that
 *          each piece can be decoded independently. A boundary falling inside a variant (as
 *          detected in the decoded sample) is deferred to the variant end. It is ignored if the
 *          variant reaches the start or end of the sample.
 * \param vc_samples Joined samples for one sequence. Consumed.
 * \param boundaries Sorted reference coordinates at which a new chunk should start.
 */
std::vector<VariantCallingSample> split_vc_samples_on_boundaries(
        std::vector<VariantCallingSample> vc_samples,
        std::span<const int64_t> boundaries,
        const std::string& draft,
        const DecoderBase& decoder);

std::vector<VariantCallingSample> trim_vc_samples(
        const std::vector<VariantCallingSample>& vc_input_data,
        const std::vector<int32_t>& ordered_ids);

}  // namespace dorado::secondary
