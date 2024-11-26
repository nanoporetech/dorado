#include <cstddef>
#include <cstdint>
#include <vector>

namespace dorado::modbase {

std::vector<int8_t> encode_kmer_context(const std::vector<int>& seq,
                                        const std::vector<uint64_t>& seq_mappings,
                                        size_t bases_before,
                                        size_t bases_after,
                                        size_t context_samples);

std::vector<int8_t> encode_kmer_chunk(const std::vector<int>& seq,
                                      const std::vector<uint64_t>& seq_mappings,
                                      size_t kmer_len,
                                      size_t context_samples,
                                      size_t padding_samples,
                                      bool kmer_centered);

}  // namespace dorado::modbase
