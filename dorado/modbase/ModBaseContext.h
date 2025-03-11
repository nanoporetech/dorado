#pragma once

#include <array>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

/** Helper class for managing modified base context.
 * 
 *  Basic string encoding is as follows:
 * 
 *  [motif_for_A]:[motif_for_C]:[motif_for_G]:[motif_for_T]
 * 
 *  The motifs substitute an X at the position of interest to indicate which one it is, in case
 *  the motif may contain multiple instances of the canonical base. For example, the motif for A
 *  could be "CGACA", with the middle A being the one of interest. In this case the [motif_for_A]
 *  would be "CGXCA".
 * 
 *  If a canonical base does not have a motif, then an underscore is used to indicate this. So the
 *  full string for a case where A can be modified in context "CAT" and C can be modified in context
 *  "CG" would
 *  be "CXT:XG:_:_". 
 */

namespace dorado::modbase {
class MotifMatcher;

class ModBaseContext {
public:
    /// Constructor.
    ModBaseContext();
    ~ModBaseContext();

    /// Get the context set for the specified base (if any).
    const std::string& motif(char base) const;

    /// Get the offset of the canonical base within the motif for the specified base (if any).
    size_t motif_offset(char base) const;

    /** Add a kmer context for a canonical base.
     *  @param motif The kmer context corresponding to a canonical base.
     *  @param offset The zero-indexed position of the canonical base within the kmer.
     * 
     *  This is used to indicate that modifications can only occur for the specified
     *  canonical base when that base is positioned within a particular kmer context.
     */
    void set_context(std::string motif, size_t offset);

    /** Set the object using an string encoding of the contexts.
     *  
     *  This will initialize the object based on a string representation of the context
     *  information.
     * 
     *  @return true if the context_string was successfully decoded, otherwise false.
     */
    bool decode(const std::string& context_string);

    /** Encode the object as a string representation.
     *  
     *  This will produce a string representation of the context information. The primary
     *  purpose of this method is to allow the context information to be embedded in a
     *  guppy::Read object as a string metadata field.
     */
    std::string encode() const;

    /** Return a vector of bools indicating which bases have been checked for modification
     *  according to the context information.
     */
    std::vector<bool> get_sequence_mask(std::string_view sequence) const;

    /** Update a mask provided by the get_sequence_mask function to flag bases for which
     *  the modification probability exceeds the specified threshold.
     * 
     *  Note that this function assumes that the provided mask vector was created via the
     *  get_sequence_mask method.
     * 
     *  The mask will not be altered for any bases which have a context associated with them,
     *  as any such bases should only have their mask values determined by whether the context
     *  is satisfied for that position in the sequence. The threshold is thus ignored for those
     *  bases.
     */
    void update_mask(std::vector<bool>& mask,
                     const std::string& sequence,
                     const std::vector<std::string>& modbase_alphabet,
                     const std::vector<uint8_t>& modbase_probs,
                     uint8_t threshold) const;

private:
    std::array<std::string, 4> m_motifs;
    std::array<size_t, 4> m_offsets = {0, 0, 0, 0};
    std::array<std::unique_ptr<MotifMatcher>, 4> m_motif_matchers;
};

}  // namespace dorado::modbase
