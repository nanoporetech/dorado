#pragma once

#include "demux/AdapterDetector.h"
#include "demux/adapter_info.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace dorado::adapter_primer_kits {

class AdapterPrimerManager {
public:
    /** Default Constructor.
     *  This provides a manager that uses the above hard-coded adapter and primer information.
     */
    AdapterPrimerManager();

    /** Custom file constructor.
     *  This will load and parse the specified fasta file.
     * 
     *  Adapter and primer sequences will be selected according to the specifications of that
     *  file.
     */
    AdapterPrimerManager(const std::string& custom_file);

    /** Get the adapter to search for corresponding to the specified kit.
     *  This will return any adapters that should be searched for, corresponding to the specified
     *  kit name.
     * 
     *  For the default case, this will either be an empty vector, indicating that no adapter is
     *  known for that kit, or a single entry, which will be the appropriate adapter for the kit.
     * 
     *  In the case of a custom adapter-primer file, there may be one or more entries, or it may
     *  be empty, depending on whether the specified kit matches the metadata of any of the custom
     *  sequences, and whether any of the sequences are listed as being compatible with all kits.
     */
    std::vector<Candidate> get_adapters(const std::string& kit_name) const {
        return get_candidates(kit_name, demux::PrimerAux::DEFAULT, ADAPTERS);
    }

    /** Get the primer to search for corresponding to the specified kit.
     *  This will return any primers that should be searched for, corresponding to the specified
     *  kit name.
     * 
     *  For the default case, this will either be an empty vector, indicating that no primer is
     *  known for that kit, or a single entry, which will be the appropriate adapter for the kit.
     * 
     *  In the case of a custom adapter-primer file, there may be one or more entries, or it may
     *  be empty, depending on whether the specified kit matches the metadata of any of the custom
     *  sequences, and whether any of the sequences are listed as being compatible with all kits.
     */
    std::vector<Candidate> get_primers(const std::string& kit_name,
                                       demux::PrimerAux primer_aux) const {
        return get_candidates(kit_name, primer_aux, PRIMERS);
    }

private:
    enum CandidateType { ADAPTERS, PRIMERS };
    std::unordered_map<std::string, std::vector<Candidate>> m_kit_adapter_lut;
    std::unordered_map<std::string, std::vector<Candidate>> m_kit_primer_lut;

    std::vector<Candidate> get_candidates(const std::string& kit_name,
                                          demux::PrimerAux primer_aux,
                                          CandidateType ty) const;
};

}  // namespace dorado::adapter_primer_kits
