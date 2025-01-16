#pragma once

#include "models/kits.h"

#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado::adapter_primer_kits {

struct Candidate {
    std::string name;
    std::string front_sequence;
    std::string rear_sequence;
};

enum class AdapterCode { LSK110, RNA004 };

enum class PrimerCode { cDNA, PCS110, RAD };

using AC = AdapterCode;
using PC = PrimerCode;
using KC = models::KitCode;

const std::unordered_map<AdapterCode, Candidate> adapters = {
        {AC::LSK110, {"LSK110", "CCTGTACTTCGTTCAGTTACGTATTGC", "AGCAATACGTAACTGAAC"}},
        {AC::RNA004, {"RNA004", "", "GGTTGTTTCTGTTGGTGCTG"}}};

// If we know the kit, and it is in this mapping, we will look for the adapters we expect
// for that kit.
const std::unordered_map<AdapterCode, std::set<dorado::models::KitCode>> adapter_kit_map = {
        {AC::LSK110,
         {KC::SQK_LSK114,       KC::SQK_LSK114_260,       KC::SQK_LSK114_XL, KC::SQK_LSK114_XL_260,
          KC::SQK_PCS114,       KC::SQK_PCS114_260,       KC::SQK_RAD114,    KC::SQK_RAD114_260,
          KC::SQK_ULK114,       KC::SQK_ULK114_260,       KC::SQK_16S114_24, KC::SQK_16S114_24_260,
          KC::SQK_MLK114_96_XL, KC::SQK_MLK114_96_XL_260, KC::SQK_NBD114_24, KC::SQK_NBD114_24_260,
          KC::SQK_NBD114_96,    KC::SQK_NBD114_96_260,    KC::SQK_PCB114_24, KC::SQK_PCB114_24_260,
          KC::SQK_RBK114_24,    KC::SQK_RBK114_24_260,    KC::SQK_RBK114_96, KC::SQK_RBK114_96_260,
          KC::SQK_RPB114_24,    KC::SQK_RPB114_24_260}},
        {AC::RNA004, {KC::SQK_RNA004}}};

const std::unordered_map<PrimerCode, Candidate> primers = {
        {PC::cDNA, {"cDNA", "ACTTGCCTGTCGCTCTATCTTC", "TTTCTGTTGGTGCTGATATTGCTGGG"}},
        {PC::PCS110,
         {"PCS110",
          "TCGCCTACCGTGACAAGAAAGTTGTCGGTGTCTTTGTGACTTGCCTGTCGCTCTATCTTCAGAGGAGAGTCCGCCGCCCGCAAGTTT",
          "AAAGCAATATCAGCACCAACAGAAACACAAAGACACCGACAACTTTCTTGTCACGGTAGGCGAT"}},
        {PC::RAD,
         {"RAD", "GCTTGGGTGTTTAACCGTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA", ""}}};

// Only Kit14 sequencing kits are listed here. If the kit is specified, and found in this map,
// then we will search for the specified primer at the front of the read, and its corresponding
// rear primer sequence (as specified in the primer_reverse_map object that follows), at the end.
// Likewise we will search for the RC of the rear primer sequence at the beginning, and the RC of
// the front primer sequence at the end.
const std::unordered_map<PrimerCode, std::set<dorado::models::KitCode>> primer_kit_map = {
        {
                PC::cDNA,
                {KC::SQK_LSK114, KC::SQK_LSK114_260, KC::SQK_LSK114_XL, KC::SQK_LSK114_XL_260},
        },
        {
                PC::PCS110,
                {KC::SQK_PCS114, KC::SQK_PCS114_260},
        },
        {PC::RAD, {KC::SQK_RAD114, KC::SQK_RAD114_260}}};

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
        return get_candidates(kit_name, ADAPTERS);
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
    std::vector<Candidate> get_primers(const std::string& kit_name) const {
        return get_candidates(kit_name, PRIMERS);
    }

private:
    enum CandidateType { ADAPTERS, PRIMERS };
    std::unordered_map<std::string, std::vector<Candidate>> m_kit_adapter_lut;
    std::unordered_map<std::string, std::vector<Candidate>> m_kit_primer_lut;

    std::vector<Candidate> get_candidates(const std::string& kit_name, CandidateType ty) const;
};

}  // namespace dorado::adapter_primer_kits
