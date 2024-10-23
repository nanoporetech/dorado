#pragma once

#include "dorado/models/kits.h"

#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado::adapter_primer_kits {

// For adapters, there are specific sequences we look for at the front of the read. We don't look for exactly
// the reverse complement at the rear of the read, though, because it will generally be truncated. So we list here
// the specific sequences to look for at the front and rear of the reads.
// RNA adapters are only looked for at the rear of the read, so don't have a front sequence.
struct Adapter {
    std::string name;
    std::string front_sequence;
    std::string rear_sequence;
};

enum class AdapterCode { LSK109, LSK110, RNA004 };

const std::unordered_map<AdapterCode, Adapter> adapters = {
        {LSK109, {"LSK109", "AATGTACTTCGTTCAGTTACGTATTGCT", "AGCAATACGTAACTGAACGAAGT"}},
        {LSK110, {"LSK110", "CCTGTACTTCGTTCAGTTACGTATTGC", "AGCAATACGTAACTGAAC"}},
        {RNA004, {"RNA004", "", "GGTTGTTTCTGTTGGTGCTG"}}};

// If we know the kit, and it is in this mapping, we will look for the adapters we expect
// for that kit. If the kit is not specified, or not in this mapping, then we will look for
// both the LSK109 and LSK110 adapters.
const std::unordered_map<AdapterCode, std::set<dorado::models::KitCode>> adapter_kit_map = {
        {LSK109, {SQK_CS9109, SQK_DCS109, SQK_LSK109, SQK_LSK109_XL, SQK_PCS109}},
        {LSK110,
         {SQK_APK114, SQK_LSK110, SQK_LSK110_XL, SQK_LSK111, SQK_LSK111_XL, SQK_LSK112,
          SQK_LSK112_XL, SQK_LSK114, SQK_LSK114_260, SQK_LSK114_XL, SQK_LSK114_XL_260, SQK_PCS111,
          SQK_PCS114, SQK_PCS114_260, SQK_RAD112, SQK_RAD114, SQK_RAD114_260, SQK_ULK114,
          SQK_ULK114_260}},
        {RNA004, {SQK_RNA004}}};

// For primers, we generally look for each primer sequence, and its reverse complement, at both the front and rear of the read.
// If we know the kit, and which primers are expected for that kit, we will look for just what we expect.
struct Primer {
    std::string name;
    std::string sequence;
};

enum class PrimerCode {
    PCR_PSK_rev1,
    PCR_PSK_rev2,
    cDNA_VNP,
    cDNA_SSP,
    PCS110_forward,
    PCS110_reverse,
    RAD,
    None
};

const std::unordered_map<PrimerCode, Primer> primers = {
        {PCR_PSK_rev1, {"PCR_PSK_rev1", "ACTTGCCTGTCGCTCTATCTTCGGCGTCTGCTTGGGTGTTTAACC"}},
        {PCR_PSK_rev2, {"PCR_PSK_rev2", "TTTCTGTTGGTGCTGATATTGCGGCGTCTGCTTGGGTGTTTAACCT"}},
        {cDNA_VNP, {"cDNA_VNP", "ACTTGCCTGTCGCTCTATCTTC"}},
        {cDNA_SSP, {"cDNA_SSP", "TTTCTGTTGGTGCTGATATTGCTGGG"}},
        {PCS110_forward,
         {"PCS110_forward",
          "TCGCCTACCGTGACAAGAAAGTTGTCGGTGTCTTTGTGACTTGCCTGTCGCTCTATCTTCAGAGGAGAGTCCGCCGCCCGCAAGTT"
          "T"}},
        {PCS110_reverse,
         {"PCS110_reverse", "ATCGCCTACCGTGACAAGAAAGTTGTCGGTGTCTTTGTGTTTCTGTTGGTGCTGATATTGCTTT"}},
        {RAD, {"RAD", "GCTTGGGTGTTTAACCGTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA"}},
        {None, {"None", ""}}};

// Only Kit14 sequencing kits are listed here. If the kit is specified, and found in this map,
// then we will search for the specified primer at the front of the read, and its corresponding
// rear primer sequence (as specified in the primer_reverse_map object that follows), at the end.
// Likewise we will search for the RC of the rear primer sequence at the beginning, and the RC of
// the front primer sequence at the end.
// If no kit is specified, or the kit is not found in this map, then we will search for all of
// primer sequences, and their RC, at both the beginning and end.
const std::unordered_map<PrimerCode, std::set<dorado::models::KitCode>> primer_kit_map = {
        {cDNA_VNP, {}};

std::vector<Adapter> get_adapters_for_kit(const std::string& kit_name) {
    auto kit_code = dorado::models::kit_code(kit_name);
    for (const auto& entry : adapter_kit_map) {
        if (entry.second.find(kit_code) != entry.second.end()) {
            return {adapters.at(entry.first)};
        }
    }
    // kit not found in map, so return all known adapters.
    std::vector<Adapter> all_adapters;
    for (const auto& entry : adapters) {
        all_adapters.push_back(entry.second);
    }
    return all_adapters;
}

std::vector<Primer> get_primers_for_kit(const std::string& kit_name) {
    std::vector<Primer> primer_matches;
    auto kit_code = dorado::models::kit_code(kit_name);
    for (const auto& entry : primer_kit_map) {
        if (entry.second.find(kit_code) != entry.second.end()) {
            primer_matches.push_back(primers.at(entry.first));
        }
    }
    // kit not found in map, so return all known primers.
    if (primer_matches.empty()) {
        for (const auto& entry : primers) {
            primer_matches.push_back(entry.second);
        }
    }
    return primer_matches;
}

}  // namespace adapter_primer_kits
