#include "adapter_primer_kits.h"

#include "models/kits.h"
#include "parse_custom_sequences.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <cctype>
#include <set>
#include <sstream>

namespace dorado::adapter_primer_kits {

namespace {

enum class AdapterCode { LSK110, RNA004 };

enum class PrimerCode { cDNA, PCS110, RAD };

using AC = AdapterCode;
using PC = PrimerCode;
using KC = models::KitCode;

const std::unordered_map<AdapterCode, Candidate> adapters = {
        {AC::LSK110, {"LSK110", "CCTGTACTTCGTTCAGTTACGTATTGC", "AGCAATACGTAACTGAAC"}},
        {AC::RNA004, {"RNA004", "", "GGTTGTTTCTGTTGGTGCTG"}}};

// If we know the kit, and it is in this mapping, we will look for the adapters we expect
// for that kit. We look for the specified front adapter sequence at the beginning of the
// read, and the specified rear adapter sequence at the end of the read.
const std::unordered_map<AdapterCode, std::set<dorado::models::KitCode>> adapter_kit_map = {
        {AC::LSK110, {KC::SQK_LSK114,        KC::SQK_LSK114_260,    KC::SQK_LSK114_XL,
                      KC::SQK_LSK114_XL_260, KC::SQK_PCS114,        KC::SQK_PCS114_260,
                      KC::SQK_RAD114,        KC::SQK_RAD114_260,    KC::SQK_ULK114,
                      KC::SQK_ULK114_260,    KC::SQK_16S114_24,     KC::SQK_16S114_24_260,
                      KC::SQK_MAB114_24,     KC::SQK_MLK114_96_XL,  KC::SQK_MLK114_96_XL_260,
                      KC::SQK_NBD114_24,     KC::SQK_NBD114_24_260, KC::SQK_NBD114_96,
                      KC::SQK_NBD114_96_260, KC::SQK_PCB114_24,     KC::SQK_PCB114_24_260,
                      KC::SQK_RBK114_24,     KC::SQK_RBK114_24_260, KC::SQK_RBK114_96,
                      KC::SQK_RBK114_96_260, KC::SQK_RPB114_24,     KC::SQK_RPB114_24_260}},
        {AC::RNA004, {KC::SQK_RNA004, KC::SQK_RNA004_XL, KC::SQK_DRB004_24}}};

// Note that for cDNA and PCS110 primers, what would normally be considered the "rear" primer
// will actually be found near the beginning of a forward read, and vice-versa for reverse
// reads. So for example, we list the SSP cDNA primer as the frint primer, and the VNP one as
// the rear primer, so that the code will work properly. The PCS and RAD primer sequences used
// here are also truncated from the beginning. This does not affect trimming, because trimming
// is done from the end of the detected primer. It does allow these sequences to be used with
// barcoding, where a truncated version of the primer appears as the inside flanking region.
const std::unordered_map<PrimerCode, Candidate> primers = {
        {PC::cDNA,
         {
                 "cDNA",
                 "TTTCTGTTGGTGCTGATATTGCTGGG",  // SSP
                 "ACTTGCCTGTCGCTCTATCTTCTTT"    // VNP
         }},
        {PC::PCS110,
         {
                 // These are actually truncated version of the actual primer sequences.
                 "PCS110",
                 "TTTCTGTTGGTGCTGATATTGCTTT",                          // SSP
                 "ACTTGCCTGTCGCTCTATCTTCAGAGGAGAGTCCGCCGCCCGCAAGTTTT"  // VNP
         }},
        {PC::RAD,
         {
                 // This is also a truncated version of the actual primer sequence.
                 "RAD", "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA",
                 ""  // No rear primer for RAD
         }}};

// Only Kit14 sequencing kits are listed here. If the kit is specified, and found in this map,
// then we will search for the specified front primer sequence near the beginning of the read,
// and the RC of the specified rear primer sequence near the end of the read. Likewise we will
// search for the rear primer sequence at the beginning, and the RC of the front primer sequence
// at the end of the read, which is what we should see if we've sequenced the reverse strand.
const std::unordered_map<PrimerCode, std::set<dorado::models::KitCode>> primer_kit_map = {
        {
                PC::cDNA,
                {KC::SQK_LSK114, KC::SQK_LSK114_260, KC::SQK_LSK114_XL, KC::SQK_LSK114_XL_260},
        },
        {
                PC::PCS110,
                {KC::SQK_PCS114, KC::SQK_PCS114_260, KC::SQK_PCB114_24, KC::SQK_PCB114_24_260},
        },
        {
                PC::RAD,
                {KC::SQK_RAD114, KC::SQK_RAD114_260, KC::SQK_ULK114, KC::SQK_ULK114_260,
                 KC::SQK_RBK114_24, KC::SQK_RBK114_24_260, KC::SQK_RBK114_96,
                 KC::SQK_RBK114_96_260},
        },
};

struct CustomItem {
    enum ItemType { UNKNOWN, ADAPTER, PRIMER };
    enum ItemPos { FRONT, REAR };

    std::string name;
    std::string front_sequence;
    std::string rear_sequence;
    ItemType item_type{UNKNOWN};

    bool add_sequence(std::string nm, std::string sequence, ItemType ty, ItemPos pos) {
        if (!name.empty() && name != nm) {
            spdlog::error("Custom adapter/primer name mismatch error: {} and {}", name, nm);
            return false;
        }
        if (item_type != ty && item_type != UNKNOWN) {
            spdlog::error("Custom adapter/primer type mismatch error. {} and {}", int(item_type),
                          int(ty));
            return false;
        }
        if (sequence.empty()) {
            spdlog::error("Custom adapter/primer parsing error: Empty sequence specified.");
            return false;
        }
        switch (pos) {
        case FRONT:
            if (!front_sequence.empty()) {
                spdlog::error(
                        "Custom adapter/primer parsing error: Duplicate front sequence for {}", nm);
                return false;
            }
            front_sequence = std::move(sequence);
            break;
        case REAR:
            if (!rear_sequence.empty()) {
                spdlog::error("Custom adapter/primer parsing error: Duplicate rear sequence for {}",
                              nm);
                return false;
            }
            rear_sequence = std::move(sequence);
            break;
        default:
            spdlog::error("Custom adapter/primer parsing error: Invalid sequence position.");
            return false;
        }
        name = std::move(nm);
        item_type = ty;
        return true;
    }
};

std::set<std::string> parse_kit_names(const std::string& kit_string) {
    std::set<std::string> kit_names;
    std::istringstream kit_stream(kit_string);
    std::string token;
    while (std::getline(kit_stream, token, ',')) {
        if (!token.empty()) {
            auto kit_name = dorado::utils::to_uppercase(token);
            kit_names.insert(std::move(kit_name));
        }
    }
    return kit_names;
}

CustomItem::ItemType item_type(const std::unordered_map<std::string, std::string>& tags) {
    auto type_field = tags.find("type");
    if (type_field == tags.end()) {
        return CustomItem::UNKNOWN;
    }
    if (type_field->second == "adapter") {
        return CustomItem::ADAPTER;
    }
    if (type_field->second == "primer") {
        return CustomItem::PRIMER;
    }
    return CustomItem::UNKNOWN;
}

std::pair<std::string, CustomItem::ItemPos> split_name(const std::string& record_name) {
    auto front_start = record_name.find("_front");
    if (front_start != std::string::npos) {
        if (record_name.size() != front_start + 6) {
            return {};
        }
        auto name = record_name.substr(0, front_start);
        return {std::move(name), CustomItem::FRONT};
    }
    auto rear_start = record_name.find("_rear");
    if (rear_start != std::string::npos) {
        if (record_name.size() != rear_start + 5) {
            return {};
        }
        auto name = record_name.substr(0, rear_start);
        return {std::move(name), CustomItem::REAR};
    }
    return {};
}

std::set<std::string> kit_names(const std::unordered_map<std::string, std::string>& tags) {
    auto kits_field = tags.find("kits");
    if (kits_field == tags.end()) {
        spdlog::error("Custom adapter/primer parsing error: No 'kits' field found.");
        throw std::runtime_error("Error parsing custom adapter/primer file.");
    }
    return parse_kit_names(kits_field->second);
}

}  // anonymous namespace

AdapterPrimerManager::AdapterPrimerManager() {
    for (const auto& entry : adapter_kit_map) {
        auto adapter_code = entry.first;
        const auto& kit_codes = entry.second;
        const auto& adapter = adapters.at(adapter_code);
        for (auto kit_code : kit_codes) {
            auto kit_name = dorado::models::to_string(kit_code);
            m_kit_adapter_lut[kit_name].push_back(adapter);
        }
    }
    for (const auto& entry : primer_kit_map) {
        auto primer_code = entry.first;
        const auto& kit_codes = entry.second;
        const auto& primer = primers.at(primer_code);
        for (auto kit_code : kit_codes) {
            auto kit_name = dorado::models::to_string(kit_code);
            m_kit_primer_lut[kit_name].push_back(primer);
        }
    }
}

AdapterPrimerManager::AdapterPrimerManager(const std::string& custom_file) {
    auto custom_sequences = dorado::demux::parse_custom_sequences(custom_file);
    std::unordered_map<std::string, CustomItem> custom_items;
    std::unordered_map<std::string, std::set<std::string>> custom_item_kits;
    for (const auto& entry : custom_sequences) {
        // Entries must specify whether they are adapters or primers, whether they
        // are found at the front or rear of a read, and what kit they are meant for.
        auto ty = item_type(entry.tags);
        auto [name, pos] = split_name(entry.name);
        std::set<std::string> kits = kit_names(entry.tags);
        if (ty == CustomItem::UNKNOWN) {
            spdlog::error("Invalid 'type' tag in custom adapter/primer file.");
            throw std::runtime_error("Error parsing custom adapter/primer file.");
        }
        if (name.empty()) {
            spdlog::error("Could not extract position from record name.");
            throw std::runtime_error("Error parsing custom adapter/primer file.");
        }
        auto& item = custom_items[name];
        if (!item.add_sequence(name, entry.sequence, ty, pos)) {
            throw std::runtime_error("Error parsing custom adapter/primer file.");
        }
        auto& item_kits = custom_item_kits[name];
        if (item_kits.empty()) {
            item_kits = std::move(kits);
        } else {
            if (!kits.empty()) {
                if (item_kits != kits) {
                    spdlog::error("inconsistent kit specifications.");
                    throw std::runtime_error("Error parsing custom adapter/primer file.");
                }
            }
        }
    }
    for (auto& entry : custom_items) {
        auto& kits = custom_item_kits[entry.first];
        if (kits.empty()) {
            kits.insert("ANY");
        }
        Candidate candidate{entry.first, entry.second.front_sequence, entry.second.rear_sequence};
        for (const auto& kit : kits) {
            if (entry.second.item_type == CustomItem::ADAPTER) {
                m_kit_adapter_lut[kit].push_back(candidate);
            } else {
                m_kit_primer_lut[kit].push_back(candidate);
            }
        }
    }
}

std::vector<Candidate> AdapterPrimerManager::get_candidates(const std::string& kit_name,
                                                            CandidateType ty) const {
    // If the requested kit name is "ALL", then all candidates will be returned, regardless of kit
    // compatibility. Otherwise only candidates matching the requested kit, and candidates listed
    // as being for any kit, will be included. Note that the latter will only exist if a custom
    // adapter/primer file was used, for entries with no kit information provided.
    const std::unordered_map<std::string, std::vector<Candidate>>* candidate_lut = nullptr;
    switch (ty) {
    case ADAPTERS:
        candidate_lut = &m_kit_adapter_lut;
        break;
    case PRIMERS:
        candidate_lut = &m_kit_primer_lut;
        break;
    default:
        throw std::runtime_error(
                "Unexpected candidate type in AdapterPrimerManager::get_candidate method.");
    }
    std::vector<Candidate> results;
    if (kit_name == "ALL") {
        for (const auto& item : *candidate_lut) {
            results.insert(results.end(), item.second.begin(), item.second.end());
        }
        return results;
    }
    auto iter = candidate_lut->find(kit_name);
    if (iter != candidate_lut->end()) {
        results = iter->second;
    }
    iter = candidate_lut->find("ANY");
    if (iter != candidate_lut->end()) {
        results.insert(results.end(), iter->second.begin(), iter->second.end());
    }
    return results;
}

}  // namespace dorado::adapter_primer_kits
