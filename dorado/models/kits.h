#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace dorado::models {

/*
Flowcell product codes that we want to support in auto model selection.
Hyphens replaced with underscores in enumeration but kept in the string representation
to match with values written by minknow into pod5s
*/
enum class Flowcell {
    FLO_FLG001,
    FLO_FLG114,
    FLO_FLG114HD,
    FLO_MIN004RA,
    FLO_MIN106,
    FLO_MIN107,
    FLO_MIN112,
    FLO_MIN114,
    FLO_MIN114HD,
    FLO_MINSP6,
    FLO_PRO001,
    FLO_PRO002,
    FLO_PRO002_ECO,
    FLO_PRO002M,
    FLO_PRO004RA,
    FLO_PRO112,
    FLO_PRO114,
    FLO_PRO114HD,
    FLO_PRO114M,
    UNKNOWN,
};

// Struct containing Flowcell information used for identification / filtering etc.
// The name string must match the minknow / pod5 represenation (using hyphens)
struct FlowcellInfo {
    std::string name;
};

const std::unordered_map<Flowcell, FlowcellInfo>& flowcell_codes();
Flowcell flowcell_code(const std::string& code);
FlowcellInfo flowcell_info(const Flowcell& fc);
std::string to_string(const Flowcell& fc);

/*
All supported sequencing and barcode sequencing kits.
*/
enum class KitCode {
    SQK_APK114,
    SQK_CS9109,
    SQK_DCS108,
    SQK_DCS109,
    SQK_LRK001,
    SQK_LSK108,
    SQK_LSK109,
    SQK_LSK109_XL,
    SQK_LSK110,
    SQK_LSK110_XL,
    SQK_LSK111,
    SQK_LSK111_XL,
    SQK_LSK112,
    SQK_LSK112_XL,
    SQK_LSK114,
    SQK_LSK114_260,
    SQK_LSK114_XL,
    SQK_LSK114_XL_260,
    SQK_LWP001,
    SQK_PCS108,
    SQK_PCS109,
    SQK_PCS111,
    SQK_PCS114,
    SQK_PCS114_260,
    SQK_PSK004,
    SQK_RAD002,
    SQK_RAD003,
    SQK_RAD004,
    SQK_RAD112,
    SQK_RAD114,
    SQK_RAD114_260,
    SQK_RAS201,
    SQK_RLI001,
    SQK_RNA001,
    SQK_RNA002,
    SQK_RNA004,
    SQK_RNA004_XL,
    SQK_ULK001,
    SQK_ULK114,
    SQK_ULK114_260,
    VSK_VBK001,
    VSK_VSK001,
    VSK_VSK003,
    VSK_VSK004,

    // Barcoding Kits
    SQK_16S024,
    SQK_16S114_24,
    SQK_16S114_24_260,
    SQK_MAB114_24,
    SQK_LWB001,
    SQK_MLK111_96_XL,
    SQK_MLK114_96_XL,
    SQK_MLK114_96_XL_260,
    SQK_NBD111_24,
    SQK_NBD111_96,
    SQK_NBD112_24,
    SQK_NBD112_96,
    SQK_NBD114_24,
    SQK_NBD114_24_260,
    SQK_NBD114_96,
    SQK_NBD114_96_260,
    SQK_PBK004,
    SQK_PCB109,
    SQK_PCB110,
    SQK_PCB111_24,
    SQK_PCB114_24,
    SQK_PCB114_24_260,
    SQK_RAB201,
    SQK_RAB204,
    SQK_RBK001,
    SQK_RBK004,
    SQK_RBK110_96,
    SQK_RBK111_24,
    SQK_RBK111_96,
    SQK_RBK114_24,
    SQK_RBK114_24_260,
    SQK_RBK114_96,
    SQK_RBK114_96_260,
    SQK_RLB001,
    SQK_RPB004,
    SQK_RPB114_24,
    SQK_RPB114_24_260,
    VSK_PTC001,
    VSK_VMK001,
    VSK_VMK004,
    VSK_VPS001,
    UNKNOWN,
};

enum class RapidChemistry { UNKNOWN, NONE, V1 };

/*
Sequencing Kit protocol information 
*/
struct ProtocolInfo {
    bool is_barcoding{false};
    RapidChemistry rapid_chemistry{RapidChemistry::NONE};
};

using MotorSpeed = uint16_t;
/*
Sequencing Kit information with string name matching the minknow / pod5
representation (using hyphens).
*/
struct KitInfo {
    std::string name;
    MotorSpeed speed;
    ProtocolInfo protocol{};
};

const std::unordered_map<KitCode, KitInfo>& kit_codes();
KitCode kit_code(const std::string& kit);
KitInfo kit_info(const KitCode& kit);
std::string to_string(const KitCode& kit);

// KitSet contains a collection of flowcells and kits. Any flowcell-kit pair can be used
// to find a Chemsitry
using KitSet = std::pair<std::vector<Flowcell>, std::vector<KitCode>>;
// Some Chemistries have multiple KitSets which we need to accommodate
using KitSets = std::vector<KitSet>;
using SamplingRate = uint16_t;

enum class SampleType {
    DNA,
    RNA002,
    RNA004,
    UNKNOWN,
};

struct SampleTypeInfo {
    const std::string name;
};

/*
The KitSets  are used to generate supported combinations
of flowcell and sequencing / barcoding kits. The cross-product of each pair of vectors 
is taken to generate a flowcell kit pairs. 

Combined with other metadata, each pair make keys for mapping to known chemistries.
*/
struct ChemistryKits {
    const std::string name;
    const SamplingRate sampling_rate;
    const SampleType sample_type;
    const KitSets kit_sets;
};

// Enumeration of chemistries
enum class Chemistry {
    DNA_R9_4_1_E8,
    DNA_R10_4_1_E8_2_260BPS,
    DNA_R10_4_1_E8_2_APK_5KHZ,
    DNA_R10_4_1_E8_2_400BPS_4KHZ,
    DNA_R10_4_1_E8_2_400BPS_5KHZ,
    RNA002_70BPS,
    RNA004_130BPS,
    UNKNOWN,
};

const std::unordered_map<Chemistry, ChemistryKits>& chemistry_kits();
using ChemistryKey = std::tuple<Flowcell, KitCode, SamplingRate>;
using ChemistryMap = std::map<ChemistryKey, Chemistry>;

const ChemistryMap& chemistry_map();
Chemistry get_chemistry(const ChemistryKey& key);
Chemistry get_chemistry(const std::string& chemistry);
ChemistryKey get_chemistry_key(const std::string& flow_cell_product_code,
                               const std::string& sequencing_kit,
                               SamplingRate sample_rate);
std::string to_string(const ChemistryKey& ck);
std::string to_string(const Chemistry& chemistry);

const std::unordered_map<SampleType, SampleTypeInfo>& sample_types();
SampleType get_sample_type(const std::string& sample_type);
SampleType get_sample_type_from_model_name(const std::string& model_name);
SampleTypeInfo get_sample_type_info(const SampleType& sample_type);
std::string to_string(const SampleType& sample_type);

class ConditionInfo {
public:
    ConditionInfo(const ChemistryKey& key)
            : m_flowcell(std::get<0>(key)),
              m_kit(std::get<1>(key)),
              m_kit_info(get_kit_info()),
              m_chemistry(get_chemistry(key)),
              m_sampling_rate(std::get<2>(key)) {}

    inline Flowcell flowcell() const { return m_flowcell; }
    inline KitCode kit() const { return m_kit; }
    inline Chemistry chemistry() const { return m_chemistry; }
    inline SamplingRate sampling_rate() const { return m_sampling_rate; }
    inline RapidChemistry rapid_chemistry() const { return m_kit_info.protocol.rapid_chemistry; }
    // `True` if sequencing kit is a barcoding kit
    inline bool is_barcoding_kit() const { return m_kit_info.protocol.is_barcoding; }

private:
    KitInfo get_kit_info() const;

    Flowcell m_flowcell;
    KitCode m_kit;
    KitInfo m_kit_info;
    Chemistry m_chemistry;
    SamplingRate m_sampling_rate;
};

}  // namespace dorado::models
