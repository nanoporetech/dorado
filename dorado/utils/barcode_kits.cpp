#include "barcode_kits.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>

namespace dorado::barcode_kits {

namespace {

// Flank sequences per barcode kit.
// There are 2 types of kits described here -
// 1. Double ended kits that have a different flanking region for the top and bottom barcodes.
// 2. Single or double ended kits where the flanking region is the same for top and/or bottom barcodes.
const std::string RAB_1st_FRONT = "CCGTGAC";
const std::string RAB_1st_REAR = "AGAGTTTGATCATGGCTCAG";
const std::string RAB_2nd_FRONT = "CCGTGAC";
const std::string RAB_2nd_REAR = "CGGTTACCTTGTTACGACTT";

const std::string RBK_FRONT = "TATTGCT";
const std::string RBK_REAR = "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA";

const std::string RBK4_FRONT = "GCTTGGGTGTTTAACC";
const std::string RBK4_REAR = "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA";

const std::string RBK4_kit14_FRONT = "GCTTGGGTGTTTAACC";
const std::string RBK4_kit14_REAR = "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA";

const std::string RLB_FRONT = "CCGTGAC";
const std::string RLB_REAR = "CGTTTTTCGTGCGCCGCTTC";

const std::string BC_1st_FRONT = "GGTGCTG";
const std::string BC_1st_REAR = "TTAACCTTTCTGTTGGTGCTGATATTGC";
const std::string BC_2nd_FRONT = "GGTGCTG";
const std::string BC_2nd_REAR = "TTAACCTACTTGCCTGTCGCTCTATCTTC";

const std::string NB_1st_FRONT = "ATTGCTAAGGTTAA";
const std::string NB_1st_REAR = "CAGCACCT";
const std::string NB_2nd_FRONT = "ATTGCTAAGGTTAA";
const std::string NB_2nd_REAR = "CAGCACC";

const std::string LWB_1st_FRONT = "CCGTGAC";
const std::string LWB_1st_REAR = "ACTTGCCTGTCGCTCTATCTTC";
const std::string LWB_2nd_FRONT = "CCGTGAC";
const std::string LWB_2nd_REAR = "TTTCTGTTGGTGCTGATATTGC";

const std::string MAB_FRONT = "GCTTGGGTGTTTAACC";
const std::string MAB_REAR = "CCATATCCGTGTCGCCCTT";

// Predefined collection of barcode sequences that are used by various kits.
// Since some of the collections are used in multiple barcoding kits, it made
// sense to pull them out separately.
const std::vector<std::string> BC_1_12 = {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06",
                                          "BC07", "BC08", "BC09", "BC10", "BC11", "BC12"};
const std::vector<std::string> BC_1_12A = {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06",
                                           "BC07", "BC08", "BC09", "BC10", "BC11", "RLB12A"};
const std::vector<std::string> BC_1_24 = {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06",
                                          "BC07", "BC08", "BC09", "BC10", "BC11", "BC12",
                                          "BC13", "BC14", "BC15", "BC16", "BC17", "BC18",
                                          "BC19", "BC20", "BC21", "BC22", "BC23", "BC24"};

// BC2_1_24 is the same as BC_1_24 except it uses 12A instead of 12.
const std::vector<std::string> BC2_1_24 = {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06",
                                           "BC07", "BC08", "BC09", "BC10", "BC11", "RLB12A",
                                           "BC13", "BC14", "BC15", "BC16", "BC17", "BC18",
                                           "BC19", "BC20", "BC21", "BC22", "BC23", "BC24"};

const std::vector<std::string> BC_1_96 = {
        "BC01", "BC02", "BC03", "BC04", "BC05", "BC06", "BC07", "BC08", "BC09", "BC10", "BC11",
        "BC12", "BC13", "BC14", "BC15", "BC16", "BC17", "BC18", "BC19", "BC20", "BC21", "BC22",
        "BC23", "BC24", "BC25", "BC26", "BC27", "BC28", "BC29", "BC30", "BC31", "BC32", "BC33",
        "BC34", "BC35", "BC36", "BC37", "BC38", "BC39", "BC40", "BC41", "BC42", "BC43", "BC44",
        "BC45", "BC46", "BC47", "BC48", "BC49", "BC50", "BC51", "BC52", "BC53", "BC54", "BC55",
        "BC56", "BC57", "BC58", "BC59", "BC60", "BC61", "BC62", "BC63", "BC64", "BC65", "BC66",
        "BC67", "BC68", "BC69", "BC70", "BC71", "BC72", "BC73", "BC74", "BC75", "BC76", "BC77",
        "BC78", "BC79", "BC80", "BC81", "BC82", "BC83", "BC84", "BC85", "BC86", "BC87", "BC88",
        "BC89", "BC90", "BC91", "BC92", "BC93", "BC94", "BC95", "BC96"};

const std::vector<std::string> NB_1_12 = {"NB01", "NB02", "NB03", "NB04", "NB05", "NB06",
                                          "NB07", "NB08", "NB09", "NB10", "NB11", "NB12"};

const std::vector<std::string> NB_13_24 = {"NB13", "NB14", "NB15", "NB16", "NB17", "NB18",
                                           "NB19", "NB20", "NB21", "NB22", "NB23", "NB24"};

const std::vector<std::string> NB_1_24 = {"NB01", "NB02", "NB03", "NB04", "NB05", "NB06",
                                          "NB07", "NB08", "NB09", "NB10", "NB11", "NB12",
                                          "NB13", "NB14", "NB15", "NB16", "NB17", "NB18",
                                          "NB19", "NB20", "NB21", "NB22", "NB23", "NB24"};

const std::vector<std::string> NB_1_96 = {
        "NB01", "NB02", "NB03", "NB04", "NB05", "NB06", "NB07", "NB08", "NB09", "NB10", "NB11",
        "NB12", "NB13", "NB14", "NB15", "NB16", "NB17", "NB18", "NB19", "NB20", "NB21", "NB22",
        "NB23", "NB24", "NB25", "NB26", "NB27", "NB28", "NB29", "NB30", "NB31", "NB32", "NB33",
        "NB34", "NB35", "NB36", "NB37", "NB38", "NB39", "NB40", "NB41", "NB42", "NB43", "NB44",
        "NB45", "NB46", "NB47", "NB48", "NB49", "NB50", "NB51", "NB52", "NB53", "NB54", "NB55",
        "NB56", "NB57", "NB58", "NB59", "NB60", "NB61", "NB62", "NB63", "NB64", "NB65", "NB66",
        "NB67", "NB68", "NB69", "NB70", "NB71", "NB72", "NB73", "NB74", "NB75", "NB76", "NB77",
        "NB78", "NB79", "NB80", "NB81", "NB82", "NB83", "NB84", "NB85", "NB86", "NB87", "NB88",
        "NB89", "NB90", "NB91", "NB92", "NB93", "NB94", "NB95", "NB96"};

// RBK_1_96 is the same as BC_1_96 except for 26, 39, 40, 58, 54 and 60.
const std::vector<std::string> RBK_1_96 = {
        "BC01", "BC02", "BC03", "BC04",  "BC05",  "BC06",  "BC07",  "BC08", "BC09", "BC10",  "BC11",
        "BC12", "BC13", "BC14", "BC15",  "BC16",  "BC17",  "BC18",  "BC19", "BC20", "BC21",  "BC22",
        "BC23", "BC24", "BC25", "RBK26", "BC27",  "BC28",  "BC29",  "BC30", "BC31", "BC32",  "BC33",
        "BC34", "BC35", "BC36", "BC37",  "BC38",  "RBK39", "RBK40", "BC41", "BC42", "BC43",  "BC44",
        "BC45", "BC46", "BC47", "RBK48", "BC49",  "BC50",  "BC51",  "BC52", "BC53", "RBK54", "BC55",
        "BC56", "BC57", "BC58", "BC59",  "RBK60", "BC61",  "BC62",  "BC63", "BC64", "BC65",  "BC66",
        "BC67", "BC68", "BC69", "BC70",  "BC71",  "BC72",  "BC73",  "BC74", "BC75", "BC76",  "BC77",
        "BC78", "BC79", "BC80", "BC81",  "BC82",  "BC83",  "BC84",  "BC85", "BC86", "BC87",  "BC88",
        "BC89", "BC90", "BC91", "BC92",  "BC93",  "BC94",  "BC95",  "BC96"};

const std::vector<std::string> TP_1_24 = {"TP01", "TP02", "TP03", "TP04", "TP05", "TP06",
                                          "TP07", "TP08", "TP09", "TP10", "TP11", "TP12",
                                          "TP13", "TP14", "TP15", "TP16", "TP17", "TP18",
                                          "TP19", "TP20", "TP21", "TP22", "TP23", "TP24"};

// Kit specific scoring parameters.

const BarcodeKitScoringParams DEFAULT_PARAMS{};

const BarcodeKitScoringParams RBK114_PARAMS{
        /*max_barcode_penalty*/ 12,
        /*barcode_end_proximity*/ 75,
        /*min_barcode_penalty_dist*/ 3,
        /*min_separation_only_dist*/ 6,
        /*flank_left_pad*/ 5,
        /*flank_right_pad*/ 10,
        /*front_barcode_window*/ 175,
        /*rear_barcode_window*/ 175,
        /*min_flank_score*/ 0.0f,
        /*midstrand_flank_score*/ 0.95f,
};

const BarcodeKitScoringParams MAB114_PARAMS{
        /*max_barcode_penalty*/ 12,
        /*barcode_end_proximity*/ 75,
        /*min_barcode_penalty_dist*/ 3,
        /*min_separation_only_dist*/ 6,
        /*flank_left_pad*/ 5,
        /*flank_right_pad*/ 10,
        /*front_barcode_window*/ 175,
        /*rear_barcode_window*/ 175,
        /*min_flank_score*/ 0.0f,
        /*midstrand_flank_score*/ 0.7f,  // Note: see wiki writeup attached to INSTX-9193.
};

const BarcodeKitScoringParams TWIST_PARAMS{
        /*max_barcode_penalty*/ 5,
        /*barcode_end_proximity*/ 75,
        /*min_barcode_penalty_dist*/ 2,
        /*min_separation_only_dist*/ 6,
        /*flank_left_pad*/ 10,
        /*flank_right_pad*/ 10,
        /*front_barcode_window*/ 175,
        /*rear_barcode_window*/ 175,
        /*min_flank_score*/ 0.5f,
        /*midstrand_flank_score*/ 0.95f,
};

// Some arrangement names are just aliases of each other. This is because they were released
// as part of different kits, but they map to the same underlying arrangement.
const KitInfo kit_16S = {
        "16S",         true,         true,    false,   RAB_1st_FRONT,  RAB_1st_REAR,
        RAB_2nd_FRONT, RAB_2nd_REAR, BC_1_24, BC_1_24, DEFAULT_PARAMS,
};

const KitInfo kit_lwb = {
        "LWB",         true,         true,    false,   LWB_1st_FRONT,  LWB_1st_REAR,
        LWB_2nd_FRONT, LWB_2nd_REAR, BC_1_12, BC_1_12, DEFAULT_PARAMS,
};

const KitInfo kit_lwb24 = {
        "LWB24",       true,         true,    false,   LWB_1st_FRONT,  LWB_1st_REAR,
        LWB_2nd_FRONT, LWB_2nd_REAR, BC_1_24, BC_1_24, DEFAULT_PARAMS,
};

const KitInfo kit_nb12 = {
        "NB12",       true,        true,    false,   NB_1st_FRONT,   NB_1st_REAR,
        NB_2nd_FRONT, NB_2nd_REAR, NB_1_12, NB_1_12, DEFAULT_PARAMS,
};

const KitInfo kit_nb24 = {
        "NB24",       true,        true,    false,   NB_1st_FRONT,   NB_1st_REAR,
        NB_2nd_FRONT, NB_2nd_REAR, NB_1_24, NB_1_24, DEFAULT_PARAMS,
};

const KitInfo kit_nb96 = {
        "NB96",       true,        true,    false,   NB_1st_FRONT,   NB_1st_REAR,
        NB_2nd_FRONT, NB_2nd_REAR, NB_1_96, NB_1_96, DEFAULT_PARAMS,
};

const KitInfo kit_rab = {
        "RAB",         true,         true,    false,   RAB_1st_FRONT,  RAB_1st_REAR,
        RAB_2nd_FRONT, RAB_2nd_REAR, BC_1_12, BC_1_12, DEFAULT_PARAMS,
};

const KitInfo kit_rbk96 = {
        "RBK96", false, false, false, RBK4_FRONT, RBK4_REAR, "", "", RBK_1_96, {}, DEFAULT_PARAMS,
};

const KitInfo kit_rbk4 = {
        "RBK4", false, false, false, RBK4_FRONT, RBK4_REAR, "", "", BC_1_12, {}, DEFAULT_PARAMS,
};

const KitInfo kit_rlb = {
        "RLB", true, false, false, RLB_FRONT, RLB_REAR, "", "", BC_1_12A, {}, DEFAULT_PARAMS,
};

// Final map to go from kit name to actual barcode arrangement information.
std::unordered_map<std::string, KitInfo> kit_info_map = {
        // SQK-16S024 && SQK-16S114-24
        {"SQK-16S024", kit_16S},
        {"SQK-16S114-24", kit_16S},
        // MAB114
        {"SQK-MAB114-24",
         {
                 "MAB114",
                 false,
                 false,
                 false,
                 MAB_FRONT,
                 MAB_REAR,
                 "",
                 "",
                 TP_1_24,
                 {},
                 MAB114_PARAMS,
         }},
        // LWB
        {"SQK-PBK004", kit_lwb},
        {"SQK-LWB001", kit_lwb},
        {"SQK-PCB109", kit_lwb},
        {"SQK-PCB110", kit_lwb},
        // LWB24
        {"SQK-PCB111-24", kit_lwb24},
        {"SQK-PCB114-24", kit_lwb24},
        // NB12
        {"EXP-NBD103", kit_nb12},
        {"EXP-NBD104", kit_nb12},
        // NB13-24
        {"EXP-NBD114",
         {
                 "NB13-24",
                 true,
                 true,
                 false,
                 NB_1st_FRONT,
                 NB_1st_REAR,
                 NB_2nd_FRONT,
                 NB_2nd_REAR,
                 NB_13_24,
                 NB_13_24,
                 DEFAULT_PARAMS,
         }},
        // NB24
        {"SQK-NBD111-24", kit_nb24},
        {"SQK-NBD114-24", kit_nb24},
        {"EXP-NBD114-24", kit_nb24},
        // NB96
        {"EXP-NBD196", kit_nb96},
        {"SQK-MLK111-96-XL", kit_nb96},
        {"SQK-NBD111-96", kit_nb96},
        {"SQK-NBD114-96", kit_nb96},
        {"SQK-MLK114-96-XL", kit_nb96},
        // PCR12
        {"EXP-PBC001",
         {
                 "PCR12",
                 true,
                 true,
                 false,
                 BC_1st_FRONT,
                 BC_1st_REAR,
                 BC_2nd_FRONT,
                 BC_2nd_REAR,
                 BC_1_12,
                 BC_1_12,
                 DEFAULT_PARAMS,
         }},
        // PCR96
        {"EXP-PBC096",
         {
                 "PCR96",
                 true,
                 true,
                 false,
                 BC_1st_FRONT,
                 BC_1st_REAR,
                 BC_2nd_FRONT,
                 BC_2nd_REAR,
                 BC_1_96,
                 BC_1_96,
                 DEFAULT_PARAMS,
         }},
        // RAB
        {"SQK-RAB204", kit_rab},
        {"SQK-RAB201", kit_rab},
        // RBK
        {"SQK-RBK001",
         {
                 "RBK",
                 false,
                 false,
                 false,
                 RBK_FRONT,
                 RBK_REAR,
                 "",
                 "",
                 BC_1_12,
                 {},
                 DEFAULT_PARAMS,
         }},
        // RBK096
        {"SQK-RBK110-96", kit_rbk96},
        {"SQK-RBK111-96", kit_rbk96},
        // RBK096_kit14
        {"SQK-RBK114-96",
         {
                 "RBK096_kit14",
                 false,
                 false,
                 false,
                 RBK4_kit14_FRONT,
                 RBK4_kit14_REAR,
                 "",
                 "",
                 RBK_1_96,
                 {},
                 RBK114_PARAMS,
         }},
        // RBK24
        {"SQK-RBK111-24",
         {
                 "RBK24",
                 false,
                 false,
                 false,
                 RBK4_FRONT,
                 RBK4_REAR,
                 "",
                 "",
                 BC_1_24,
                 {},
                 DEFAULT_PARAMS,
         }},
        // RBK24_kit14
        {"SQK-RBK114-24",
         {
                 "RBK24_kit14",
                 false,
                 false,
                 false,
                 RBK4_kit14_FRONT,
                 RBK4_kit14_REAR,
                 "",
                 "",
                 BC_1_24,
                 {},
                 RBK114_PARAMS,
         }},
        //  RBK4
        {"SQK-RBK004", kit_rbk4},
        {"VSK-PTC001", kit_rbk4},
        {"VSK-VPS001", kit_rbk4},
        // RLB
        {"SQK-RPB004", kit_rlb},
        {"SQK-RLB001", kit_rlb},
        // RPB24-Kit14
        {"SQK-RPB114-24",
         {
                 "RPB24-Kit14",
                 true,
                 false,
                 false,
                 RLB_FRONT,
                 RLB_REAR,
                 "",
                 "",
                 BC2_1_24,
                 {},
                 DEFAULT_PARAMS,
         }},
        // VMK
        {"VSK-VMK001",
         {
                 "VMK",
                 false,
                 false,
                 false,
                 RBK_FRONT,
                 RBK_REAR,
                 "",
                 "",
                 {"BC01", "BC02", "BC03", "BC04"},
                 {},
                 DEFAULT_PARAMS,
         }},
        // VMK4
        {"VSK-VMK004",
         {
                 "VMK4",
                 false,
                 false,
                 false,
                 RBK4_FRONT,
                 RBK4_REAR,
                 "",
                 "",
                 {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06", "BC07", "BC08", "BC09", "BC10"},
                 {},
                 DEFAULT_PARAMS,
         }},
        {"TWIST-96A-UDI",
         {
                 "PGx",
                 true,
                 true,
                 false,
                 "CATACGAGAT",
                 "GTGACTGGAG",
                 "AGATCTACAC",
                 "ACACTCTTTC",
                 {"AA01F_01", "AB01F_02", "AC01F_03", "AD01F_04", "AE01F_05", "AF01F_06",
                  "AG01F_07", "AH01F_08", "AA02F_09", "AB02F_10", "AC02F_11", "AD02F_12",
                  "AE02F_13", "AF02F_14", "AG02F_15", "AH02F_16", "AA03F_17", "AB03F_18",
                  "AC03F_19", "AD03F_20", "AE03F_21", "AF03F_22", "AG03F_23", "AH03F_24",
                  "AA04F_25", "AB04F_26", "AC04F_27", "AD04F_28", "AE04F_29", "AF04F_30",
                  "AG04F_31", "AH04F_32", "AA05F_33", "AB05F_34", "AC05F_35", "AD05F_36",
                  "AE05F_37", "AF05F_38", "AG05F_39", "AH05F_40", "AA06F_41", "AB06F_42",
                  "AC06F_43", "AD06F_44", "AE06F_45", "AF06F_46", "AG06F_47", "AH06F_48",
                  "AA07F_49", "AB07F_50", "AC07F_51", "AD07F_52", "AE07F_53", "AF07F_54",
                  "AG07F_55", "AH07F_56", "AA08F_57", "AB08F_58", "AC08F_59", "AD08F_60",
                  "AE08F_61", "AF08F_62", "AG08F_63", "AH08F_64", "AA09F_65", "AB09F_66",
                  "AC09F_67", "AD09F_68", "AE09F_69", "AF09F_70", "AG09F_71", "AH09F_72",
                  "AA10F_73", "AB10F_74", "AC10F_75", "AD10F_76", "AE10F_77", "AF10F_78",
                  "AG10F_79", "AH10F_80", "AA11F_81", "AB11F_82", "AC11F_83", "AD11F_84",
                  "AE11F_85", "AF11F_86", "AG11F_87", "AH11F_88", "AA12F_89", "AB12F_90",
                  "AC12F_91", "AD12F_92", "AE12F_93", "AF12F_94", "AG12F_95", "AH12F_96"},
                 {"AA01R_01", "AB01R_02", "AC01R_03", "AD01R_04", "AE01R_05", "AF01R_06",
                  "AG01R_07", "AH01R_08", "AA02R_09", "AB02R_10", "AC02R_11", "AD02R_12",
                  "AE02R_13", "AF02R_14", "AG02R_15", "AH02R_16", "AA03R_17", "AB03R_18",
                  "AC03R_19", "AD03R_20", "AE03R_21", "AF03R_22", "AG03R_23", "AH03R_24",
                  "AA04R_25", "AB04R_26", "AC04R_27", "AD04R_28", "AE04R_29", "AF04R_30",
                  "AG04R_31", "AH04R_32", "AA05R_33", "AB05R_34", "AC05R_35", "AD05R_36",
                  "AE05R_37", "AF05R_38", "AG05R_39", "AH05R_40", "AA06R_41", "AB06R_42",
                  "AC06R_43", "AD06R_44", "AE06R_45", "AF06R_46", "AG06R_47", "AH06R_48",
                  "AA07R_49", "AB07R_50", "AC07R_51", "AD07R_52", "AE07R_53", "AF07R_54",
                  "AG07R_55", "AH07R_56", "AA08R_57", "AB08R_58", "AC08R_59", "AD08R_60",
                  "AE08R_61", "AF08R_62", "AG08R_63", "AH08R_64", "AA09R_65", "AB09R_66",
                  "AC09R_67", "AD09R_68", "AE09R_69", "AF09R_70", "AG09R_71", "AH09R_72",
                  "AA10R_73", "AB10R_74", "AC10R_75", "AD10R_76", "AE10R_77", "AF10R_78",
                  "AG10R_79", "AH10R_80", "AA11R_81", "AB11R_82", "AC11R_83", "AD11R_84",
                  "AE11R_85", "AF11R_86", "AG11R_87", "AH11R_88", "AA12R_89", "AB12R_90",
                  "AC12R_91", "AD12R_92", "AE12R_93", "AF12R_94", "AG12R_95", "AH12R_96"},
                 TWIST_PARAMS,
         }},
        {"TWIST-16-UDI",
         {
                 "PGx",
                 true,
                 true,
                 false,
                 "CATACGAGAT",
                 "GTGACTGGAG",
                 "AGATCTACAC",
                 "ACACTCTTTC",
                 {"16X_A01F_01", "16X_B01F_02", "16X_C01F_03", "16X_D01F_04", "16X_E01F_05",
                  "16X_F01F_06", "16X_G01F_07", "16X_H01F_08", "16X_A02F_09", "16X_B02F_10",
                  "16X_C02F_11", "16X_D02F_12", "16X_E02F_13", "16X_F02F_14", "16X_G02F_15",
                  "16X_H02F_16"},
                 {"16X_A01R_01", "16X_B01R_02", "16X_C01R_03", "16X_D01R_04", "16X_E01R_05",
                  "16X_F01R_06", "16X_G01R_07", "16X_H01R_08", "16X_A02R_09", "16X_B02R_10",
                  "16X_C02R_11", "16X_D02R_12", "16X_E02R_13", "16X_F02R_14", "16X_G02R_15",
                  "16X_H02R_16"},
                 TWIST_PARAMS,
         }},
};

std::unordered_map<std::string, std::string> barcodes = {
        // BC** barcodes.
        {"BC01", "AAGAAAGTTGTCGGTGTCTTTGTG"},
        {"BC02", "TCGATTCCGTTTGTAGTCGTCTGT"},
        {"BC03", "GAGTCTTGTGTCCCAGTTACCAGG"},
        {"BC04", "TTCGGATTCTATCGTGTTTCCCTA"},
        {"BC05", "CTTGTCCAGGGTTTGTGTAACCTT"},
        {"BC06", "TTCTCGCAAAGGCAGAAAGTAGTC"},
        {"BC07", "GTGTTACCGTGGGAATGAATCCTT"},
        {"BC08", "TTCAGGGAACAAACCAAGTTACGT"},
        {"BC09", "AACTAGGCACAGCGAGTCTTGGTT"},
        {"BC10", "AAGCGTTGAAACCTTTGTCCTCTC"},
        {"BC11", "GTTTCATCTATCGGAGGGAATGGA"},
        {"BC12", "CAGGTAGAAAGAAGCAGAATCGGA"},
        {"RLB12A", "GTTGAGTTACAAAGCACCGATCAG"},
        {"BC13", "AGAACGACTTCCATACTCGTGTGA"},
        {"BC14", "AACGAGTCTCTTGGGACCCATAGA"},
        {"BC15", "AGGTCTACCTCGCTAACACCACTG"},
        {"BC16", "CGTCAACTGACAGTGGTTCGTACT"},
        {"BC17", "ACCCTCCAGGAAAGTACCTCTGAT"},
        {"BC18", "CCAAACCCAACAACCTAGATAGGC"},
        {"BC19", "GTTCCTCGTGCAGTGTCAAGAGAT"},
        {"BC20", "TTGCGTCCTGTTACGAGAACTCAT"},
        {"BC21", "GAGCCTCTCATTGTCCGTTCTCTA"},
        {"BC22", "ACCACTGCCATGTATCAAAGTACG"},
        {"BC23", "CTTACTACCCAGTGAACCTCCTCG"},
        {"BC24", "GCATAGTTCTGCATGATGGGTTAG"},
        {"BC25", "GTAAGTTGGGTATGCAACGCAATG"},
        {"BC26", "CATACAGCGACTACGCATTCTCAT"},
        {"RBK26", "ACTATGCCTTTCCGTGAAACAGTT"},
        {"BC27", "CGACGGTTAGATTCACCTCTTACA"},
        {"BC28", "TGAAACCTAAGAAGGCACCGTATC"},
        {"BC29", "CTAGACACCTTGGGTTGACAGACC"},
        {"BC30", "TCAGTGAGGATCTACTTCGACCCA"},
        {"BC31", "TGCGTACAGCAATCAGTTACATTG"},
        {"BC32", "CCAGTAGAAGTCCGACAACGTCAT"},
        {"BC33", "CAGACTTGGTACGGTTGGGTAACT"},
        {"BC34", "GGACGAAGAACTCAAGTCAAAGGC"},
        {"BC35", "CTACTTACGAAGCTGAGGGACTGC"},
        {"BC36", "ATGTCCCAGTTAGAGGAGGAAACA"},
        {"BC37", "GCTTGCGATTGATGCTTAGTATCA"},
        {"BC38", "ACCACAGGAGGACGATACAGAGAA"},
        {"BC39", "CCACAGTGTCAACTAGAGCCTCTC"},
        {"RBK39", "TCTGCCACACACTCGTAAGTCCTT"},
        {"BC40", "TAGTTTGGATGACCAAGGATAGCC"},
        {"RBK40", "GTCGATACTGGACCTATCCCTTGG"},
        {"BC41", "GGAGTTCGTCCAGAGAAGTACACG"},
        {"BC42", "CTACGTGTAAGGCATACCTGCCAG"},
        {"BC43", "CTTTCGTTGTTGACTCGACGGTAG"},
        {"BC44", "AGTAGAAAGGGTTCCTTCCCACTC"},
        {"BC45", "GATCCAACAGAGATGCCTTCAGTG"},
        {"BC46", "GCTGTGTTCCACTTCATTCTCCTG"},
        {"BC47", "GTGCAACTTTCCCACAGGTAGTTC"},
        {"BC48", "CATCTGGAACGTGGTACACCTGTA"},
        {"RBK48", "GAGTCCGTGACAACTTCTGAAAGC"},
        {"BC49", "ACTGGTGCAGCTTTGAACATCTAG"},
        {"BC50", "ATGGACTTTGGTAACTTCCTGCGT"},
        {"BC51", "GTTGAATGAGCCTACTGGGTCCTC"},
        {"BC52", "TGAGAGACAAGATTGTTCGTGGAC"},
        {"BC53", "AGATTCAGACCGTCTCATGCAAAG"},
        {"BC54", "CAAGAGCTTTGACTAAGGAGCATG"},
        {"RBK54", "GGGTGCCAACTACATACCAAACCT"},
        {"BC55", "TGGAAGATGAGACCCTGATCTACG"},
        {"BC56", "TCACTACTCAACAGGTGGCATGAA"},
        {"BC57", "GCTAGGTCAATCTCCTTCGGAAGT"},
        {"BC58", "CAGGTTACTCCTCCGTGAGTCTGA"},
        {"BC59", "TCAATCAAGAAGGGAAAGCAAGGT"},
        {"BC60", "CATGTTCAACCAAGGCTTCTATGG"},
        {"RBK60", "GAACCCTACTTTGGACAGACACCT"},
        {"BC61", "AGAGGGTACTATGTGCCTCAGCAC"},
        {"BC62", "CACCCACACTTACTTCAGGACGTA"},
        {"BC63", "TTCTGAAGTTCCTGGGTCTTGAAC"},
        {"BC64", "GACAGACACCGTTCATCGACTTTC"},
        {"BC65", "TTCTCAGTCTTCCTCCAGACAAGG"},
        {"BC66", "CCGATCCTTGTGGCTTCTAACTTC"},
        {"BC67", "GTTTGTCATACTCGTGTGCTCACC"},
        {"BC68", "GAATCTAAGCAAACACGAAGGTGG"},
        {"BC69", "TACAGTCCGAGCCTCATGTGATCT"},
        {"BC70", "ACCGAGATCCTACGAATGGAGTGT"},
        {"BC71", "CCTGGGAGCATCAGGTAGTAACAG"},
        {"BC72", "TAGCTGACTGTCTTCCATACCGAC"},
        {"BC73", "AAGAAACAGGATGACAGAACCCTC"},
        {"BC74", "TACAAGCATCCCAACACTTCCACT"},
        {"BC75", "GACCATTGTGATGAACCCTGTTGT"},
        {"BC76", "ATGCTTGTTACATCAACCCTGGAC"},
        {"BC77", "CGACCTGTTTCTCAGGGATACAAC"},
        {"BC78", "AACAACCGAACCTTTGAATCAGAA"},
        {"BC79", "TCTCGGAGATAGTTCTCACTGCTG"},
        {"BC80", "CGGATGAACATAGGATAGCGATTC"},
        {"BC81", "CCTCATCTTGTGAAGTTGTTTCGG"},
        {"BC82", "ACGGTATGTCGAGTTCCAGGACTA"},
        {"BC83", "TGGCTTGATCTAGGTAAGGTCGAA"},
        {"BC84", "GTAGTGGACCTAGAACCTGTGCCA"},
        {"BC85", "AACGGAGGAGTTAGTTGGATGATC"},
        {"BC86", "AGGTGATCCCAACAAGCGTAAGTA"},
        {"BC87", "TACATGCTCCTGTTGTTAGGGAGG"},
        {"BC88", "TCTTCTACTACCGATCCGAAGCAG"},
        {"BC89", "ACAGCATCAATGTTTGGCTAGTTG"},
        {"BC90", "GATGTAGAGGGTACGGTTTGAGGC"},
        {"BC91", "GGCTCCATAGGAACTCACGCTACT"},
        {"BC92", "TTGTGAGTGGAAAGATACAGGACC"},
        {"BC93", "AGTTTCCATCACTTCAGACTTGGG"},
        {"BC94", "GATTGTCCTCAAACTGCCACCTAC"},
        {"BC95", "CCTGTCTGGAAGAAGAATGGACTT"},
        {"BC96", "CTGAACGGTCATAGAGTCCACCAT"},
        // BP** barcodes.
        {"BP01", "CAAGAAAGTTGTCGGTGTCTTTGTGAC"},
        {"BP02", "CTCGATTCCGTTTGTAGTCGTCTGTAC"},
        {"BP03", "CGAGTCTTGTGTCCCAGTTACCAGGAC"},
        {"BP04", "CTTCGGATTCTATCGTGTTTCCCTAAC"},
        {"BP05", "CCTTGTCCAGGGTTTGTGTAACCTTAC"},
        {"BP06", "CTTCTCGCAAAGGCAGAAAGTAGTCAC"},
        {"BP07", "CGTGTTACCGTGGGAATGAATCCTTAC"},
        {"BP08", "CTTCAGGGAACAAACCAAGTTACGTAC"},
        {"BP09", "CAACTAGGCACAGCGAGTCTTGGTTAC"},
        {"BP10", "CAAGCGTTGAAACCTTTGTCCTCTCAC"},
        {"BP11", "CGTTTCATCTATCGGAGGGAATGGAAC"},
        {"BP12", "CCAGGTAGAAAGAAGCAGAATCGGAAC"},
        {"BP13", "CAGAACGACTTCCATACTCGTGTGAAC"},
        {"BP14", "CAACGAGTCTCTTGGGACCCATAGAAC"},
        {"BP15", "CAGGTCTACCTCGCTAACACCACTGAC"},
        {"BP16", "CCGTCAACTGACAGTGGTTCGTACTAC"},
        {"BP17", "CACCCTCCAGGAAAGTACCTCTGATAC"},
        {"BP18", "CCCAAACCCAACAACCTAGATAGGCAC"},
        {"BP19", "CGTTCCTCGTGCAGTGTCAAGAGATAC"},
        {"BP20", "CTTGCGTCCTGTTACGAGAACTCATAC"},
        {"BP21", "CGAGCCTCTCATTGTCCGTTCTCTAAC"},
        {"BP22", "CACCACTGCCATGTATCAAAGTACGAC"},
        {"BP23", "CCTTACTACCCAGTGAACCTCCTCGAC"},
        {"BP24", "CGCATAGTTCTGCATGATGGGTTAGAC"},
        // NB** barcodes.
        {"NB01", "CACAAAGACACCGACAACTTTCTT"},
        {"NB02", "ACAGACGACTACAAACGGAATCGA"},
        {"NB03", "CCTGGTAACTGGGACACAAGACTC"},
        {"NB04", "TAGGGAAACACGATAGAATCCGAA"},
        {"NB05", "AAGGTTACACAAACCCTGGACAAG"},
        {"NB06", "GACTACTTTCTGCCTTTGCGAGAA"},
        {"NB07", "AAGGATTCATTCCCACGGTAACAC"},
        {"NB08", "ACGTAACTTGGTTTGTTCCCTGAA"},
        {"NB09", "AACCAAGACTCGCTGTGCCTAGTT"},
        {"NB10", "GAGAGGACAAAGGTTTCAACGCTT"},
        {"NB11", "TCCATTCCCTCCGATAGATGAAAC"},
        {"NB12", "TCCGATTCTGCTTCTTTCTACCTG"},
        {"NB13", "AGAACGACTTCCATACTCGTGTGA"},
        {"NB14", "AACGAGTCTCTTGGGACCCATAGA"},
        {"NB15", "AGGTCTACCTCGCTAACACCACTG"},
        {"NB16", "CGTCAACTGACAGTGGTTCGTACT"},
        {"NB17", "ACCCTCCAGGAAAGTACCTCTGAT"},
        {"NB18", "CCAAACCCAACAACCTAGATAGGC"},
        {"NB19", "GTTCCTCGTGCAGTGTCAAGAGAT"},
        {"NB20", "TTGCGTCCTGTTACGAGAACTCAT"},
        {"NB21", "GAGCCTCTCATTGTCCGTTCTCTA"},
        {"NB22", "ACCACTGCCATGTATCAAAGTACG"},
        {"NB23", "CTTACTACCCAGTGAACCTCCTCG"},
        {"NB24", "GCATAGTTCTGCATGATGGGTTAG"},
        {"NB25", "GTAAGTTGGGTATGCAACGCAATG"},
        {"NB26", "CATACAGCGACTACGCATTCTCAT"},
        {"NB27", "CGACGGTTAGATTCACCTCTTACA"},
        {"NB28", "TGAAACCTAAGAAGGCACCGTATC"},
        {"NB29", "CTAGACACCTTGGGTTGACAGACC"},
        {"NB30", "TCAGTGAGGATCTACTTCGACCCA"},
        {"NB31", "TGCGTACAGCAATCAGTTACATTG"},
        {"NB32", "CCAGTAGAAGTCCGACAACGTCAT"},
        {"NB33", "CAGACTTGGTACGGTTGGGTAACT"},
        {"NB34", "GGACGAAGAACTCAAGTCAAAGGC"},
        {"NB35", "CTACTTACGAAGCTGAGGGACTGC"},
        {"NB36", "ATGTCCCAGTTAGAGGAGGAAACA"},
        {"NB37", "GCTTGCGATTGATGCTTAGTATCA"},
        {"NB38", "ACCACAGGAGGACGATACAGAGAA"},
        {"NB39", "CCACAGTGTCAACTAGAGCCTCTC"},
        {"NB40", "TAGTTTGGATGACCAAGGATAGCC"},
        {"NB41", "GGAGTTCGTCCAGAGAAGTACACG"},
        {"NB42", "CTACGTGTAAGGCATACCTGCCAG"},
        {"NB43", "CTTTCGTTGTTGACTCGACGGTAG"},
        {"NB44", "AGTAGAAAGGGTTCCTTCCCACTC"},
        {"NB45", "GATCCAACAGAGATGCCTTCAGTG"},
        {"NB46", "GCTGTGTTCCACTTCATTCTCCTG"},
        {"NB47", "GTGCAACTTTCCCACAGGTAGTTC"},
        {"NB48", "CATCTGGAACGTGGTACACCTGTA"},
        {"NB49", "ACTGGTGCAGCTTTGAACATCTAG"},
        {"NB50", "ATGGACTTTGGTAACTTCCTGCGT"},
        {"NB51", "GTTGAATGAGCCTACTGGGTCCTC"},
        {"NB52", "TGAGAGACAAGATTGTTCGTGGAC"},
        {"NB53", "AGATTCAGACCGTCTCATGCAAAG"},
        {"NB54", "CAAGAGCTTTGACTAAGGAGCATG"},
        {"NB55", "TGGAAGATGAGACCCTGATCTACG"},
        {"NB56", "TCACTACTCAACAGGTGGCATGAA"},
        {"NB57", "GCTAGGTCAATCTCCTTCGGAAGT"},
        {"NB58", "CAGGTTACTCCTCCGTGAGTCTGA"},
        {"NB59", "TCAATCAAGAAGGGAAAGCAAGGT"},
        {"NB60", "CATGTTCAACCAAGGCTTCTATGG"},
        {"NB61", "AGAGGGTACTATGTGCCTCAGCAC"},
        {"NB62", "CACCCACACTTACTTCAGGACGTA"},
        {"NB63", "TTCTGAAGTTCCTGGGTCTTGAAC"},
        {"NB64", "GACAGACACCGTTCATCGACTTTC"},
        {"NB65", "TTCTCAGTCTTCCTCCAGACAAGG"},
        {"NB66", "CCGATCCTTGTGGCTTCTAACTTC"},
        {"NB67", "GTTTGTCATACTCGTGTGCTCACC"},
        {"NB68", "GAATCTAAGCAAACACGAAGGTGG"},
        {"NB69", "TACAGTCCGAGCCTCATGTGATCT"},
        {"NB70", "ACCGAGATCCTACGAATGGAGTGT"},
        {"NB71", "CCTGGGAGCATCAGGTAGTAACAG"},
        {"NB72", "TAGCTGACTGTCTTCCATACCGAC"},
        {"NB73", "AAGAAACAGGATGACAGAACCCTC"},
        {"NB74", "TACAAGCATCCCAACACTTCCACT"},
        {"NB75", "GACCATTGTGATGAACCCTGTTGT"},
        {"NB76", "ATGCTTGTTACATCAACCCTGGAC"},
        {"NB77", "CGACCTGTTTCTCAGGGATACAAC"},
        {"NB78", "AACAACCGAACCTTTGAATCAGAA"},
        {"NB79", "TCTCGGAGATAGTTCTCACTGCTG"},
        {"NB80", "CGGATGAACATAGGATAGCGATTC"},
        {"NB81", "CCTCATCTTGTGAAGTTGTTTCGG"},
        {"NB82", "ACGGTATGTCGAGTTCCAGGACTA"},
        {"NB83", "TGGCTTGATCTAGGTAAGGTCGAA"},
        {"NB84", "GTAGTGGACCTAGAACCTGTGCCA"},
        {"NB85", "AACGGAGGAGTTAGTTGGATGATC"},
        {"NB86", "AGGTGATCCCAACAAGCGTAAGTA"},
        {"NB87", "TACATGCTCCTGTTGTTAGGGAGG"},
        {"NB88", "TCTTCTACTACCGATCCGAAGCAG"},
        {"NB89", "ACAGCATCAATGTTTGGCTAGTTG"},
        {"NB90", "GATGTAGAGGGTACGGTTTGAGGC"},
        {"NB91", "GGCTCCATAGGAACTCACGCTACT"},
        {"NB92", "TTGTGAGTGGAAAGATACAGGACC"},
        {"NB93", "AGTTTCCATCACTTCAGACTTGGG"},
        {"NB94", "GATTGTCCTCAAACTGCCACCTAC"},
        {"NB95", "CCTGTCTGGAAGAAGAATGGACTT"},
        {"NB96", "CTGAACGGTCATAGAGTCCACCAT"},
        // MAB114 barcodes
        {"TP01", "GCACCTGGAACTTGTGCCTTCCAC"},
        {"TP02", "CCGAAATAGGTTATCTGTTGTTGT"},
        {"TP03", "ATCAATCGCTGGACGATGGATTAG"},
        {"TP04", "CCACCCGCTCCTGCCGGTGGGCGT"},
        {"TP05", "AGACTCTTGGGCTCGCCACGTCCC"},
        {"TP06", "TCTGTATCCGGAGACGGGATGGAC"},
        {"TP07", "TTTCGGATCAATCGACCGCAAACG"},
        {"TP08", "ACTCAAACATTCTGTTAGATCGCG"},
        {"TP09", "AAATGGAACCCGGATATGTTTACT"},
        {"TP10", "TAAATCGACCTATGATGAACACAG"},
        {"TP11", "ACATGTTGGAGTGAAAGTCGGGTA"},
        {"TP12", "CCTGGACCACGATCATTGTAACAT"},
        {"TP13", "TATGGTGGATCTCCCTCTATCTTC"},
        {"TP14", "AAGTAAATGGGACGCCCACTCCGA"},
        {"TP15", "TGTTCGCGGCTTGATCTAATATTA"},
        {"TP16", "AGAGAGCTTCCCGGGAGGGTGGTC"},
        {"TP17", "TTGTGAATATCTGTCACAAACACC"},
        {"TP18", "CAATCGTACCAGGGAACATAAAGT"},
        {"TP19", "CACACCCAAACAATATGGACCCGT"},
        {"TP20", "AATAACCACATCCGCCCTCCGCAC"},
        {"TP21", "TCCTAATAATGTGTAGATCGGTCC"},
        {"TP22", "AGTCGATGGAACAAGAGAAGTTAT"},
        {"TP23", "AAACTCACTGTATGTCGTTTCTAT"},
        {"TP24", "TGACATCACTGATCGAGGAAGATC"},
        // Twist 96x A-plate barcodes
        {"AA01F_01", "GCTGAAGATA"},
        {"AA01R_01", "CCAATATTCG"},
        {"AB01F_02", "TATCCGTGCA"},
        {"AB01R_02", "CGCAGACAAC"},
        {"AC01F_03", "TCTATCAACC"},
        {"AC01R_03", "TCGGAGCAGA"},
        {"AD01F_04", "AGGCAGGAGT"},
        {"AD01R_04", "GAGTCCGTAG"},
        {"AE01F_05", "CGACTATCGG"},
        {"AE01R_05", "ATGTTCACGT"},
        {"AF01F_06", "TTCGATCTTG"},
        {"AF01R_06", "TTCGATGGTT"},
        {"AG01F_07", "GAAGGAGCCT"},
        {"AG01R_07", "TATCCGTGCA"},
        {"AH01F_08", "CTATCCGTAT"},
        {"AH01R_08", "AAGCGCAGAG"},
        {"AA02F_09", "TGAGGCTATT"},
        {"AA02R_09", "CCGACTTAGT"},
        {"AB02F_10", "CCGATTGCAG"},
        {"AB02R_10", "TTCTGCATCG"},
        {"AC02F_11", "ATAACTCAGG"},
        {"AC02R_11", "GGAAGTGCCA"},
        {"AD02F_12", "TCTGGACGTC"},
        {"AD02R_12", "AGATTCAACC"},
        {"AE02F_13", "CCGATTATTC"},
        {"AE02R_13", "TTCAGGAGAT"},
        {"AF02F_14", "ACACACTCCG"},
        {"AF02R_14", "AAGGCGTCTG"},
        {"AG02F_15", "CGGTCGGTAA"},
        {"AG02R_15", "ACGCTTGACA"},
        {"AH02F_16", "GGCGAACACT"},
        {"AH02R_16", "CATGAAGTGA"},
        {"AA03F_17", "AAGAACGTAG"},
        {"AA03R_17", "TTACGACCTG"},
        {"AB03F_18", "TTCGTGTCGA"},
        {"AB03R_18", "ATGCAAGCCG"},
        {"AC03F_19", "AAGTTATCGG"},
        {"AC03R_19", "CTCCGTATAC"},
        {"AD03F_20", "CGATGTCCAA"},
        {"AD03R_20", "GAATCTGGTC"},
        {"AE03F_21", "TCTCAACGTT"},
        {"AE03R_21", "CGGTCGGTAA"},
        {"AF03F_22", "TTCACTGGCC"},
        {"AF03R_22", "TCTGCTAATG"},
        {"AG03F_23", "CCGGAGACAT"},
        {"AG03R_23", "CTCTTATTCG"},
        {"AH03F_24", "GAACGCCTTC"},
        {"AH03R_24", "CACCTCTAGC"},
        {"AA04F_25", "TCTAGGAACA"},
        {"AA04R_25", "TTACTTACCG"},
        {"AB04F_26", "ACCTCGAGAG"},
        {"AB04R_26", "CTATGCCTTA"},
        {"AC04F_27", "TACCGTACAG"},
        {"AC04R_27", "GGAAGGTACG"},
        {"AD04F_28", "TTGCCATAAG"},
        {"AD04R_28", "GAGGAGACGT"},
        {"AE04F_29", "GCTATGCGGA"},
        {"AE04R_29", "ACGCAAGGCA"},
        {"AF04F_30", "AGGTGCTTGC"},
        {"AF04R_30", "TATCCTGACG"},
        {"AG04F_31", "TAGGACAGGC"},
        {"AG04R_31", "GAAGACCGCT"},
        {"AH04F_32", "GATAGACAGT"},
        {"AH04R_32", "CAACGTGGAC"},
        {"AA05F_33", "TACATGGACG"},
        {"AA05R_33", "TAAGTGCTCG"},
        {"AB05F_34", "TTGCAGTTAG"},
        {"AB05R_34", "CACATCGTAG"},
        {"AC05F_35", "ACCACAAGCA"},
        {"AC05R_35", "ACTACCGAGG"},
        {"AD05F_36", "TGTGCTTACA"},
        {"AD05R_36", "GATGTGTTCT"},
        {"AE05F_37", "ACGCAACGAG"},
        {"AE05R_37", "AAGTGTCGTA"},
        {"AF05F_38", "CACCTCTAGC"},
        {"AF05R_38", "GGAGAACCAC"},
        {"AG05F_39", "TTCTCCGCTT"},
        {"AG05R_39", "TGTACGAACT"},
        {"AH05F_40", "CAGCGTCATT"},
        {"AH05R_40", "GGATGAGTGC"},
        {"AA06F_41", "CGCGTACCAA"},
        {"AA06R_41", "TAGTAGGACA"},
        {"AB06F_42", "TTCACCTTCA"},
        {"AB06R_42", "ACGCCTCGTT"},
        {"AC06F_43", "AAGCCACTAC"},
        {"AC06R_43", "CACCGCTGTT"},
        {"AD06F_44", "TTCTGTTACG"},
        {"AD06R_44", "TCTATAGCGG"},
        {"AE06F_45", "TTATGGCCTT"},
        {"AE06R_45", "CCGATGGACA"},
        {"AF06F_46", "GGTCTATGAA"},
        {"AF06R_46", "TTCAACATGC"},
        {"AG06F_47", "TCGGAGTTGG"},
        {"AG06R_47", "GGAGTAACGC"},
        {"AH06F_48", "CATACTCGTG"},
        {"AH06R_48", "AGCCTTAGCG"},
        {"AA07F_49", "TTGGTAGCGG"},
        {"AA07R_49", "TTACCTCAGT"},
        {"AB07F_50", "GGAGGTTCAG"},
        {"AB07R_50", "CAGGCATTGT"},
        {"AC07F_51", "TAACAAGGCC"},
        {"AC07R_51", "GTGTTCCACG"},
        {"AD07F_52", "TCTGCGTTAA"},
        {"AD07R_52", "TTGATCCGCC"},
        {"AE07F_53", "CGCACTACCT"},
        {"AE07R_53", "GGAGGCTGAT"},
        {"AF07F_54", "AAGTTACACG"},
        {"AF07R_54", "AACGTGACAA"},
        {"AG07F_55", "CGTCACAAGT"},
        {"AG07R_55", "CACAAGCTCC"},
        {"AH07F_56", "CAACGCATGG"},
        {"AH07R_56", "CCGTGTTGTC"},
        {"AA08F_57", "CGCTACAAGG"},
        {"AA08R_57", "TTGAGCCAGC"},
        {"AB08F_58", "TCACGTATGT"},
        {"AB08R_58", "GCGTTACAGA"},
        {"AC08F_59", "GGATATCAAG"},
        {"AC08R_59", "TCCAGACATT"},
        {"AD08F_60", "ACATCGGCTG"},
        {"AD08R_60", "TCGAACTCTT"},
        {"AE08F_61", "TAGCGCATGA"},
        {"AE08R_61", "ACCTTCTCGG"},
        {"AF08F_62", "TGGACGGAGT"},
        {"AF08R_62", "AGACGCCAAC"},
        {"AG08F_63", "CAAGGCTGTC"},
        {"AG08R_63", "CAACCGTAAT"},
        {"AH08F_64", "CAGATAACCG"},
        {"AH08R_64", "TTATGCGTTG"},
        {"AA09F_65", "CCGTGGAGTA"},
        {"AA09R_65", "CTATGAGAAC"},
        {"AB09F_66", "TGCCGGAAGT"},
        {"AB09R_66", "AAGTTACACG"},
        {"AC09F_67", "GCAGCTTCAC"},
        {"AC09R_67", "GCAATGTGAG"},
        {"AD09F_68", "AGAAGAGCAA"},
        {"AD09R_68", "CGAAGTCGCA"},
        {"AE09F_69", "TACGTGCGTT"},
        {"AE09R_69", "CCTGATTCAA"},
        {"AF09F_70", "CCTGCAGTAA"},
        {"AF09R_70", "TAGAACGTGC"},
        {"AG09F_71", "CCTCAACTGG"},
        {"AG09R_71", "TTCGCAAGGT"},
        {"AH09F_72", "TTAACGCACA"},
        {"AH09R_72", "TTAATGCCGA"},
        {"AA10F_73", "AAGCACTAGT"},
        {"AA10R_73", "AGAACAGAGT"},
        {"AB10F_74", "GTGTTCCACG"},
        {"AB10R_74", "CCATCTGTTC"},
        {"AC10F_75", "CCACTTCCAT"},
        {"AC10R_75", "TTCGTAGGTG"},
        {"AD10F_76", "TGTGATCTCA"},
        {"AD10R_76", "GCACGGTACA"},
        {"AE10F_77", "CACCAAGGAC"},
        {"AE10R_77", "TGTCAAGAGG"},
        {"AF10F_78", "TTCCACGCTC"},
        {"AF10R_78", "TCTAAGGTAC"},
        {"AG10F_79", "ACAGCGTGTG"},
        {"AG10R_79", "GAACGGAGAC"},
        {"AH10F_80", "TGTACAACCA"},
        {"AH10R_80", "CGCTACCATC"},
        {"AA11F_81", "TGTGAGTGAT"},
        {"AA11R_81", "TTACGGTAAC"},
        {"AB11F_82", "TCTACCTCCG"},
        {"AB11R_82", "TTCAGATGGA"},
        {"AC11F_83", "TTGTCAACTC"},
        {"AC11R_83", "TAGCATCTGT"},
        {"AD11F_84", "CAAGTTCGGC"},
        {"AD11R_84", "GGACGAGATC"},
        {"AE11F_85", "TGTGAGGCCT"},
        {"AE11R_85", "AGGTTCTGTT"},
        {"AF11F_86", "CTAACAGAGA"},
        {"AF11R_86", "CATACTCGTG"},
        {"AG11F_87", "AATCGTCGGA"},
        {"AG11R_87", "CCGGATACCA"},
        {"AH11F_88", "AACATAGCCT"},
        {"AH11R_88", "ATGTCCACCG"},
        {"AA12F_89", "CAAGAGAACG"},
        {"AA12R_89", "CACCAAGTGG"},
        {"AB12F_90", "CCATAGACAA"},
        {"AB12R_90", "TTGAGTACAC"},
        {"AC12F_91", "TGTATCCATC"},
        {"AC12R_91", "CGGTTCCGTA"},
        {"AD12F_92", "CGCCTAAGTG"},
        {"AD12R_92", "GGAGGTCCTA"},
        {"AE12F_93", "TAGCCAGTGT"},
        {"AE12R_93", "CCTGCTTGGA"},
        {"AF12F_94", "CAGTGGCGAT"},
        {"AF12R_94", "TTCACGTCAG"},
        {"AG12F_95", "TTCACGTCAG"},
        {"AG12R_95", "AACATAGCCT"},
        {"AH12F_96", "AGACGATTGA"},
        {"AH12R_96", "TGACATAGTC"},
        // Twist 16x barcodes
        {"16X_A01F_01", "AACTTCGTCT"},
        {"16X_A01R_01", "CGAGCACGTT"},
        {"16X_B01F_02", "CTAGTAGTGA"},
        {"16X_B01R_02", "TTCATAGGAC"},
        {"16X_C01F_03", "TTACCTCAGT"},
        {"16X_C01R_03", "CAGGTTATAC"},
        {"16X_D01F_04", "GCTAGATAAG"},
        {"16X_D01R_04", "TCTCATCATG"},
        {"16X_E01F_05", "CAATGGTGTA"},
        {"16X_E01R_05", "TCAGTCGTTG"},
        {"16X_F01F_06", "TTGACGTTAC"},
        {"16X_F01R_06", "CAACAACTCT"},
        {"16X_G01F_07", "TTCCAACCTA"},
        {"16X_G01R_07", "TCTAGGAACA"},
        {"16X_H01F_08", "AGGTGCACTT"},
        {"16X_H01R_08", "CGATATGCTA"},
        {"16X_A02F_09", "CTGTCGAGCA"},
        {"16X_A02R_09", "TTGGCTTGGT"},
        {"16X_B02F_10", "TAGAACACCT"},
        {"16X_B02R_10", "AAGAACGTAG"},
        {"16X_C02F_11", "ACCTTCTCGG"},
        {"16X_C02R_11", "AGCTCCACTG"},
        {"16X_D02F_12", "CTATTGCTTC"},
        {"16X_D02R_12", "TGTGCGATTC"},
        {"16X_E02F_13", "AGCGATTAAC"},
        {"16X_E02R_13", "CTCAATGCTC"},
        {"16X_F02F_14", "TGGAGTTACA"},
        {"16X_F02R_14", "TTGTTGTCAG"},
        {"16X_G02F_15", "TAGATACTGG"},
        {"16X_G02R_15", "TATCCGCGGT"},
        {"16X_H02F_16", "CTATGAGAAC"},
        {"16X_H02R_16", "CACAGCAAGA"},
};

std::unordered_set<std::string> custom_kit_names;
std::unordered_set<std::string> custom_barcode_names;

}  // namespace

const std::unordered_map<std::string, KitInfo>& get_kit_infos() { return kit_info_map; }

const KitInfo* get_kit_info(const std::string& kit_name) {
    const auto& barcode_kit_infos = get_kit_infos();
    auto kit_iter = barcode_kit_infos.find(kit_name);
    if (kit_iter == barcode_kit_infos.end()) {
        return nullptr;
    }
    return &kit_iter->second;
}

const std::unordered_map<std::string, std::string>& get_barcodes() { return barcodes; }

const std::unordered_set<std::string>& get_barcode_identifiers() {
    static auto identifiers = []() {
        std::unordered_set<std::string> ids;
        for (auto& [identifier, _] : barcodes) {
            ids.insert(identifier);
        }
        return ids;
    }();
    return identifiers;
}

void add_custom_barcode_kit(const std::string& kit_name, const KitInfo& custom_kit_info) {
    auto [_, success] = kit_info_map.insert({kit_name, custom_kit_info});
    if (!success) {
        std::string error =
                std::string("Custom kit name \"").append(kit_name).append("\" already exists.");
        throw std::runtime_error(error);
    }
    custom_kit_names.insert(kit_name);
}

void add_custom_barcodes(const std::unordered_map<std::string, std::string>& custom_barcodes) {
    std::set<std::string> duplicated_barcode_names;
    for (const auto& [barcode_name, barcode_seq] : custom_barcodes) {
        auto [_, success] = barcodes.insert({barcode_name, barcode_seq});
        if (!success) {
            duplicated_barcode_names.insert(barcode_name);
        }
    }

    if (!duplicated_barcode_names.empty()) {
        std::string error = std::string("Custom barcode names already exist:");
        for (const auto& name : duplicated_barcode_names) {
            error.append("\n").append(name);
        }
        throw std::runtime_error(error);
    }

    for (const auto& [barcode_name, _] : custom_barcodes) {
        custom_barcode_names.insert(barcode_name);
    }
}

void clear_custom_barcode_kits() {
    for (const auto& kit_name : custom_kit_names) {
        kit_info_map.erase(kit_name);
    }
}

void clear_custom_barcodes() {
    for (const auto& barcode_name : custom_barcode_names) {
        barcodes.erase(barcode_name);
    }
}

bool is_valid_barcode_kit(const std::string& kit_name) {
    const auto kit_info = barcode_kits::get_kit_info(kit_name);
    return kit_info != nullptr;
}

std::string barcode_kits_list_str() {
    std::vector<std::string> kit_names;
    kit_names.reserve(kit_info_map.size());
    for (auto& [kit_name, _] : kit_info_map) {
        kit_names.push_back(kit_name);
    }
    std::sort(kit_names.begin(), kit_names.end());
    return std::accumulate(kit_names.begin(), kit_names.end(), std::string(),
                           [](const auto& a, const auto& b) -> std::string {
                               return a + (a.empty() ? "" : " ") + b;
                           });
}

std::string normalize_barcode_name(const std::string& barcode_name) {
    std::string digits = "";
    // Normalize using only the digits at the end of the barcode name.
    bool found_digits = false;
    for (auto rit = barcode_name.rbegin(); rit != barcode_name.rend(); ++rit) {
        if (std::isdigit(static_cast<unsigned char>(*rit))) {
            digits += *rit;
            found_digits = true;
        } else if (found_digits) {
            break;
        }
    }

    std::reverse(digits.begin(), digits.end());
    return "barcode" + digits;
}

std::string generate_standard_barcode_name(const std::string& kit_name,
                                           const std::string& barcode_name) {
    return kit_name + "_" + normalize_barcode_name(barcode_name);
}

}  // namespace dorado::barcode_kits
