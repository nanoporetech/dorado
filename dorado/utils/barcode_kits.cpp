#include "barcode_kits.h"

#include <algorithm>
#include <numeric>

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

const std::string RBK4_kit14_FRONT = "C";
const std::string RBK4_kit14_REAR = "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA";

const std::string RLB_FRONT = "CCGTGAC";
const std::string RLB_REAR = "CGTTTTTCGTGCGCCGCTTC";

const std::string BC_1st_FRONT = "GGTGCTG";
const std::string BC_1st_REAR = "TTAACCTTTCTGTTGGTGCTGATATTGC";
const std::string BC_2nd_FRONT = "GGTGCTG";
const std::string BC_2nd_REAR = "TTAACCTACTTGCCTGTCGCTCTATCTTC";

const std::string NB_1st_FRONT = "AGGTTAA";
const std::string NB_1st_REAR = "CAGCACCT";
const std::string NB_2nd_FRONT = "ATTGCTAAGGTTAA";
const std::string NB_2nd_REAR = "CAGCACC";

const std::string LWB_1st_FRONT = "CCGTGAC";
const std::string LWB_1st_REAR = "ACTTGCCTGTCGCTCTATCTTC";
const std::string LWB_2nd_FRONT = "CCGTGAC";
const std::string LWB_2nd_REAR = "TTTCTGTTGGTGCTGATATTGC";

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

// Some arrangement names are just aliases of each other. This is because they were released
// as part of different kits, but they map to the same underlying arrangement.
const KitInfo kit_16S = {
        "16S",         true,         true,    RAB_1st_FRONT, RAB_1st_REAR,
        RAB_2nd_FRONT, RAB_2nd_REAR, BC_1_24, BC_1_24,
};

const KitInfo kit_lwb = {
        "LWB",         true,         true,    LWB_1st_FRONT, LWB_1st_REAR,
        LWB_2nd_FRONT, LWB_2nd_REAR, BC_1_12, BC_1_12,
};

const KitInfo kit_lwb24 = {
        "LWB24",       true,         true,    LWB_1st_FRONT, LWB_1st_REAR,
        LWB_2nd_FRONT, LWB_2nd_REAR, BC_1_24, BC_1_24,
};

const KitInfo kit_nb12 = {
        "NB12", true, true, NB_1st_FRONT, NB_1st_REAR, NB_2nd_FRONT, NB_2nd_REAR, NB_1_12, NB_1_12,
};

const KitInfo kit_nb24 = {
        "NB24", true, true, NB_1st_FRONT, NB_1st_REAR, NB_2nd_FRONT, NB_2nd_REAR, NB_1_24, NB_1_24,
};

const KitInfo kit_nb96 = {
        "NB96", true, true, NB_1st_FRONT, NB_1st_REAR, NB_2nd_FRONT, NB_2nd_REAR, NB_1_96, NB_1_96,
};

const KitInfo kit_rab = {
        "RAB",         true,         true,    RAB_1st_FRONT, RAB_1st_REAR,
        RAB_2nd_FRONT, RAB_2nd_REAR, BC_1_12, BC_1_12,
};

const KitInfo kit_rbk96 = {
        "RBK96", false, false, RBK4_FRONT, RBK4_REAR, "", "", RBK_1_96, {},
};

const KitInfo kit_rbk4 = {
        "RBK4", false, false, RBK4_FRONT, RBK4_REAR, "", "", BC_1_12, {},
};

const KitInfo kit_rlb = {
        "RLB", true, false, RLB_FRONT, RLB_REAR, "", "", BC_1_12A, {},
};

// Final map to go from kit name to actual barcode arrangement information.
const std::unordered_map<std::string, KitInfo> kit_info_map = {
        // SQK-16S024 && SQK-16S114-24
        {"SQK-16S024", kit_16S},
        {"SQK-16S114-24", kit_16S},
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
                 NB_1st_FRONT,
                 NB_1st_REAR,
                 NB_2nd_FRONT,
                 NB_2nd_REAR,
                 NB_13_24,
                 NB_13_24,
         }},
        // NB24
        {"SQK-NBD111-24", kit_nb24},
        {"SQK-NBD114-24", kit_nb24},
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
                 BC_1st_FRONT,
                 BC_1st_REAR,
                 BC_2nd_FRONT,
                 BC_2nd_REAR,
                 BC_1_12,
                 BC_1_12,
         }},
        // PCR96
        {"EXP-PBC096",
         {
                 "PCR96",
                 true,
                 true,
                 BC_1st_FRONT,
                 BC_1st_REAR,
                 BC_2nd_FRONT,
                 BC_2nd_REAR,
                 BC_1_96,
                 BC_1_96,
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
                 RBK_FRONT,
                 RBK_REAR,
                 "",
                 "",
                 BC_1_12,
                 {},
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
                 RBK4_kit14_FRONT,
                 RBK4_kit14_REAR,
                 "",
                 "",
                 RBK_1_96,
                 {},
         }},
        // RBK24
        {"SQK-RBK111-24",
         {
                 "RBK24",
                 false,
                 false,
                 RBK4_FRONT,
                 RBK4_REAR,
                 "",
                 "",
                 BC_1_24,
                 {},
         }},
        // RBK24_kit14
        {"SQK-RBK114-24",
         {
                 "RBK24_kit14",
                 false,
                 false,
                 RBK4_kit14_FRONT,
                 RBK4_kit14_REAR,
                 "",
                 "",
                 BC_1_24,
                 {},
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
                 RLB_FRONT,
                 RLB_REAR,
                 "",
                 "",
                 BC2_1_24,
                 {},
         }},
        // VMK
        {"VSK-VMK001",
         {
                 "VMK",
                 false,
                 false,
                 RBK_FRONT,
                 RBK_REAR,
                 "",
                 "",
                 {"BC01", "BC02", "BC03", "BC04"},
                 {},
         }},
        // VMK4
        {"VSK-VMK004",
         {"VMK4",
          false,
          false,
          RBK4_FRONT,
          RBK4_REAR,
          "",
          "",
          {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06", "BC07", "BC08", "BC09", "BC10"},
          {}}},
        {"TWIST-ALL",
         {
                 "PGx",
                 true,
                 true,
                 "AATGATACGGCGACCACCGAGATCTACAC",
                 "ACACTCTTTCCCTACACGACGCTCTTCCGATCT",
                 "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC",
                 "ATCTCGTATGCCGTCTTCTGCTTG",
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
         }},
};

const std::unordered_map<std::string, std::string> barcodes = {
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
        // Twist barcodes
        {"AA01F_01", "CCAATATTCG"},
        {"AA01R_01", "TATCTTCAGC"},
        {"AB01F_02", "CGCAGACAAC"},
        {"AB01R_02", "TGCACGGATA"},
        {"AC01F_03", "TCGGAGCAGA"},
        {"AC01R_03", "GGTTGATAGA"},
        {"AD01F_04", "GAGTCCGTAG"},
        {"AD01R_04", "ACTCCTGCCT"},
        {"AE01F_05", "ATGTTCACGT"},
        {"AE01R_05", "CCGATAGTCG"},
        {"AF01F_06", "TTCGATGGTT"},
        {"AF01R_06", "CAAGATCGAA"},
        {"AG01F_07", "TATCCGTGCA"},
        {"AG01R_07", "AGGCTCCTTC"},
        {"AH01F_08", "AAGCGCAGAG"},
        {"AH01R_08", "ATACGGATAG"},
        {"AA02F_09", "CCGACTTAGT"},
        {"AA02R_09", "AATAGCCTCA"},
        {"AB02F_10", "TTCTGCATCG"},
        {"AB02R_10", "CTGCAATCGG"},
        {"AC02F_11", "GGAAGTGCCA"},
        {"AC02R_11", "CCTGAGTTAT"},
        {"AD02F_12", "AGATTCAACC"},
        {"AD02R_12", "GACGTCCAGA"},
        {"AE02F_13", "TTCAGGAGAT"},
        {"AE02R_13", "GAATAATCGG"},
        {"AF02F_14", "AAGGCGTCTG"},
        {"AF02R_14", "CGGAGTGTGT"},
        {"AG02F_15", "ACGCTTGACA"},
        {"AG02R_15", "TTACCGACCG"},
        {"AH02F_16", "CATGAAGTGA"},
        {"AH02R_16", "AGTGTTCGCC"},
        {"AA03F_17", "TTACGACCTG"},
        {"AA03R_17", "CTACGTTCTT"},
        {"AB03F_18", "ATGCAAGCCG"},
        {"AB03R_18", "TCGACACGAA"},
        {"AC03F_19", "CTCCGTATAC"},
        {"AC03R_19", "CCGATAACTT"},
        {"AD03F_20", "GAATCTGGTC"},
        {"AD03R_20", "TTGGACATCG"},
        {"AE03F_21", "CGGTCGGTAA"},
        {"AE03R_21", "AACGTTGAGA"},
        {"AF03F_22", "TCTGCTAATG"},
        {"AF03R_22", "GGCCAGTGAA"},
        {"AG03F_23", "CTCTTATTCG"},
        {"AG03R_23", "ATGTCTCCGG"},
        {"AH03F_24", "CACCTCTAGC"},
        {"AH03R_24", "GAAGGCGTTC"},
        {"AA04F_25", "TTACTTACCG"},
        {"AA04R_25", "TGTTCCTAGA"},
        {"AB04F_26", "CTATGCCTTA"},
        {"AB04R_26", "CTCTCGAGGT"},
        {"AC04F_27", "GGAAGGTACG"},
        {"AC04R_27", "CTGTACGGTA"},
        {"AD04F_28", "GAGGAGACGT"},
        {"AD04R_28", "CTTATGGCAA"},
        {"AE04F_29", "ACGCAAGGCA"},
        {"AE04R_29", "TCCGCATAGC"},
        {"AF04F_30", "TATCCTGACG"},
        {"AF04R_30", "GCAAGCACCT"},
        {"AG04F_31", "GAAGACCGCT"},
        {"AG04R_31", "GCCTGTCCTA"},
        {"AH04F_32", "CAACGTGGAC"},
        {"AH04R_32", "ACTGTCTATC"},
        {"AA05F_33", "TAAGTGCTCG"},
        {"AA05R_33", "CGTCCATGTA"},
        {"AB05F_34", "CACATCGTAG"},
        {"AB05R_34", "CTAACTGCAA"},
        {"AC05F_35", "ACTACCGAGG"},
        {"AC05R_35", "TGCTTGTGGT"},
        {"AD05F_36", "GATGTGTTCT"},
        {"AD05R_36", "TGTAAGCACA"},
        {"AE05F_37", "AAGTGTCGTA"},
        {"AE05R_37", "CTCGTTGCGT"},
        {"AF05F_38", "GGAGAACCAC"},
        {"AF05R_38", "GCTAGAGGTG"},
        {"AG05F_39", "TGTACGAACT"},
        {"AG05R_39", "AAGCGGAGAA"},
        {"AH05F_40", "GGATGAGTGC"},
        {"AH05R_40", "AATGACGCTG"},
        {"AA06F_41", "TAGTAGGACA"},
        {"AA06R_41", "TTGGTACGCG"},
        {"AB06F_42", "ACGCCTCGTT"},
        {"AB06R_42", "TGAAGGTGAA"},
        {"AC06F_43", "CACCGCTGTT"},
        {"AC06R_43", "GTAGTGGCTT"},
        {"AD06F_44", "TCTATAGCGG"},
        {"AD06R_44", "CGTAACAGAA"},
        {"AE06F_45", "CCGATGGACA"},
        {"AE06R_45", "AAGGCCATAA"},
        {"AF06F_46", "TTCAACATGC"},
        {"AF06R_46", "TTCATAGACC"},
        {"AG06F_47", "GGAGTAACGC"},
        {"AG06R_47", "CCAACTCCGA"},
        {"AH06F_48", "AGCCTTAGCG"},
        {"AH06R_48", "CACGAGTATG"},
        {"AA07F_49", "TTACCTCAGT"},
        {"AA07R_49", "CCGCTACCAA"},
        {"AB07F_50", "CAGGCATTGT"},
        {"AB07R_50", "CTGAACCTCC"},
        {"AC07F_51", "GTGTTCCACG"},
        {"AC07R_51", "GGCCTTGTTA"},
        {"AD07F_52", "TTGATCCGCC"},
        {"AD07R_52", "TTAACGCAGA"},
        {"AE07F_53", "GGAGGCTGAT"},
        {"AE07R_53", "AGGTAGTGCG"},
        {"AF07F_54", "AACGTGACAA"},
        {"AF07R_54", "CGTGTAACTT"},
        {"AG07F_55", "CACAAGCTCC"},
        {"AG07R_55", "ACTTGTGACG"},
        {"AH07F_56", "CCGTGTTGTC"},
        {"AH07R_56", "CCATGCGTTG"},
        {"AA08F_57", "TTGAGCCAGC"},
        {"AA08R_57", "CCTTGTAGCG"},
        {"AB08F_58", "GCGTTACAGA"},
        {"AB08R_58", "ACATACGTGA"},
        {"AC08F_59", "TCCAGACATT"},
        {"AC08R_59", "CTTGATATCC"},
        {"AD08F_60", "TCGAACTCTT"},
        {"AD08R_60", "CAGCCGATGT"},
        {"AE08F_61", "ACCTTCTCGG"},
        {"AE08R_61", "TCATGCGCTA"},
        {"AF08F_62", "AGACGCCAAC"},
        {"AF08R_62", "ACTCCGTCCA"},
        {"AG08F_63", "CAACCGTAAT"},
        {"AG08R_63", "GACAGCCTTG"},
        {"AH08F_64", "TTATGCGTTG"},
        {"AH08R_64", "CGGTTATCTG"},
        {"AA09F_65", "CTATGAGAAC"},
        {"AA09R_65", "TACTCCACGG"},
        {"AB09F_66", "AAGTTACACG"},
        {"AB09R_66", "ACTTCCGGCA"},
        {"AC09F_67", "GCAATGTGAG"},
        {"AC09R_67", "GTGAAGCTGC"},
        {"AD09F_68", "CGAAGTCGCA"},
        {"AD09R_68", "TTGCTCTTCT"},
        {"AE09F_69", "CCTGATTCAA"},
        {"AE09R_69", "AACGCACGTA"},
        {"AF09F_70", "TAGAACGTGC"},
        {"AF09R_70", "TTACTGCAGG"},
        {"AG09F_71", "TTCGCAAGGT"},
        {"AG09R_71", "CCAGTTGAGG"},
        {"AH09F_72", "TTAATGCCGA"},
        {"AH09R_72", "TGTGCGTTAA"},
        {"AA10F_73", "AGAACAGAGT"},
        {"AA10R_73", "ACTAGTGCTT"},
        {"AB10F_74", "CCATCTGTTC"},
        {"AB10R_74", "CGTGGAACAC"},
        {"AC10F_75", "TTCGTAGGTG"},
        {"AC10R_75", "ATGGAAGTGG"},
        {"AD10F_76", "GCACGGTACA"},
        {"AD10R_76", "TGAGATCACA"},
        {"AE10F_77", "TGTCAAGAGG"},
        {"AE10R_77", "GTCCTTGGTG"},
        {"AF10F_78", "TCTAAGGTAC"},
        {"AF10R_78", "GAGCGTGGAA"},
        {"AG10F_79", "GAACGGAGAC"},
        {"AG10R_79", "CACACGCTGT"},
        {"AH10F_80", "CGCTACCATC"},
        {"AH10R_80", "TGGTTGTACA"},
        {"AA11F_81", "TTACGGTAAC"},
        {"AA11R_81", "ATCACTCACA"},
        {"AB11F_82", "TTCAGATGGA"},
        {"AB11R_82", "CGGAGGTAGA"},
        {"AC11F_83", "TAGCATCTGT"},
        {"AC11R_83", "GAGTTGACAA"},
        {"AD11F_84", "GGACGAGATC"},
        {"AD11R_84", "GCCGAACTTG"},
        {"AE11F_85", "AGGTTCTGTT"},
        {"AE11R_85", "AGGCCTCACA"},
        {"AF11F_86", "CATACTCGTG"},
        {"AF11R_86", "TCTCTGTTAG"},
        {"AG11F_87", "CCGGATACCA"},
        {"AG11R_87", "TCCGACGATT"},
        {"AH11F_88", "ATGTCCACCG"},
        {"AH11R_88", "AGGCTATGTT"},
        {"AA12F_89", "CACCAAGTGG"},
        {"AA12R_89", "CGTTCTCTTG"},
        {"AB12F_90", "TTGAGTACAC"},
        {"AB12R_90", "TTGTCTATGG"},
        {"AC12F_91", "CGGTTCCGTA"},
        {"AC12R_91", "GATGGATACA"},
        {"AD12F_92", "GGAGGTCCTA"},
        {"AD12R_92", "CACTTAGGCG"},
        {"AE12F_93", "CCTGCTTGGA"},
        {"AE12R_93", "ACACTGGCTA"},
        {"AF12F_94", "TTCACGTCAG"},
        {"AF12R_94", "ATCGCCACTG"},
        {"AG12F_95", "AACATAGCCT"},
        {"AG12R_95", "CTGACGTGAA"},
        {"AH12F_96", "TGACATAGTC"},
        {"AH12R_96", "TCAATCGTCT"},
};

}  // namespace

const std::unordered_map<std::string, KitInfo>& get_kit_infos() { return kit_info_map; }

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

std::string barcode_kits_list_str() {
    std::vector<std::string> kit_names;
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
    for (const auto& c : barcode_name) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            digits += c;
        }
    }

    return "barcode" + digits;
}

std::string generate_standard_barcode_name(const std::string& kit_name,
                                           const std::string& barcode_name) {
    return kit_name + "_" + normalize_barcode_name(barcode_name);
}

}  // namespace dorado::barcode_kits
