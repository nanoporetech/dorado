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
         {
                 "VMK4",
                 false,
                 false,
                 RBK4_FRONT,
                 RBK4_REAR,
                 "",
                 "",
                 {
                         "BC01",
                         "BC02",
                         "BC03",
                         "BC04",
                         "BC05",
                         "BC06",
                         "BC07",
                         "BC08",
                         "BC09",
                         "BC10",
                 },
                 {},
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
