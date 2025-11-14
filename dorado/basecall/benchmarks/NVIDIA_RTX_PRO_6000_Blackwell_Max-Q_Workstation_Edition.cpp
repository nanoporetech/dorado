#include "NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition.h"

namespace dorado::basecall {

void AddNVIDIA_RTX_PRO_6000_Blackwell_Max_Q_Workstation_EditionBenchmarks(
        std::map<std::pair<std::string, std::string>, std::unordered_map<int, float>>&
                chunk_benchmarks) {
    // Note that FAST and SUP benchmarks have been removed from the auto-generated list as they
    //  do not aid performance.  See INSTX-12044 for details.
    chunk_benchmarks[{"NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                      "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"}] = {
            {64, 0.384381f},    {128, 0.208279f},   {192, 0.141983f},   {256, 0.108148f},
            {320, 0.0877168f},  {384, 0.0744454f},  {448, 0.0644674f},  {512, 0.0579995f},
            {640, 0.0575828f},  {704, 0.05245f},    {768, 0.0491871f},  {832, 0.0479058f},
            {896, 0.0467756f},  {960, 0.0448113f},  {1728, 0.0436231f}, {1984, 0.0432826f},
            {2432, 0.0374674f}, {2560, 0.0356709f}, {2688, 0.0343029f}, {2816, 0.0331251f},
            {2944, 0.0327057f}, {3072, 0.0315901f}, {3200, 0.031502f},  {3328, 0.0312252f},
            {3456, 0.0305812f}, {3584, 0.0296734f}, {3840, 0.0290661f}, {7680, 0.0287717f},
            {7936, 0.0281911f},
    };
    chunk_benchmarks[{"NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                      "rna004_130bps_hac@v5.1.0"}] = {
            {64, 0.333429f},    {128, 0.182033f},   {192, 0.124248f},   {256, 0.0947753f},
            {320, 0.0777192f},  {384, 0.0679068f},  {448, 0.0582066f},  {512, 0.0529045f},
            {640, 0.0512222f},  {704, 0.0489093f},  {768, 0.0468186f},  {832, 0.0463359f},
            {896, 0.0450458f},  {960, 0.0441323f},  {1664, 0.0432798f}, {1728, 0.042914f},
            {1792, 0.0422648f}, {1856, 0.0418981f}, {1920, 0.0414412f}, {1984, 0.0409464f},
            {2432, 0.0360777f}, {2560, 0.0341107f}, {2688, 0.0331613f}, {2816, 0.0315154f},
            {3072, 0.0308841f}, {3328, 0.0306345f}, {3456, 0.0301707f}, {3584, 0.0291636f},
            {3840, 0.0287181f}, {7168, 0.0284137f}, {7680, 0.0278615f},
    };
}

}  // namespace dorado::basecall
