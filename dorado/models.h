#include <map>
#include <string>
#include <vector>

namespace basecaller {
// Root URL for Models
static const std::string URL_ROOT = "https://nanoporetech.box.com";

// Serialised, released models
static const std::map<std::string, std::string> models = {

        {"dna_r9.4.1_e8_fast@v3.4", "/shared/static/vrfk86xjtot5jndy8175w4mus1zcojy4.zip"},
        {"dna_r9.4.1_e8_hac@v3.3", "/shared/static/g6rbgd12xfunw5plgec3zlyy35692vy3.zip"},
        {"dna_r9.4.1_e8_sup@v3.3", "/shared/static/ezdt941r757a4tldam46l6sjcpgjflj2.zip"},

        {"dna_r10.4.1_e8.2_260_bps_fast@v3.5.2",
         "/shared/static/uegadbtxw3j76ommk092a9711tk6unli.zip"},
        {"dna_r10.4.1_e8.2_260_bps_hac@v3.5.2",
         "/shared/static/xwvlfsc2oygtjozy837sie59xzp0gy7m.zip"},
        {"dna_r10.4.1_e8.2_260_bps_sup@v3.5.2",
         "/shared/static/dcpusbhqx7mdc3gb6ei6yg3zpyeiktgj.zip"},

        {"dna_r10.4.1_e8.2_400_bps_fast@v3.5.2",
         "/shared/static/80i0n7bcvik9sibnubv57tv3a8mhgnts.zip"},
        {"dna_r10.4.1_e8.2_400_bps_hac@v3.5.2",
         "/shared/static/ynenu1czyh853lw1xu3z3fz0yrdtw7yk.zip"},
        {"dna_r10.4.1_e8.2_400_bps_sup@v3.5.2",
         "/shared/static/2qnx2lsc0bhfjgwydgvon8zxv9c9bu9j.zip"},

};

}  // namespace basecaller
