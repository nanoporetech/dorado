#include <map>
#include <string>
#include <vector>

namespace dorado::urls {

static const std::string URL_ROOT = "https://nanoporetech.box.com";

// Serialised, released models
namespace simplex {

static const std::map<std::string, std::string> models = {

        {"dna_r9.4.1_e8_fast@v3.4", "/shared/static/vrfk86xjtot5jndy8175w4mus1zcojy4.zip"},
        {"dna_r9.4.1_e8_hac@v3.3", "/shared/static/g6rbgd12xfunw5plgec3zlyy35692vy3.zip"},
        {"dna_r9.4.1_e8_sup@v3.3", "/shared/static/ezdt941r757a4tldam46l6sjcpgjflj2.zip"},

        {"dna_r10.4.1_e8.2_260bps_fast@v3.5.2",
         "/shared/static/uegadbtxw3j76ommk092a9711tk6unli.zip"},
        {"dna_r10.4.1_e8.2_260bps_hac@v3.5.2",
         "/shared/static/xwvlfsc2oygtjozy837sie59xzp0gy7m.zip"},
        {"dna_r10.4.1_e8.2_260bps_sup@v3.5.2",
         "/shared/static/dcpusbhqx7mdc3gb6ei6yg3zpyeiktgj.zip"},

        {"dna_r10.4.1_e8.2_400bps_fast@v3.5.2",
         "/shared/static/80i0n7bcvik9sibnubv57tv3a8mhgnts.zip"},
        {"dna_r10.4.1_e8.2_400bps_hac@v3.5.2",
         "/shared/static/ynenu1czyh853lw1xu3z3fz0yrdtw7yk.zip"},
        {"dna_r10.4.1_e8.2_400bps_sup@v3.5.2",
         "/shared/static/2qnx2lsc0bhfjgwydgvon8zxv9c9bu9j.zip"},

};

}  // namespace simplex

namespace modified {

static const std::map<std::string, std::string> models = {

        {"dna_r9.4.1_e8_fast@v3.4_5mCG@v0", "/shared/static/340iw6qus57wfvqseeqbcn6k7m4ifmfl.zip"},
        {"dna_r9.4.1_e8_hac@v3.4_5mCG@v0", "/shared/static/umgvspqcpwb3lv0ps956amjnvnko8ly6.zip"},
        {"dna_r9.4.1_e8_sup@v3.4_5mCG@v0", "/shared/static/dmgvhghorlpa12y4s782x1fx0fqdqgxl.zip"},

        {"dna_r10.4.1_e8.2_260bps_fast@v3.5.2_5mCG@v2",
         "/shared/static/4fj87s3b03eiewmz3uzeoa29yecr5l80.zip"},
        {"dna_r10.4.1_e8.2_260bps_hac@v3.5.2_5mCG@v2",
         "/shared/static/1t2j1dpu9aa1u8onbn3qz1mixtwyf5e8.zip"},
        {"dna_r10.4.1_e8.2_260bps_sup@v3.5.2_5mCG@v2",
         "/shared/static/7qmmo3iv2r0ivyt72rotklvy8ws8n2eo.zip"},

        {"dna_r10.4.1_e8.2_400bps_fast@v3.5.2_5mCG@v2",
         "/shared/static/nyz4vwdo87a9640qz4pbr8r1f4lf8r04.zip"},
        {"dna_r10.4.1_e8.2_400bps_hac@v3.5.2_5mCG@v2",
         "/shared/static/scqhs3le7dylxadtcinrc0cmh7brxbsf.zip"},
        {"dna_r10.4.1_e8.2_400bps_sup@v3.5.2_5mCG@v2",
         "/shared/static/8emosrp9i3vfqg3e4ehsifnecr9dpiot.zip"},

};

}  // namespace modified

}  // namespace dorado::urls
