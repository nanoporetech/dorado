#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace dorado {
namespace urls {

static const std::string URL_ROOT = "https://nanoporetech.box.com";

// Serialised, released models
namespace simplex {

static const std::map<std::string, std::string> models = {

        // v3.{3,4}
        {"dna_r9.4.1_e8_fast@v3.4", "/shared/static/vrfk86xjtot5jndy8175w4mus1zcojy4.zip"},
        {"dna_r9.4.1_e8_hac@v3.3", "/shared/static/g6rbgd12xfunw5plgec3zlyy35692vy3.zip"},
        {"dna_r9.4.1_e8_sup@v3.3", "/shared/static/ezdt941r757a4tldam46l6sjcpgjflj2.zip"},

        // v3.5.2
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

        // v4.0.0
        {"dna_r10.4.1_e8.2_260bps_fast@v4.0.0",
         "/shared/static/ky2mrrmdwep5qgi8ht6b5hfvy5b2ojy9.zip"},
        {"dna_r10.4.1_e8.2_260bps_hac@v4.0.0",
         "/shared/static/nhxea41t11dg30hc70fg18bafj0yju8q.zip"},
        {"dna_r10.4.1_e8.2_260bps_sup@v4.0.0",
         "/shared/static/exzd2ezi6q708ynltkm6ckv361xgmam4.zip"},

        {"dna_r10.4.1_e8.2_400bps_fast@v4.0.0",
         "/shared/static/6xmmoltxeo8budtsxlak4qi0130m3opx.zip"},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.0.0",
         "/shared/static/2ed0ab3r6b8tptjq582f98r17nyojvpk.zip"},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.0.0",
         "/shared/static/i1f2unj1s6dplzpebl2bx4ytk8iqj0de.zip"},

        // RNA003
        {"rna003_120bps_sup@v3", "/shared/static/na0q0nooudrpe5yxgzkn58zlxfmlyuyo.zip"},
};

}  // namespace simplex

namespace stereo {

static const std::map<std::string, std::string> models = {
        {"dna_r10.4.2_e8.2_4khz_stereo@v1.0",
         "/shared/static/45zmew48ktyq6zbw05hq3t40qyrhuuya.zip"},
};

}

namespace modified {

static const std::vector<std::string> mods = {
        "5mCG",
        "5mCG_5hmCG",
};

static const std::map<std::string, std::string> models = {

        // v3.{3,4}
        {"dna_r9.4.1_e8_fast@v3.4_5mCG@v0", "/shared/static/340iw6qus57wfvqseeqbcn6k7m4ifmfl.zip"},
        {"dna_r9.4.1_e8_hac@v3.4_5mCG@v0", "/shared/static/umgvspqcpwb3lv0ps956amjnvnko8ly6.zip"},
        {"dna_r9.4.1_e8_sup@v3.4_5mCG@v0", "/shared/static/dmgvhghorlpa12y4s782x1fx0fqdqgxl.zip"},

        // v3.5.2
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

        // v4.0.0
        {"dna_r10.4.1_e8.2_400bps_fast@v4.0.0_5mCG_5hmCG@v2",
         "/shared/static/3hhvl90kfkdwch6uuglsa0ggk78nmcqa.zip"},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.0.0_5mCG_5hmCG@v2",
         "/shared/static/kijrf268zwurcpotq1v4sjj757ftc51l.zip"},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.0.0_5mCG_5hmCG@v2",
         "/shared/static/1pv196twy5m3ob0gfyufvjm1zogzh8vu.zip"},

};

}  // namespace modified
}  // namespace urls

namespace utils {

bool is_rna_model(const std::filesystem::path& model);
bool is_valid_model(const std::string& selected_model);
void download_models(const std::string& target_directory, const std::string& selected_model);

// finds the matching modification model for a given modification i.e. 5mCG and a simplex model
// is the matching modification model is not found in the same model directory as the simplex
// model then it is downloaded.
std::string get_modification_model(const std::string& simplex_model,
                                   const std::string& modification);

}  // namespace utils

}  // namespace dorado
