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

        {"dna_r10.4.1_e8.2_fast@v3.5.1", "/shared/static/d4wnbro47x1kbyhunhqu5x1lguq6yczu.zip"},
        {"dna_r10.4.1_e8.2_hac@v3.5.1", "/shared/static/9wo87gztgmz38mmeikwyfy05yfax4axr.zip"},
        {"dna_r10.4.1_e8.2_sup@v3.5.1", "/shared/static/ny4684yq0194t2mrda21x0v26ywkiog1.zip"},

};

}  // namespace basecaller
