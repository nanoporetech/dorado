#include <map>
#include <string>
#include <vector>

namespace basecaller {

  static const std::string URL_ROOT = "https://nanoporetech.box.com";

  static const std::map<std::string, std::string> models = {

    {"dna_r9.4.1_e8_fast@v3.4", "/shared/static/vrfk86xjtot5jndy8175w4mus1zcojy4.zip"},
    {"dna_r9.4.1_e8_hac@v3.3", "/shared/static/g6rbgd12xfunw5plgec3zlyy35692vy3.zip"},
    {"dna_r9.4.1_e8_sup@v3.3", "/shared/static/ezdt941r757a4tldam46l6sjcpgjflj2.zip"},

    {"dna_r9.4.1_e8.1_fast@v3.4","buvtwoh7wg73yext2wphq5mkqkqltgzz.zip"},
    {"dna_r9.4.1_e8.1_hac@v3.3", "/shared/static/i2rjjq0t3tlaktkipjjl8ef14p29eiss.zip"},
    {"dna_r9.4.1_e8.1_sup@v3.3", "/shared/static/xmpfpcq9eplsr2yoxha9pzmpurcerfey.zip"},
    
    {"dna_r10.4_e8.1_fast@v3.4", "/shared/static/fu350v2prt1qa164zei312lwa5hjb00v.zip"},
    {"dna_r10.4_e8.1_hac@v3.4", "/shared/static/uc71v2bsfpu69t5mnqgxyi7f4kmjax5z.zip"},
    {"dna_r10.4_e8.1_sup@v3.4", "/shared/static/7o8916vxeqzu6y1l6jmerwm3cnf12g2b.zip"},

  };

}
