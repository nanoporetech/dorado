#include <string>

namespace dorado::utils {

struct DefaultParameters {
#ifdef __APPLE__
    std::string device{"metal"};
#else
    std::string device{"cuda:all"};
#endif
    int batchsize{0};
    int chunksize{10000};
    int overlap{500};
    int num_runners{2};
#ifdef DORADO_TX2
    int remora_batchsize{128};
#else
    int remora_batchsize{1024};
#endif
    int remora_threads{2};
};

static const DefaultParameters default_parameters{};

};  // namespace dorado::utils
