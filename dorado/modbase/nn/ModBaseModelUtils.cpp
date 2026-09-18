#include "ModBaseModelUtils.h"

namespace dorado::modbase::model {

std::vector<std::string> load_modbase_conv_lstm_weights(bool has_upsample) {
    std::vector<std::string> tensors{
            "sig_conv1.weight.tensor",   "sig_conv1.bias.tensor",
            "sig_conv2.weight.tensor",   "sig_conv2.bias.tensor",
            "sig_conv3.weight.tensor",   "sig_conv3.bias.tensor",

            "seq_conv1.weight.tensor",   "seq_conv1.bias.tensor",
            "seq_conv2.weight.tensor",   "seq_conv2.bias.tensor",

            "merge_conv1.weight.tensor", "merge_conv1.bias.tensor",

            "lstm1.weight_ih_l0.tensor", "lstm1.weight_hh_l0.tensor",
            "lstm1.bias_ih_l0.tensor",   "lstm1.bias_hh_l0.tensor",

            "lstm2.weight_ih_l0.tensor", "lstm2.weight_hh_l0.tensor",
            "lstm2.bias_ih_l0.tensor",   "lstm2.bias_hh_l0.tensor",

            "fc.weight.tensor",          "fc.bias.tensor",
    };

    if (has_upsample) {
        tensors.push_back("linear_up.linear.weight.tensor");
        tensors.push_back("linear_up.linear.bias.tensor");
    }
    return tensors;
}

}  // namespace dorado::modbase::model
