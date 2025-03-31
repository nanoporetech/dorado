#!/bin/bash

set -ex

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dorado executable> [device_string]"
    exit 1
fi

device_string=${2:-"auto"}
echo "Using device string -x $device_string"
data_dir=$(dirname $0)/../tests/data
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)
pod5_dir=${data_dir}/pod5/dna_r10.4.1_e8.2_400bps_5khz/

echo deleting any previous benchmark data
rm chunk_benchmarks* || true

echo Running benchmarks for models of interest
for model_name in \
                  dna_r9.4.1_e8_fast@v3.4 \
                  dna_r9.4.1_e8_hac@v3.3 \
                  dna_r9.4.1_e8_sup@v3.3 \
                  dna_r10.4.1_e8.2_260bps_fast@v4.1.0 \
                  dna_r10.4.1_e8.2_260bps_hac@v4.1.0 \
                  dna_r10.4.1_e8.2_260bps_sup@v4.1.0 \
                  dna_r10.4.1_e8.2_400bps_fast@v4.1.0 \
                  dna_r10.4.1_e8.2_400bps_hac@v4.1.0 \
                  dna_r10.4.1_e8.2_400bps_sup@v4.1.0 \
                  dna_r10.4.1_e8.2_400bps_fast@v4.3.0 \
                  dna_r10.4.1_e8.2_400bps_hac@v4.3.0 \
                  dna_r10.4.1_e8.2_400bps_sup@v4.3.0 \
                  dna_r10.4.1_e8.2_400bps_fast@v5.0.0 \
                  dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
                  dna_r10.4.1_e8.2_400bps_sup@v5.0.0 \
                  rna004_130bps_fast@v5.1.0 \
                  rna004_130bps_hac@v5.1.0 \
                  rna004_130bps_sup@v5.1.0 \
                  ; do
    echo $model_name;
    $dorado_bin download --model $model_name
    $dorado_bin basecaller -x $device_string --skip-model-compatibility-check --emit-batchsize-benchmarks $model_name $pod5_dir > /dev/null
done

# Extract the GPU name from the benchmark filenames
benchmark_files=(chunk_benchmarks__*)
gpu_name="${benchmark_files[0]}"
# strip off beginning "chunk_benchmarks__"
gpu_name=${gpu_name#*__}
# strip off the __ sections at the end (model name)
gpu_name=${gpu_name%%__*}
# Replace spaces with underscores
gpu_name="${gpu_name// /_}"
gpu_name_no_dashes="${gpu_name//-/_}"
echo gpu name is: $gpu_name

# Delete any existing cpp source files
rm ${gpu_name}.cpp || true
rm ${gpu_name}.h || true

echo "#include \"${gpu_name}.h\"

namespace dorado::basecall {

void Add${gpu_name_no_dashes}Benchmarks(std::map<std::pair<std::string, std::string>, std::unordered_map<int, float>>& chunk_benchmarks) {" >> ${gpu_name}.cpp

# Add the chunk benchmarks for every model
cat chunk_benchmarks__*.txt >> ${gpu_name}.cpp

echo "}

} // namespace dorado::basecall
" >> ${gpu_name}.cpp

gpu_name=${gpu_name%%_cuda*}

echo "#pragma once

#include <map>
#include <string>
#include <unordered_map>

namespace dorado::basecall {
    void Add${gpu_name_no_dashes}Benchmarks(std::map<std::pair<std::string, std::string>, std::unordered_map<int, float>>& chunk_benchmarks);
} // namespace dorado::basecall
" >> ${gpu_name}.h