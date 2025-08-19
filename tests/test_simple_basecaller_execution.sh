#!/bin/bash

# Test expected log output from the dorado binary execution.

set -ex
set -o pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dorado executable> [5k model] [batch size] [5k v43 model] [rna004 model] [model speed] [model version]"
    exit 1
fi

test_dir=$(dirname $0)
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)
model_name_5k=${2:-dna_r10.4.1_e8.2_400bps_hac@v5.0.0}
batch=${3:-384}
model_name_5k_v43=${4:-dna_r10.4.1_e8.2_400bps_hac@v4.3.0}
model_name_rna004=${5:-rna004_130bps_hac@v3.0.1}

model_speed=${6:-"hac"}
version=${7:-"v5.0.0"}
model_complex="${model_speed}@${version}"

data_dir=${test_dir}/data
output_dir_name=test_simple_basecaller_output_${RANDOM}
output_dir=${test_dir}/${output_dir_name}
mkdir -p ${output_dir}

models_directory=${output_dir}/models
mkdir -p ${models_directory}

models_directory_arg="--models-directory ${models_directory}"

ONT_OUTPUT_SPEC_REF="c85d5f094a5b4a73b98b939d0ab3bfd014add372" # currently INSTX-10694_hts_style_fastq branch
SPECIFICATION_URL="${ONT_OUTPUT_SPEC_REPO}-/archive/${ONT_OUTPUT_SPEC_REF}/ont-output-specification-${ONT_OUTPUT_SPEC_REF}.zip"
SPECIFICATION_FILE="ont_output_spec.zip"
VALIDATOR_COMMIT="3d7cf74ef1c45328f8cc545d82c3b1b43757b1fb"

# Set up the output specification validator so we can check output file formats
if [[ "${VALIDATE_FASTQ}" -eq "1" ]]; then
    echo "Enabling validation of fastq files against spec from ${SPECIFICATION_URL}"
    $PYTHON --version
    $PYTHON -m venv venv
    source ./venv/*/activate
    # Install output-file specification validator.
    rm -rf ont-output-specification-validator
    git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@${VALIDATOR_REPO}
    pushd ont-output-specification-validator
    git checkout ${VALIDATOR_COMMIT}
    popd
    pip install -e ont-output-specification-validator
    curl -LfsS ${SPECIFICATION_URL} > ${SPECIFICATION_FILE}
fi

echo dorado download models
$dorado_bin download --list
$dorado_bin download --list-structured
$dorado_bin download --model ${model_name_5k} ${models_directory_arg}
model_5k=${models_directory}/${model_name_5k}
$dorado_bin download --model ${model_name_5k_v43} ${models_directory_arg}
model_5k_v43=${models_directory}/${model_name_5k_v43}
$dorado_bin download --model ${model_name_rna004} ${models_directory_arg}
model_rna004=${models_directory}/${model_name_rna004}

dorado_check_bam_not_empty() {
    if [[ -n "$SAMTOOLS_UNAVAILABLE" ]]; then
        echo "Skipping dorado_check_bam_not_empty as SAMTOOLS_UNAVAILABLE is set"
        return 0
    fi
    samtools quickcheck -u $output_dir/calls.bam
    samtools view -h $output_dir/calls.bam > $output_dir/calls.sam
    num_lines=$(wc -l $output_dir/calls.sam | awk '{print $1}')
    if [[ ${num_lines} -eq "0" ]]; then
        echo "Error: empty bam file"
        exit 1
    fi
}

pod5_data=$data_dir/pod5/dna_r10.4.1_e8.2_400bps_5khz
echo using pod5_data: $pod5_data

echo dorado basecaller test stage
# Not included models_directory_arg here to test temporary model download and delete.
$dorado_bin basecaller ${model_5k} $pod5_data -b ${batch} --emit-fastq > $output_dir/ref.fq
if [[ "${VALIDATE_FASTQ}" -eq "1" ]]; then
    $PYTHON ./${test_dir}/validate_fastq.py $output_dir/ref.fq $SPECIFICATION_FILE
fi
$dorado_bin basecaller ${model_5k} $pod5_data ${models_directory_arg} -b ${batch} --modified-bases 5mCG_5hmCG --emit-moves > $output_dir/calls.bam
dorado_check_bam_not_empty
$dorado_bin basecaller ${model_5k} $pod5_data/ ${models_directory_arg} -x cpu --modified-bases 5mCG_5hmCG -vv > $output_dir/calls.bam
dorado_check_bam_not_empty

# Test that we can run the deprecated 9.4.1 HAC models for Q V2.
pushd $models_directory
curl -LfsS https://cdn.oxfordnanoportal.com/software/analysis/dorado/dna_r9.4.1_e8_hac@v3.3.zip > dna_r9.4.1_e8_hac@v3.3.zip
unzip dna_r9.4.1_e8_hac@v3.3.zip
curl -LfsS https://cdn.oxfordnanoportal.com/software/analysis/dorado/dna_r9.4.1_e8_hac@v3.3_5mCG_5hmCG@v0.zip > dna_r9.4.1_e8_hac@v3.3_5mCG_5hmCG@v0.zip
unzip dna_r9.4.1_e8_hac@v3.3_5mCG_5hmCG@v0.zip
popd
$dorado_bin basecaller ${models_directory}/dna_r9.4.1_e8_hac@v3.3 $pod5_data/ --modified-bases-models ${models_directory}/dna_r9.4.1_e8_hac@v3.3_5mCG_5hmCG@v0 --skip-model-compatibility-check --enable-deprecated-models > $output_dir/calls.bam

$dorado_bin basecaller $model_complex,5mCG_5hmCG $pod5_data/ ${models_directory_arg} -b ${batch} --emit-moves > $output_dir/calls.bam

# Check that the read group has the required model info in its header
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    if ! grep -q "basecall_model=${model_name_5k}" $output_dir/calls.sam; then
        echo "Output SAM file does not contain basecall model name in header!"
        exit 1
    fi
    if ! grep -q "modbase_models=${model_name_5k}_5mCG_5hmCG" $output_dir/calls.sam; then
        echo "Output SAM file does not contain modbase model name in header!"
        exit 1
    fi
fi

echo dorado basecaller mixed model complex and --modified-bases
$dorado_bin basecaller $model_complex $pod5_data/ ${models_directory_arg} -b ${batch} --modified-bases 5mCG_5hmCG -vv > $output_dir/calls.bam
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    samtools view -h $output_dir/calls.bam | grep "ML:B:C,"
    samtools view -h $output_dir/calls.bam | grep "MM:Z:C+h"
    samtools view -h $output_dir/calls.bam | grep "MN:i:"
fi
set +e
if $dorado_bin basecaller ${model_5k} $pod5_data ${models_directory_arg} -b ${batch} --emit-fastq --reference $output_dir/ref.fq > $output_dir/error_condition.fq; then
    echo "Error: dorado basecaller should fail with combination of emit-fastq and reference!"
    exit 1
fi
if $dorado_bin basecaller ${model_5k} $pod5_data ${models_directory_arg} -b ${batch} --emit-fastq --modified-bases 5mCG_5hmCG > $output_dir/error_condition.fq; then
    echo "Error: dorado basecaller should fail with combination of emit-fastq and modbase!"
    exit 1
fi
if $dorado_bin basecaller $model_5k_v43 $data_dir/duplex/pod5 ${models_directory_arg} --modified-bases 5mC_5hmC 5mCG_5hmCG > $output_dir/error_condition.fq; then
    echo "Error: dorado basecaller should fail with multiple modbase configs having overlapping mods!"
    exit 1
fi
set -e

# Check INSTX-5275 problematic read does not crash
$dorado_bin basecaller $model_5k_v43 $data_dir/split/INSTX-5275 ${models_directory_arg} -b ${batch} --emit-fastq --dump_stats_file $output_dir/INSTX-5275_stats.txt > $output_dir/INSTX-5275.fq

# Check that dorado handles degenerate reads without crashing
$dorado_bin basecaller $model_5k_v43 $data_dir/pod5/degenerate/trimming_bomb.pod5 ${models_directory_arg} -b ${batch} --skip-model-compatibility-check > $output_dir/error_condition.fq
$dorado_bin basecaller $model_5k_v43 $data_dir/pod5/degenerate/overtrim.pod5 ${models_directory_arg} -b ${batch} --skip-model-compatibility-check --kit-name EXP-NBD196 > $output_dir/error_condition.fq

{
    set +e
    echo "Testing split read without '--disable-read-splitting'"
    $dorado_bin basecaller ${model_5k} ${data_dir}/single_split_read > $output_dir/calls.bam
    if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
        num_sam_records=$(samtools view $output_dir/calls.bam | wc -l | awk '{print $1}')
        if [[ $num_sam_records -ne "2" ]]; then
            echo "Expected 2 sam records from 1 split read but found: ${num_sam_records}"
            exit 1
        fi
        num_unique_parents=$(samtools view $output_dir/calls.bam | grep "pi:Z:[^\s]*" -oh | uniq | wc -l | awk '{print $1}')
        if [[ $num_unique_parents -ne "1" ]]; then
            echo "Expected 1 unique parent read id in 'pi:Z' tag but found: ${num_unique_parents}"
            exit 1
        fi
    fi

    echo "Testing split read with '--disable-read-splitting'"
    $dorado_bin basecaller ${model_5k} ${data_dir}/single_split_read --disable-read-splitting > $output_dir/calls.bam
    if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
        num_sam_records=$(samtools view $output_dir/calls.bam | wc -l | awk '{print $1}')
        if [[ $num_sam_records -ne "1" ]]; then
            echo "Expected 1 sam records from 1 unsplit read with '--disable-read-splitting' but found: ${num_sam_records}"
            exit 1
        fi
        num_parent_tags=$(samtools view $output_dir/calls.bam | grep "pi:Z:[^\s]*" -oh | wc -l | awk '{print $1}')
        if [[ $num_parent_tags -ne "0" ]]; then
            echo "Expected 0 instances of 'pi:Z' tag with '--disable-read-splitting' but found: ${num_parent_tags}"
            exit 1
        fi
    fi
    set -e
}

${test_dir}/test_nested_output_structure.sh ${dorado_bin} ${model_speed} ${batch}

echo dorado summary test stage
$dorado_bin summary $output_dir/calls.bam

echo redirecting stderr to stdout: check output is still valid
# The debug layer prints to stderr to say that it's enabled, so disable it for this test.
env -u MTL_DEBUG_LAYER $dorado_bin basecaller ${model_5k} $pod5_data/ -b ${batch} --modified-bases 5mCG_5hmCG --emit-moves > $output_dir/calls.bam 2>&1
dorado_check_bam_not_empty

echo dorado aligner test stage

# Make a sam file to use as input
$dorado_bin basecaller ${model_5k} $pod5_data/ -b ${batch} --modified-bases 5mCG_5hmCG --emit-moves --emit-sam > $output_dir/calls.sam

$dorado_bin aligner $output_dir/ref.fq $output_dir/calls.sam > $output_dir/calls.bam
dorado_check_bam_not_empty
mkdir $output_dir/folder
mkdir $output_dir/folder/subfolder
cp $output_dir/calls.sam $output_dir/folder/calls.sam
cp $output_dir/calls.sam $output_dir/folder/subfolder/calls.sam
$dorado_bin aligner $output_dir/ref.fq $output_dir/folder -o $output_dir/aligner_out
dorado_check_bam_not_empty
$dorado_bin basecaller ${model_5k} $pod5_data/ ${models_directory_arg} -b ${batch} --modified-bases 5mCG_5hmCG | $dorado_bin aligner $output_dir/ref.fq > $output_dir/calls.bam
dorado_check_bam_not_empty
$dorado_bin basecaller ${model_5k} $pod5_data/ ${models_directory_arg} -b ${batch} --modified-bases 5mCG_5hmCG --reference $output_dir/ref.fq > $output_dir/calls.bam
dorado_check_bam_not_empty
# Check that the aligner strips old alignment tags
$dorado_bin aligner $data_dir/aligner_test/5mers_rand_ref.fa $data_dir/aligner_test/prealigned.sam > $output_dir/realigned.bam
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    num_nm_tags=$(samtools view $output_dir/realigned.bam | grep -o NM:i | wc -l)
    # This alignment creates a secondary output, so there should be exactly 2 NM:i tags
    if [[ $num_nm_tags -ne "2" ]]; then
        echo "dorado aligner has emitted incorrect number of NM tags."
        exit 1
    fi
fi

echo dorado aligner options test stage
dorado_aligner_options_test() (
    set +e
    set +x

    MM2=$(dirname $dorado_bin)/minimap2
    echo -n "minimap2 version: "; $MM2 --version

    # list of options and whether they affect the output
    REF=$data_dir/aligner_test/lambda_ecoli.fasta
    RDS=$data_dir/aligner_test/dataset.fastq

    RETURN=true
    touch err
    ERROR() { echo $*; RETURN=false; cat err; }
    SKIP() { echo $*; cat err; }

    MM2_OPTIONS=(""    "-k 20" "-w 100" "-I 100K" "--secondary no" "-N 1" "-r 10,100" "-Y" "--secondary-seq" "--print-aln-seq")
    DOR_OPTIONS=(""    "-k 20" "-w 100" "-I 100K" "--secondary no" "-N 1" "-r 10,100" "-Y" "--secondary-seq" "--print-aln-seq")
    CHANGES=(false true    true     false     true             true   true        true true              false            )
    for ((i = 0; i < ${#MM_OPTIONS[@]}; i++)); do
        mm2_opt=${MM2_OPTIONS[$i]}
        dor_opt=${DOR_OPTIONS[$i]}
        echo -n "$i: with mm2 option '$mm2_opt' and dorado option '$dor_opt' ... "

        # run dorado aligner
        if ! $dorado_bin aligner $dor_opt $REF $RDS 2>err | samtools view -h 2>>err > $output_dir/dorado-$i.sam; then
            ERROR failed running dorado aligner
            continue
        fi

        # check output integrity
        if ! samtools quickcheck -u $output_dir/dorado-$i.sam 2>err; then
            ERROR failed sam check
            continue
        fi

        # sort and cut output for comparison
        filter_header="grep -ve ^@PG -e ^@HD"
        sort $output_dir/dorado-$i.sam | $filter_header | cut -f-11 > $output_dir/dorado-$i.ssam

        # compare with minimap2 output
        if $MM2 -a $mm2_opt $REF $RDS 2>err > $output_dir/minimap2-$i.sam; then
            sort $output_dir/minimap2-$i.sam | $filter_header | cut -f-11 > $output_dir/minimap2-$i.ssam
            if ! diff $output_dir/dorado-$i.ssam $output_dir/minimap2-$i.ssam > err; then
                ERROR failed comparison with minimap2 output
                continue
            fi
        else
            SKIP skipped
            continue
        fi

        # check output changed
        should_change=${CHANGES[$i]}
        if diff $output_dir/dorado-$i.ssam $output_dir/dorado-0.ssam > err; then
            does_change=false
        else
            does_change=true
        fi
        if [[ $should_change != $does_change ]]; then
            $should_change && ERROR failed to change output || ERROR failed to preserve output
            continue
        fi

        echo success
    done
    $RETURN
)
dorado_aligner_options_test

# Skip duplex tests if NO_TEST_DUPLEX is set.
if [[ "${NO_TEST_DUPLEX}" -ne "1" ]]; then
    echo dorado duplex basespace test stage
    $dorado_bin duplex basespace $data_dir/basespace/pairs.bam ${models_directory_arg} --threads 1 --pairs $data_dir/basespace/pairs.txt > $output_dir/calls.bam

    echo dorado in-line duplex test stage - model name
    $dorado_bin duplex $model_5k $data_dir/duplex/pod5 ${models_directory_arg} > $output_dir/duplex_calls.bam
    if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
        samtools quickcheck -u $output_dir/duplex_calls.bam
        num_duplex_reads=$(samtools view $output_dir/duplex_calls.bam | grep dx:i:1 | wc -l | awk '{print $1}')
        if [[ $num_duplex_reads -ne "2" ]]; then
            echo "Duplex basecalling missing reads - in-line"
            exit 1
        fi
    fi

    echo dorado in-line duplex test stage - complex
    $dorado_bin duplex ${model_complex} $data_dir/duplex/pod5 ${models_directory_arg} > $output_dir/duplex_calls.bam
    if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
        samtools quickcheck -u $output_dir/duplex_calls.bam
        num_duplex_reads=$(samtools view $output_dir/duplex_calls.bam | grep dx:i:1 | wc -l | awk '{print $1}')
        if [[ $num_duplex_reads -ne "2" ]]; then
            echo "Duplex basecalling missing reads."
            exit 1
        fi
    fi

    echo dorado pairs file based duplex test stage - model name
    $dorado_bin duplex $model_5k $data_dir/duplex/pod5 ${models_directory_arg} --pairs $data_dir/duplex/pairs.txt > $output_dir/duplex_calls.bam
    if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
        samtools quickcheck -u $output_dir/duplex_calls.bam
        num_duplex_reads=$(samtools view $output_dir/duplex_calls.bam | grep dx:i:1 | wc -l | awk '{print $1}')
        if [[ $num_duplex_reads -ne "2" ]]; then
            echo "Duplex basecalling missing reads - pairs file"
            exit 1
        fi
    fi

    echo dorado pairs file based duplex test stage - complex
    $dorado_bin duplex ${model_complex} $data_dir/duplex/pod5 ${models_directory_arg} --pairs $data_dir/duplex/pairs.txt > $output_dir/duplex_calls.bam
    if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
        samtools quickcheck -u $output_dir/duplex_calls.bam
        num_duplex_reads=$(samtools view $output_dir/duplex_calls.bam | grep dx:i:1 | wc -l | awk '{print $1}')
        if [[ $num_duplex_reads -ne "2" ]]; then
            echo "Duplex basecalling missing reads."
            exit 1
        fi
    fi

    echo dorado in-line modbase duplex from model complex
    $dorado_bin duplex ${model_complex},5mCG_5hmCG $data_dir/duplex/pod5 ${models_directory_arg} > $output_dir/duplex_calls_mods.bam
    if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
        samtools quickcheck -u $output_dir/duplex_calls_mods.bam
        num_duplex_reads=$(samtools view $output_dir/duplex_calls_mods.bam | grep dx:i:1 | wc -l | awk '{print $1}')
        if [[ $num_duplex_reads -ne "2" ]]; then
            echo "Duplex basecalling missing reads - mods"
            exit 1
        fi
    fi
fi

if command -v truncate > /dev/null; then
    echo dorado basecaller resume feature
    # n.b. some of these options (--skip, --mm2-opts) won't affect the basecall but are included to test that we can resume with them present
    $dorado_bin basecaller ${models_directory_arg} -b ${batch} ${model_5k} $data_dir/multi_read_pod5 --mm2-opts "-k 15 -w 10" --skip-model-compatibility-check > $output_dir/tmp.bam
    truncate -s 20K $output_dir/tmp.bam
    $dorado_bin basecaller ${models_directory_arg} -b ${batch} ${model_5k} $data_dir/multi_read_pod5 --mm2-opts "-k 15 -w 10" --skip-model-compatibility-check --resume-from $output_dir/tmp.bam > $output_dir/calls.bam
    if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
        samtools quickcheck -u $output_dir/calls.bam
        num_reads=$(samtools view -c $output_dir/calls.bam)
        if [[ $num_reads -ne "4" ]]; then
            echo "Resumed basecalling has incorrect number of reads."
            exit 1
        fi
    fi
fi

echo dorado aligner output directory test stage
$dorado_bin aligner $data_dir/aligner_test/basecall_target.fa $data_dir/aligner_test/basecall.sam --output-dir $output_dir/aligned --emit-summary
num_summary_lines=$(wc -l < $output_dir/aligned/alignment_summary.txt)
if [[ $num_summary_lines -ne "2" ]]; then
    echo "2 lines in summary expected. Found ${num_summary_lines}"
    exit 1
fi

# The @RG header tests can only run if Samtools is available.
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    check_add_fastq_rg_header_count() {
        local expected_count=$1
        local bam_file=$2
        local description=$3

        local count
        count=$(samtools view -h "$bam_file" \
            | grep "@RG" \
            | grep "ID:4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0" \
            | grep "PU:PAM93185" \
            | grep "DT:2022-10-18T10:18:07.247000+00:00" \
            | grep "DS:runid=4524e8b9-b90e-4ffb-a13a-380266513b64 basecall_model=dna_r10.4.1_e8.2_400bps_hac@v5.0.0" \
            | grep "LB:PCR_zymo" \
            | wc -l \
            | awk '{ print $1 }' || true)

        if [[ "$count" -ne "$expected_count" ]]; then
            echo "Expected ${expected_count} @RG header line(s) ${description}. Found ${count}"
            exit 1
        fi
    }

    {
        test_command="$dorado_bin aligner $data_dir/aligner_test/basecall_target.fa $data_dir/aligner_test/example-hts.fastq.gz --output-dir $output_dir/aligned"

        ${test_command} --add-fastq-rg
        check_add_fastq_rg_header_count 1 $output_dir/aligned/example-hts.bam "with --add-fastq-rg"

        ${test_command}
        check_add_fastq_rg_header_count 0 $output_dir/aligned/example-hts.bam "without --add-fastq-rg"
    }
fi

echo dorado demux test stage
$dorado_bin demux $data_dir/barcode_demux/double_end_variant/EXP-PBC096_BC04.fastq --kit-name EXP-PBC096 --output-dir $output_dir/demux --emit-summary
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    samtools quickcheck -u $output_dir/demux/unknown_run_id_EXP-PBC096_barcode04.bam
    num_demuxed_reads=$(samtools view -c $output_dir/demux/unknown_run_id_EXP-PBC096_barcode04.bam)
    if [[ $num_demuxed_reads -ne "3" ]]; then
        echo "3 demuxed reads expected. Found ${num_demuxed_reads}"
        exit 1
    fi
fi

$dorado_bin demux $data_dir/barcode_demux/double_end_variant/ --kit-name EXP-PBC096 --output-dir $output_dir/demux_from_folder
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    samtools quickcheck -u $output_dir/demux_from_folder/unknown_run_id_EXP-PBC096_barcode04.bam
fi
num_summary_lines=$(wc -l < $output_dir/demux/barcoding_summary.txt)
if [[ $num_summary_lines -ne "4" ]]; then
    echo "4 lines in summary expected. Found ${num_summary_lines}"
    exit 1
fi

echo dorado custom demux test stage
$dorado_bin demux $data_dir/barcode_demux/double_end/SQK-RPB004_BC01.fastq --output-dir $output_dir/custom_demux --kit-name CUSTOM-SQK-RPB004 --barcode-arrangement $data_dir/barcode_demux/custom_barcodes/RPB004.toml --barcode-sequences $data_dir/barcode_demux/custom_barcodes/RPB004_sequences.fasta
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    samtools quickcheck -u $output_dir/custom_demux/unknown_run_id_CUSTOM-SQK-RPB004_barcode01.bam
    num_demuxed_reads=$(samtools view -c $output_dir/custom_demux/unknown_run_id_CUSTOM-SQK-RPB004_barcode01.bam)
    if [[ $num_demuxed_reads -ne "2" ]]; then
        echo "3 demuxed reads expected. Found ${num_demuxed_reads}"
        exit 1
    fi
fi

echo dorado demux doesnt crash on an empty input directory
rm -rf empty_dir
mkdir empty_dir
$dorado_bin demux empty_dir --kit-name EXP-PBC096 --output-dir $output_dir/empty_dir
if [[ $? -ne "0" ]]; then
    echo "dorado crashed when given an empty input directory"
    exit 1
fi

echo dorado trim test stage
file1=$data_dir/adapter_trim/lsk110_single_read.fastq
file2=$output_dir/lsk110_single_read_trimmed.fastq
$dorado_bin trim --sequencing-kit SQK-LSK114 --emit-fastq $file1 > $file2
if cmp --silent -- "$file1" "$file2"; then
    echo "Adapter was not trimmed. Input and output reads are identical."
    exit 1
fi

echo "dorado test poly(A) tail estimation"
$dorado_bin basecaller ${models_directory_arg} -b ${batch} ${model_5k} $data_dir/poly_a/r10_4_1_5khz_cdna_pod5/ --estimate-poly-a > $output_dir/cdna_polya.bam
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    samtools quickcheck -u $output_dir/cdna_polya.bam
    num_estimated_reads=$(samtools view $output_dir/cdna_polya.bam | grep pt:i: | wc -l | awk '{print $1}')
    if [[ $num_estimated_reads -ne "2" ]]; then
        echo "2 poly(A) estimated reads expected. Found ${num_estimated_reads}"
        exit 1
    fi
fi

$dorado_bin basecaller ${models_directory_arg} -b ${batch} ${model_5k} $data_dir/poly_a/r10_4_1_5khz_cdna_pod5/ --estimate-poly-a --poly-a-config $data_dir/poly_a/configs/polya.toml > $output_dir/no_detect_cdna_polya.bam
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    samtools quickcheck -u $output_dir/no_detect_cdna_polya.bam
    if [[ $? -ne "0" ]]; then
        echo "PolyA tail estimation with custom config file failed."
        exit 1
    fi
fi

$dorado_bin basecaller ${models_directory_arg} -b ${batch} ${model_rna004} $data_dir/poly_a/rna004_pod5/ --estimate-poly-a > $output_dir/rna_polya.bam
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    samtools quickcheck -u $output_dir/rna_polya.bam
    num_estimated_reads=$(samtools view $output_dir/rna_polya.bam | grep pt:i: | wc -l | awk '{print $1}')
    if [[ $num_estimated_reads -ne "1" ]]; then
        echo "1 poly(A) estimated reads expected. Found ${num_estimated_reads}"
        exit 1
    fi
fi

echo dorado basecaller barcoding read groups
test_barcoding_read_groups() (
    while (("$#" >= 2 )); do
        barcode=$1
        export expected_read_groups_${barcode}=$2
        shift 2
    done
    sample_sheet=$1
    output_name=read_group_test${sample_sheet:+_sample_sheet}
    demux_data=$data_dir/barcode_demux/read_group_test
    $dorado_bin basecaller ${models_directory_arg} -b ${batch} --kit-name SQK-RBK114-96 ${sample_sheet:+--sample-sheet ${sample_sheet}} ${model_5k} ${demux_data} --no-trim > $output_dir/${output_name}.bam

    samtools quickcheck -u $output_dir/${output_name}.bam

    split_dir=$output_dir/${output_name}
    mkdir $split_dir
    samtools split -u $split_dir/unknown.bam -f "$split_dir/rg_%!.bam" $output_dir/${output_name}.bam

    # There shouldn't be any unknown groups.
    num_read_groups=$(samtools view -c $split_dir/unknown.bam)
    if [[ $num_read_groups -ne "0" ]]; then
        echo "Reads with unknown read groups found."
        exit 1
    fi

    check_barcodes() (
        bam=$1
        echo "Checking file: $bam"
        if [[ $bam =~ "_SQK-RBK114-96_" ]]; then
            # Arrangement is |<kit>_<barcode>|, so trim the kit from the prefix and the .bam from the suffix.
            barcode=${bam#*_SQK-RBK114-96_}
            barcode=${barcode%.bam*}
        elif [[ $bam =~ "_${model_name_5k}_" ]]; then
            # Arrangement is |<barcode_alias>|, so trim the model from the prefix and the .bam from the suffix.
            barcode=${bam#*_${model_name_5k}_}
            barcode=${barcode%.bam*}
        elif [[ $bam =~ "/0d85015e-6a4e-400c-a80f-c187c65a6d03_" ]]; then
            # Demuxed file, so trim the run_id from the prefix and the .bam from the suffix.
            barcode=${bam#*0d85015e-6a4e-400c-a80f-c187c65a6d03_}
            barcode=${barcode%.bam*}
        else
            barcode="unclassified"
        fi
        # Lookup expected count, defaulting to 0 if not set.
        expected=expected_read_groups_${barcode}
        expected=${!expected:-0}
        num_read_groups=$(samtools view -c ${bam})
        if [[ $num_read_groups -ne $expected ]]; then
            echo "Barcoding read group '${barcode}' has incorrect number of reads. '${bam}': ${num_read_groups} != ${expected}"
            exit 1
        fi
        exit 0
    )
    for bam in $split_dir/rg_*.bam; do
        check_barcodes $bam
    done

    $dorado_bin basecaller ${models_directory_arg} -b ${batch} ${model_5k} ${demux_data} --no-trim >$output_dir/${output_name}-demux.bam
    $dorado_bin demux --no-trim --kit-name SQK-RBK114-96 ${sample_sheet:+--sample-sheet ${sample_sheet}} --output-dir $output_dir/${output_name}-demux $output_dir/${output_name}-demux.bam

    for bam in $output_dir/${output_name}-demux/*.bam; do
        check_barcodes $bam
    done
)

if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    # There should be 4 reads with BC01, 2 with BC04, and 1 unclassified groups.
    test_barcoding_read_groups barcode01 4 barcode04 2 unclassified 1
    # There should be 4 reads with BC01 aliased to patient_id_1, and 5 unclassified groups.
    test_barcoding_read_groups patient_id_1 4 unclassified 3 $data_dir/barcode_demux/sample_sheet.csv
fi

# Test demux only on a pre-classified BAM file
# Start by making an input file for demux:
demux_data=$data_dir/barcode_demux/read_group_test
$dorado_bin basecaller ${models_directory_arg} -b ${batch} --kit-name SQK-RBK114-96 ${model_5k} ${demux_data} --no-trim > $output_dir/read_group_test.bam

$dorado_bin demux --no-classify --output-dir "$output_dir/demux_only_test/" $output_dir/read_group_test.bam
for bam in $output_dir/demux_only_test/0d85015e-6a4e-400c-a80f-c187c65a6d03_SQK-RBK114-96_barcode01.bam $output_dir/demux_only_test/0d85015e-6a4e-400c-a80f-c187c65a6d03_SQK-RBK114-96_barcode04.bam $output_dir/demux_only_test/0d85015e-6a4e-400c-a80f-c187c65a6d03_unclassified.bam; do
    if [ ! -f $bam ]; then
        echo "Missing expected bam file $bam.  Generated files:"
        ls -l $output_dir/demux_only_test/
        exit 1
    fi
done

rm -rf $output_dir
