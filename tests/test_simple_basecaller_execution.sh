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

ONT_OUTPUT_SPEC_REF="11a7f1001ad04484ce7ef84038168969139a9b15"
SPECIFICATION_URL="${ONT_OUTPUT_SPEC_REPO}-/archive/${ONT_OUTPUT_SPEC_REF}/ont-output-specification-${ONT_OUTPUT_SPEC_REF}.zip"
SPECIFICATION_FILE="ont_output_spec.zip"
VALIDATOR_COMMIT="156b6e2ebbe0c832f9f568166797205758b83a73"

# Set up the output specification validator so we can check output file formats
if [[ "${VALIDATE_FASTQ}" -eq "1" || "${VALIDATE_BAM}" -eq "1" ]]; then
    echo "Enabling validation of output files against spec from ${SPECIFICATION_URL}"
    $PYTHON --version
    $PYTHON -m venv venv
    source ./venv/*/activate
    # Install output-file specification validator.
    rm -rf ont-output-specification-validator
    git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@${VALIDATOR_REPO}
    pushd ont-output-specification-validator
    git checkout ${VALIDATOR_COMMIT}
    popd
    # Note we use --prefer-binary to avoid issues with h5py 3.15, see DOR-1410
    pip install --prefer-binary -e ont-output-specification-validator
    curl -LfsS ${SPECIFICATION_URL} > ${SPECIFICATION_FILE}
fi

echo dorado download models
$dorado_bin download --list
$dorado_bin download --list-structured | $PYTHON ${test_dir}/validate_json.py -
$dorado_bin download --model ${model_name_5k} ${models_directory_arg}
model_5k=${models_directory}/${model_name_5k}
$dorado_bin download --model ${model_name_5k_v43} ${models_directory_arg}
model_5k_v43=${models_directory}/${model_name_5k_v43}
$dorado_bin download --model ${model_name_rna004} ${models_directory_arg}
model_rna004=${models_directory}/${model_name_rna004}

dorado_check_bam_not_empty() {
    local htslib_file="${1:-"${output_dir}/calls.bam"}"

    if [[ "${VALIDATE_BAM}" -eq "1" ]]; then
        $PYTHON ${test_dir}/validate_bam.py ${htslib_file} $SPECIFICATION_FILE
    fi

    if [[ -n "$SAMTOOLS_UNAVAILABLE" ]]; then
        echo "Skipping dorado_check_bam_not_empty as SAMTOOLS_UNAVAILABLE is set"
        return 0
    fi
    samtools quickcheck -u ${htslib_file}
    samtools view -h ${htslib_file} > $output_dir/calls.sam
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
    $PYTHON ${test_dir}/validate_fastq.py $output_dir/ref.fq $SPECIFICATION_FILE
fi
$dorado_bin basecaller ${model_5k} $pod5_data ${models_directory_arg} -b ${batch} --modified-bases 5mCG_5hmCG --emit-moves > $output_dir/calls.bam
dorado_check_bam_not_empty
$dorado_bin basecaller ${model_5k} $pod5_data/ ${models_directory_arg} -x cpu --modified-bases 5mCG_5hmCG -vv > $output_dir/calls.bam
dorado_check_bam_not_empty

$dorado_bin basecaller $model_complex,5mCG_5hmCG $pod5_data/ ${models_directory_arg} -b ${batch} --emit-moves > $output_dir/calls.bam


dorado_check_sq_m5_headers() {
    if [[ -n "$SAMTOOLS_UNAVAILABLE" ]]; then
        echo "Skipping dorado_check_sq_m5_headers as SAMTOOLS_UNAVAILABLE is set"
        return 0
    fi

    local cram_path=$1
    local ref_path=$2
    local expected_m5=$3
    local header_path=$4

    if [[ -n "${ref_path}" ]]; then
        # We should be able to open the a CRAM without explicitly adding 
        # the reference path with `samtools view -T ${ref_path} ...` because we set SQ UR tag.
        samtools view -H  "${cram_path}" > "${header_path}"
        if [[ -n "${expected_m5}" ]]; then
            local observed_m5
            observed_m5=$(grep -E "^@SQ" "${header_path}" | awk -F'\t' '{for (i=1;i<=NF;i++) if ($i ~ /^M5:/) {print substr($i,4); exit}}')
            if [[ "${observed_m5}" != ${expected_m5} ]]; then
                echo "Header SQ M5 mismatch: expected ${expected_m5}, got ${observed_m5}"
                exit 1
            fi
        else
            if ! grep -q $'\tM5:' "${header_path}"; then
                echo "Header missing M5 tag"
                exit 1
            fi
        fi
        if ! grep -q $'\tUR:' "${header_path}"; then
            echo "Header missing UR tag"
            exit 1
        fi
    else
        samtools view -H "${cram_path}" > "${header_path}"
    fi
}

dorado_emit_cram_iupac_reference() {
    # Extract the expeted M5 tag from the samtools dict file
    local ref_dict=$data_dir/../cram/single_read/chr1_MAT_iupac.dict
    local expected_m5
    expected_m5=$(grep -E "^@SQ" "${ref_dict}" | awk -F'\t' '{for (i=1;i<=NF;i++) if ($i ~ /^M5:/) {print substr($i,4); exit}}')
    if [[ -z "${expected_m5}" ]]; then
        echo "Failed to extract expected M5 from ${ref_dict}"
        exit 1
    fi

    # IUPAC doped FASTA file and a single read which should align to it.
    local ref_fasta=$data_dir/../cram/single_read/chr1_MAT_iupac.fasta
    local pod5_single=$data_dir/../cram/single_read/chr1_MAT.pod5

    local cram_out=$output_dir/calls.cram
    local header_out=$output_dir/header.txt

    $dorado_bin basecaller ${model_5k} ${pod5_single} ${models_directory_arg} -b ${batch} --emit-cram --reference "${ref_fasta}" > "${cram_out}"
    dorado_check_sq_m5_headers "${cram_out}" "${ref_fasta}" "${expected_m5}" "${header_out}"
    dorado_check_bam_not_empty "${cram_out}"
}

dorado_emit_cram_iupac_reference


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
    mkdir -p $output_dir/read_splitting
    $dorado_bin basecaller ${model_5k} ${data_dir}/single_split_read > $output_dir/read_splitting/calls.bam
    if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
        num_sam_records=$(samtools view $output_dir/read_splitting/calls.bam | wc -l | awk '{print $1}')
        if [[ $num_sam_records -ne "2" ]]; then
            echo "Expected 2 sam records from 1 split read but found: ${num_sam_records}"
            exit 1
        fi
        num_unique_parents=$(samtools view $output_dir/read_splitting/calls.bam | grep "pi:Z:[^\s]*" -oh | uniq | wc -l | awk '{print $1}')
        if [[ $num_unique_parents -ne "1" ]]; then
            echo "Expected 1 unique parent read id in 'pi:Z' tag but found: ${num_unique_parents}"
            exit 1
        fi
    fi

    echo "Testing split read with '--disable-read-splitting'"
    $dorado_bin basecaller ${model_5k} ${data_dir}/single_split_read --disable-read-splitting > $output_dir/read_splitting/calls-no-split.bam
    if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
        num_sam_records=$(samtools view $output_dir/read_splitting/calls-no-split.bam | wc -l | awk '{print $1}')
        if [[ $num_sam_records -ne "1" ]]; then
            echo "Expected 1 sam records from 1 unsplit read with '--disable-read-splitting' but found: ${num_sam_records}"
            exit 1
        fi
        num_parent_tags=$(samtools view $output_dir/read_splitting/calls-no-split.bam | grep "pi:Z:[^\s]*" -oh | wc -l | awk '{print $1}')
        if [[ $num_parent_tags -ne "0" ]]; then
            echo "Expected 0 instances of 'pi:Z' tag with '--disable-read-splitting' but found: ${num_parent_tags}"
            exit 1
        fi
    fi
    set -e
}

echo dorado summary test stage
$dorado_bin summary $output_dir/read_splitting/calls.bam > /dev/null
$dorado_bin summary -r $output_dir/read_splitting > /dev/null
$dorado_bin basecaller $model_complex $pod5_data/ -b ${batch} | $dorado_bin summary > /dev/null
$dorado_bin summary $output_dir/not_a_real_file.txt

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

function dorado_aligner_secondary_supplementary_test {
    ###############################################################
    ### This tests that Dorado Aligner will not align           ###
    ### secondary/supplementary alignments when input is a BAM. ###
    ###############################################################
    set +e
    set +x

    echo "Testing Dorado Aligner - realigning a BAM with secondary/supplementary alignments"

    local in_all_reads="${data_dir}/aligner_test/dataset.fastq"
    local in_all_ref="${data_dir}/aligner_test/lambda_ecoli.fasta"

    local local_out_dir="${output_dir}/aligner-01"
    local generated_ref="${local_out_dir}/ref.fasta"
    local generated_reads="${local_out_dir}/reads.fasta"

    mkdir -p ${local_out_dir}

    # Create a synthetic reference which will cause secondary/supplementary alignments.
    samtools faidx ${in_all_ref} "Lambda:1-45000" > ${generated_ref}
    samtools faidx ${in_all_ref} "Lambda:45001-48400" >> ${generated_ref}
    samtools faidx ${in_all_ref} "Lambda:1-10000" >> ${generated_ref}
    samtools faidx ${generated_ref}

    # Create a small input - extract only a handful of reads.
    # There are duplicate reads in the input `datasets.fastq` file.
    # This has a unique mapping and should be just a primary alignment:
    #       `de6f1738-5247-4648-9bc6-c12dc681f029    7191    2       7187    +       Lambda  48400   17660   24971   6937    7185    60`
    local seq1_prim=$(grep -A1 "de6f1738-5247-4648-9bc6-c12dc681f029" ${in_all_reads} | head -n 2 | tail -n 1)
    # This one aligns further down and is longer, good for a split alignment:
    #       `c7bbae11-7498-45a2-99bc-b03194889968    11837   32      11821   -       Lambda  48400   36158   48400   11556   11789   60`
    local seq2_suppl=$(grep -A1 "c7bbae11-7498-45a2-99bc-b03194889968" ${in_all_reads} | head -n 2 | tail -n 1)
    # This one alignes to the front of the reference:
    #       `b4195a0e-3a95-4f96-8852-2aa6beba19a8    3433    32      3416    -       Lambda  48400   0       3650    3225    3384    60`
    local seq3_sec=$(grep -A1 "b4195a0e-3a95-4f96-8852-2aa6beba19a8" ${in_all_reads} | head -n 2 | tail -n 1)

    # Write the test input.
    printf ">read1-prim\n%s\n" ${seq1_prim} > ${generated_reads}
    printf ">read2-suppl\n%s\n" ${seq2_suppl} >> ${generated_reads}
    printf ">read3-sec\n%s\n" ${seq3_sec} >> ${generated_reads}
    samtools faidx ${generated_reads}

    # 1. Align the input reads.
    ${dorado_bin} aligner ${generated_ref} ${generated_reads} | samtools sort > ${local_out_dir}/aligned.01.bam

    # 2. Align manually filtered BAM.
    samtools view -hb -F 0x900 ${local_out_dir}/aligned.01.bam > ${local_out_dir}/aligned.01.filtered.bam
    ${dorado_bin} aligner ${generated_ref} ${local_out_dir}/aligned.01.filtered.bam | samtools sort > ${local_out_dir}/aligned.02.bam
    samtools view ${local_out_dir}/aligned.02.bam > ${local_out_dir}/aligned.02.sam

    # 3. Align the full unfiltered aligned BAM.
    ${dorado_bin} aligner ${generated_ref} ${local_out_dir}/aligned.01.bam | samtools sort > ${local_out_dir}/aligned.03.bam
    samtools view ${local_out_dir}/aligned.03.bam > ${local_out_dir}/aligned.03.sam

    # 4. Align the full unfiltered aligned BAM and allow the secondary/supplementary alignments to be used as input.
    ${dorado_bin} aligner --allow-sec-supp ${generated_ref} ${local_out_dir}/aligned.01.bam | samtools sort > ${local_out_dir}/aligned.04.bam
    samtools view ${local_out_dir}/aligned.04.bam > ${local_out_dir}/aligned.04.sam

    # Test that the default does not align secondary/supplementary.
    local num_diff_lines=$(diff ${local_out_dir}/aligned.02.sam ${local_out_dir}/aligned.03.sam | wc -l | awk '{ print $1 }')
    if [[ "${num_diff_lines}" != "0" ]]; then
        echo "ERROR: Dorado Aligner also aligned secondary/supplementary records!"
        exit 1
    fi

    # Test that the `--alow-sec-supp` will also realign the secondary/supplementary alignments.
    echo "read1-prim 0 Lambda:1-45000 17661 60
read2-suppl 16 Lambda:1-45000 36159 60
read2-suppl 16 Lambda:45001-48400 1 60
read2-suppl 2064 Lambda:45001-48400 1 60
read3-sec 16 Lambda:1-45000 1 0
read3-sec 272 Lambda:1-10000 1 0
read3-sec 4 * 0 0" > ${local_out_dir}/expected.txt
    samtools view ${local_out_dir}/aligned.04.bam | awk '{ print $1, $2, $3, $4, $5 }' | sort > ${local_out_dir}/result.txt
    set -ex
    diff ${local_out_dir}/expected.txt ${local_out_dir}/result.txt
    set +ex

    echo success
}

function dorado_aligner_realigning_and_unmapped {
    ###############################################################
    ### When Dorado Aligner fails to realign a read (i.e. it is ###
    ### unmapped), it should reset the alignment information.   ###
    ###############################################################
    set +e
    set +x

    echo "Testing Dorado Aligner - realigning a BAM with secondary/supplementary alignments"

    local in_all_reads="${data_dir}/aligner_test/dataset.fastq"
    local in_all_ref="${data_dir}/aligner_test/lambda_ecoli.fasta"

    local local_out_dir="${output_dir}/aligner-02"
    local generated_ref="${local_out_dir}/lambda.fasta"
    local generated_dummy_ref="${local_out_dir}/dummy.fasta"
    local generated_reads="${local_out_dir}/reads.fasta"

    mkdir -p ${local_out_dir}

    # Generate a small test case of 1 read and the Lambda reference, plus a dummy reference for realignemnt.
    samtools faidx ${in_all_ref} "Lambda" > ${generated_ref}
    samtools faidx ${in_all_ref} "Ecoli:100001-110000" > ${generated_dummy_ref}
    local seq1=$(grep -A1 "de6f1738-5247-4648-9bc6-c12dc681f029" ${in_all_reads} | head -n 2 | tail -n 1)
    printf ">de6f1738-5247-4648-9bc6-c12dc681f029\n%s\n" ${seq1} > ${generated_reads}

    # 1. Align the input read to create a mapped record.
    ${dorado_bin} aligner ${generated_ref} ${generated_reads} | samtools sort > ${local_out_dir}/aligned.01.bam

    # 2. Realign the mapped BAM to a dummy reference. The read should not be aligned, and the mapping information should be reset.
    ${dorado_bin} aligner ${generated_dummy_ref} ${local_out_dir}/aligned.01.bam | samtools sort > ${local_out_dir}/aligned.02.bam

    echo "de6f1738-5247-4648-9bc6-c12dc681f029	0	Lambda	17661	60	2S7M1I2M1I5M5D31M3I23M1I29M2D10M1D1M3D9M3I1M1I69M1I4M1I27M1D11M1D5M1D24M1I6M2D20M2I30M1D17M1D12M1D3M1I31M1I11M1D39M3I41M1I83M2D11M1D46M5D20M1D38M1D46M1I16M2I12M2D62M1D16M2I5M1I47M1I32M1I18M2I11M1I36M1I42M3D22M2D2M1I5M2D2M3D8M3D4M3I2M3I5M2D4M1I14M2D26M1I36M1I9M2I18M2D1M1D2M3D16M1D21M1I5M1I43M2I42M1I9M1I27M1D4M2D4M1I61M4D17M2I18M7D24M3D40M1D21M1I2M1I15M1I24M1I2M1I9M2D3M1D7M1I54M1D11M1I16M1D8M1I13M2I19M1D2M3D15M1D27M2D10M1D13M1I5M4D1M1D26M1I34M1D20M1D6M2I10M2D13M1I56M1I10M1I58M4D44M5D9M1I6M1I3M1D42M1D2M3I21M1I14M2D21M1D10M1I4M3D39M2I10M2D4M1I8M2I5M1I29M4D15M1D1M1D11M3D1M2D12M2I2M1I10M1I18M1D28M1D2M1I25M1I13M2I15M2I18M1D1M1D1M2D20M2I7M1I12M4D8M2I18M1D10M1I2M2D4M1D23M4D9M1D27M1I25M1D6M2I7M1D13M1I8M2I5M1I6M1I7M1I31M4I8M1D39M1D9M1D9M1D17M1D3M1D30M3D59M2D17M1I11M2I3M3D17M4I24M1I6M2I3M1I20M1I4M1I14M2I9M1I31M1I8M2I29M2D36M1I14M1D2M2D63M4D96M1I9M2I2M1I13M5D22M2I34M1I19M1D10M1D59M1I59M1D15M2D5M1D5M1D1M1D32M6D7M4I3M2D18M2D3M2D2M2D72M2I3M3I15M1I28M1I33M2D126M1D2M1D2M1I15M1I5M1D1M2I2M1D23M1I8M2D28M1I25M2D7M1I31M1I8M1D10M1D28M3D15M3I20M1D39M1I11M2D14M3I9M4D54M1D27M2D6M1D3M1I5M6D21M1D24M2D9M5D8M4D3M1I12M2D8M1D47M1I12M1I7M1I9M1D15M2D16M2D5M2I45M6D61M3D3M1I93M2I21M4I5M1D10M1I53M4D32M1D6M1I1M1I4M1I33M1D7M2I15M1I41M1D15M2D3M2D11M1D6M1D24M2I2M2D22M1D51M1I83M1I20M1D11M1D24M2D2M3D18M1D19M1D7M1I16M7D5M1D30M4D13M1I42M2D58M1D13M1D51M1D6M1D21M3I3M1D22M1I7M1I56M1I52M1I12M1I3M1D23M1D16M5I31M3D6M1D32M1D20M1D7M1I1M1I8M1I22M1D16M1D4M1I11M1I1M2I8M1I2M1I8M3D38M1D6M1D21M1D70M1D4M1I59M1D45M3I4M1I68M1I12M1I26M3D36M1I9M1D11M3D27M2D7M2D14M1I6M1D40M5I4M2D1M1D30M1D10M1D7M1D4M1I7M1D17M3D15M1I4M1D28M1I20M1I19M1D8M1D3M2D43M1D46M9D5M1I25M1I5M1I86M1D53M1I1M2I8M2D16M3I19M2I12M1I6M4D19M2D20M4D5M1D5M1I6M1D24M1I7M4S	*	0	0" > ${local_out_dir}/expected.01.txt
    echo "de6f1738-5247-4648-9bc6-c12dc681f029	4	*	0	0	*	*	0	0" > ${local_out_dir}/expected.02.txt

    samtools view ${local_out_dir}/aligned.01.bam | cut -f 1-9 > ${local_out_dir}/aligned.01.bam.first_9
    samtools view ${local_out_dir}/aligned.02.bam | cut -f 1-9 > ${local_out_dir}/aligned.02.bam.first_9

    set -ex
    # Check that the first BAM is successfully mapped.
    diff ${local_out_dir}/expected.01.txt ${local_out_dir}/aligned.01.bam.first_9
    # Check that the second BAM is not mapped.
    diff ${local_out_dir}/expected.02.txt ${local_out_dir}/aligned.02.bam.first_9
    set +ex

    echo success
}

dorado_aligner_options_test
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    dorado_aligner_secondary_supplementary_test
    dorado_aligner_realigning_and_unmapped
fi

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
num_summary_lines=$(wc -l < $output_dir/aligned/sequencing_summary.txt)
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

    $dorado_bin aligner $data_dir/aligner_test/basecall_target.fa $data_dir/aligner_test/example-hts.fastq.gz --output-dir $output_dir/aligned/rg/
    check_add_fastq_rg_header_count 1 $output_dir/aligned/rg/PCR_zymo/20221018_1018_0_PAM93185_4524e8b9/bam_pass/alias_for_bc03/PAM93185_pass_alias_for_bc03_4524e8b9_00000000_0.bam "includes RG header lines"
fi

echo dorado demux test stage
$dorado_bin demux $data_dir/barcode_demux/double_end_variant/EXP-PBC096_BC04.fastq --kit-name EXP-PBC096 --output-dir $output_dir/demux --emit-summary
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    expected_path="$output_dir/demux/no_sample/19700101_0000_0_UNKNOWN_00000000/bam_pass/barcode04/UNKNOWN_pass_barcode04_00000000_00000000_0.bam"
    samtools quickcheck -u $expected_path
    num_demuxed_reads=$(samtools view -c $expected_path)
    if [[ $num_demuxed_reads -ne "3" ]]; then
        echo "3 demuxed reads expected. Found ${num_demuxed_reads}"
        exit 1
    fi
fi

$dorado_bin demux $data_dir/barcode_demux/double_end_variant/ --kit-name EXP-PBC096 --output-dir $output_dir/demux_from_folder
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    expected_path="$output_dir/demux_from_folder/no_sample/19700101_0000_0_UNKNOWN_00000000/bam_pass/barcode04/UNKNOWN_pass_barcode04_00000000_00000000_0.bam"
    samtools quickcheck -u $expected_path
fi
num_summary_lines=$(wc -l < $output_dir/demux/sequencing_summary.txt)
if [[ $num_summary_lines -ne "4" ]]; then
    echo "4 lines in summary expected. Found ${num_summary_lines}"
    exit 1
fi

echo dorado custom demux test stage
$dorado_bin demux $data_dir/barcode_demux/double_end/SQK-RPB004_BC01.fastq --output-dir $output_dir/custom_demux --kit-name CUSTOM-SQK-RPB004 --barcode-arrangement $data_dir/barcode_demux/custom_barcodes/RPB004.toml --barcode-sequences $data_dir/barcode_demux/custom_barcodes/RPB004_sequences.fasta
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    expected_path="$output_dir/custom_demux/no_sample/19700101_0000_0_UNKNOWN_00000000/bam_pass/barcode01/UNKNOWN_pass_barcode01_00000000_00000000_0.bam"
    samtools quickcheck -u $expected_path
    num_demuxed_reads=$(samtools view -c $expected_path)
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
    num_estimated_reads=$(samtools view $output_dir/no_detect_cdna_polya.bam | grep pt:i: | wc -l | awk '{print $1}')
    if [[ $num_estimated_reads -ne "2" ]]; then
        echo "2 poly(A) estimated reads expected. Found ${num_estimated_reads}"
        exit 1
    fi    
fi

$dorado_bin basecaller ${models_directory_arg} -b ${batch} ${model_5k} $data_dir/poly_a/r10_4_1_5khz_cdna_pod5/ --kit-name SQK-PCB114-24 --estimate-poly-a --poly-a-config $data_dir/poly_a/configs/polya_bc01_disabled.toml > $output_dir/disabled_cdna_polya.bam
if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    samtools quickcheck -u $output_dir/disabled_cdna_polya.bam
    # grep returns 1 if no lines matched, so add the { ... || test } part to stop -o pipefail from aborting
    num_estimated_reads=$(samtools view $output_dir/disabled_cdna_polya.bam | { grep pt:i: || test $? = 1; } | wc -l | awk '{print $1}')
     if [[ $num_estimated_reads -ne "0" ]]; then
         echo "0 poly(A) estimated reads expected. Found ${num_estimated_reads}"
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

    check_barcodes() (
        bam=$1
        echo "Checking file: $bam"
        if [[ $bam =~ _SQK-RBK114-96_(.+)\.bam ]]; then
            # Arrangement is |<kit>_<barcode>|, so find the barcode between the kit and the extension
            barcode=${BASH_REMATCH[1]}
        elif [[ $bam =~ _${model_name_5k}_(.+)\.bam ]]; then
            # Arrangement is |<barcode_alias>|, so find the barcode between the model name and the extension
            barcode=${BASH_REMATCH[1]}
        elif [[ $bam =~ rg_.*\.bam ]]; then
            # Split bam file that doesn't contain a barcode, therefore unclassified
            barcode="unclassified"
        elif [[ $bam =~ bam_pass/(.+)/PAO25751 ]]; then
            # Demuxed file, grab the barcode from the path
            barcode=${BASH_REMATCH[1]}
        else
            echo "Unexpected filename structure: $bam"
            exit 1
        fi
        # Lookup expected count, defaulting to 0 if not set.
        expected=expected_read_groups_${barcode}
        expected=${!expected:-0}
        num_read_groups=$(samtools view -c ${bam})
        if [[ $num_read_groups -ne $expected ]]; then
            echo "Barcoding read group '${barcode}' has incorrect number of reads. '${bam}': ${num_read_groups} != ${expected}"
            exit 1
        fi

        num_rg_lines=$(samtools view -H ${bam} | grep "@RG" | wc -l)
        if [[ $num_rg_lines -ne 1 ]]; then
            echo "Barcoding read group '${barcode}' has incorrect number of RG headers. '${bam}': ${num_rg_lines} != 1"
            exit 1
        fi
        exit 0
    )

    split_dir=$output_dir/${output_name}
    mkdir $split_dir
    samtools split -u $split_dir/unknown.bam -f "$split_dir/rg_%!.bam" $output_dir/${output_name}.bam

    # There shouldn't be any unknown groups.
    num_read_groups=$(samtools view -c $split_dir/unknown.bam)
    if [[ $num_read_groups -ne "0" ]]; then
        echo "Reads with unknown read groups found."
        exit 1
    fi
    for bam in $(find "$split_dir" -type f -iname "rg_*.bam" ); do
        check_barcodes $bam
    done

    # check that we correctly barcode and demux a basecalled bam file
    $dorado_bin basecaller ${models_directory_arg} -b ${batch} ${model_5k} ${demux_data} --no-trim >$output_dir/${output_name}-demux.bam
    $dorado_bin demux --no-trim --kit-name SQK-RBK114-96 ${sample_sheet:+--sample-sheet ${sample_sheet}} --output-dir $output_dir/${output_name}-demux $output_dir/${output_name}-demux.bam
    for bam in $(find "$output_dir/${output_name}-demux/" -type f -iname "*.bam" ); do
        check_barcodes $bam
    done

    # check that we correctly demux a basecalled and barcoded bam file
    $dorado_bin demux --no-classify --output-dir $output_dir/${output_name}-demux-no-classify $output_dir/${output_name}.bam
    for bam in $(find "$output_dir/${output_name}-demux-no-classify/" -type f -iname "*.bam" ); do
        check_barcodes $bam
    done
)

if [[ -z "$SAMTOOLS_UNAVAILABLE" ]]; then
    # There should be 4 reads with BC01, 2 with BC04, and 1 unclassified groups.
    test_barcoding_read_groups barcode01 4 barcode04 2 unclassified 1
    # There should be 4 reads with BC01 aliased to patient_id_1, and 3 unclassified groups.
    test_barcoding_read_groups patient_id_1 4 unclassified 3 $data_dir/barcode_demux/sample_sheet.csv
fi

rm -rf $output_dir
