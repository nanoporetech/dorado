function prepare_mock_data {
    # Create a micro BAM file of only 1 record in the current directory.
    # The model name (in the header) of the original test data BAM will be replaced with
    # the target_model value.
    local in_bam="$1"; shift
    local source_model="$1"; shift      # E.g. "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    local target_model="$1"; shift      # E.g. "dna_r10.4.1_e8.2_400bps_hac@v6.1.0"
    local remove_rg="$1"; shift         # 1 to remove all @RG lines from the header
    local remove_tags="$1"; shift       # 1 to remove all optional tags
    local out_fn="$1"; shift

    # Take the header and rename the model name to a target model.
    if [[ "${remove_rg}" == "1" ]]; then
        samtools view -H ${in_bam} | sed "s/${source_model}/${target_model}/g" | grep -v "@RG" > ${out_fn}.sam
    else
        samtools view -H ${in_bam} | sed "s/${source_model}/${target_model}/g" > ${out_fn}.sam
    fi

    # Strip tags (e.g. for tests without dwells).
    if [[ "${remove_tags}" == "1" ]]; then
        samtools view -x "^" ${in_bam} | head -n 1 >> ${out_fn}.sam
    else
        samtools view ${in_bam} | sed "s/${source_model}/${target_model}/g" | head -n 1 >> ${out_fn}.sam
    fi

    # Convert to BAM.
    samtools view -Sb ${out_fn}.sam | samtools sort > ${out_fn}
    samtools index ${out_fn}
    rm -f ${out_fn}.sam
}

function prepare_variant_mock_data {
    local in_bam="$1"; shift
    local target_model="$1"; shift      # E.g. "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    local remove_rg="$1"; shift         # 1 to remove all @RG lines from the header
    local remove_tags="$1"; shift       # 1 to remove all optional tags
    local out_fn="$1"; shift
    local source_model="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    prepare_mock_data "${in_bam}" "${source_model}" "${target_model}" "${remove_rg}" "${remove_tags}" "${out_fn}"
}

function prepare_variant_mock_data_two_rg {
    # Create a BAM file with two read groups.
    local in_bam="$1"; shift
    local target_model_1="$1"; shift      # First read group, e.g. "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    local target_model_2="$1"; shift      # Second read group, e.g. "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    local out_fn="$1"; shift

    local source_model="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"

    prepare_variant_mock_data ${in_bam} ${target_model_1} 0 0 ${out_fn}.rg01.bam
    prepare_variant_mock_data ${in_bam} ${target_model_2} 0 0 ${out_fn}.rg02.bam

    rm -f ${out_fn}
    samtools merge ${out_fn} ${out_fn}.rg01.bam ${out_fn}.rg02.bam
    samtools index ${out_fn}
    rm -f ${out_fn}.rg01.bam ${out_fn}.rg02.bam
}
