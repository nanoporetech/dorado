If this test fails, it means that one or more polishing models were added to Dorado.
Make sure that the integration Cram tests were updated to test run those models (i.e. add a new cram-polish-models-v?.?.?.t for the new basecaller model).
  $ ${DORADO_BIN} download --list-yaml 2>/dev/null 1>out.yaml
  > cat out.yaml | grep "dna.*polish.*"
    - "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl"
    - "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"
    - "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_polish_rl"
    - "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_polish_rl_mv"
    - "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_polish_rl"
    - "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_polish_rl_mv"
    - "dna_r10.4.1_e8.2_400bps_sup@v5.2.0_polish_rl"
    - "dna_r10.4.1_e8.2_400bps_sup@v5.2.0_polish_rl_mv"
    - "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"
    - "dna_r10.4.1_e8.2_400bps_hac@v4.2.0_polish"
    - "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_polish"
    - "dna_r10.4.1_e8.2_400bps_hac@v4.3.0_polish"
    - "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_polish"
