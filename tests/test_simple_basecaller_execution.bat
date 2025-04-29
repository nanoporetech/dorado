set dorado_bin=%1
set model=dna_r10.4.1_e8.2_400bps_hac@v5.0.0
set modbase=5mCG_5hmCG
set test_data=tests/data/pod5/dna_r10.4.1_e8.2_400bps_5khz
set model_speed=hac
set batch=384

echo dorado basecaller test stage
%dorado_bin% download --model %model%
if %errorlevel% neq 0 exit /b %errorlevel%
%dorado_bin% basecaller %model% %test_data% -b %batch% --emit-fastq > ref.fq
if %errorlevel% neq 0 exit /b %errorlevel%
%dorado_bin% basecaller %model% %test_data% -b %batch% --modified-bases %modbase% --emit-moves --reference ref.fq --emit-sam > calls.sam
if %errorlevel% neq 0 exit /b %errorlevel%

echo dorado summary test stage
%dorado_bin% summary calls.sam
if %errorlevel% neq 0 exit /b %errorlevel%

echo dorado aligner test stage
%dorado_bin% aligner ref.fq calls.sam > aligned-calls.bam
if %errorlevel% neq 0 exit /b %errorlevel%

echo dorado duplex basespace test stage
%dorado_bin% duplex basespace tests/data/basespace/pairs.bam --threads 1 --pairs tests/data/basespace/pairs.txt > calls.bam
if %errorlevel% neq 0 exit /b %errorlevel%

echo dorado duplex hac complex
%dorado_bin% duplex hac tests/data/duplex/pod5 --threads 1  > $output_dir/duplex_calls.bam
if %errorlevel% neq 0 exit /b %errorlevel%

echo dorado duplex hac complex with mods
%dorado_bin% duplex %model_speed%,%modbase% tests/data/duplex/pod5 --threads 1 > $output_dir/duplex_calls_mods.bam
if %errorlevel% neq 0 exit /b %errorlevel%

echo dorado demux test stage
%dorado_bin% demux tests/data/barcode_demux/double_end_variant/EXP-PBC096_BC83.fastq --threads 1 --kit-name EXP-PBC096 --output-dir ./demux
if %errorlevel% neq 0 exit /b %errorlevel%

echo dorado auto model basecaller test stage
%dorado_bin% basecaller %model_speed% %test_data% -b %batch% --emit-fastq > ref.fq
if %errorlevel% neq 0 exit /b %errorlevel%
%dorado_bin% basecaller %model_speed%,%modbase% %test_data% -b %batch% --emit-moves --reference ref.fq --emit-sam > calls.sam
if %errorlevel% neq 0 exit /b %errorlevel%

echo dorado auto summary test stage
%dorado_bin% summary calls.sam
if %errorlevel% neq 0 exit /b %errorlevel%
%dorado_bin% correct tests/data/read_correction/reads.fq --threads 1 -v > $output_dir/corrected.fasta 

echo dorado correct test stage

echo finished
