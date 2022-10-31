#include "Version.h"
#include "data_loader/DataLoader.h"
#include "decode/CPUDecoder.h"
#ifdef __APPLE__
#include "nn/MetalCRFModel.h"
#include "utils/metal_utils.h"
#else
#include "nn/CudaCRFModel.h"
#include "utils/cuda_utils.h"
#endif
#include "3rdparty/edlib/edlib/include/edlib.h"
#include "htslib/sam.h"
#include "nn/ModelRunner.h"
#include "nn/RemoraModel.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/ModBaseCallerNode.h"
#include "read_pipeline/ScalerNode.h"
#include "read_pipeline/WriterNode.h"

#include <argparse.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

struct read {
    std::string read_id;
    std::vector<char> sequence;
    std::vector<uint8_t> scores;
};

struct read_pair {
    std::string temp;
    std::string comp;
};

void setup_duplex(std::vector<std::string> args,
                  const std::filesystem::path& model_path,
                  const std::string& data_path,
                  const std::string& remora_models,
                  const std::string& device,
                  size_t chunk_size,
                  size_t overlap,
                  size_t batch_size,
                  size_t num_runners,
                  size_t remora_batch_size,
                  size_t num_remora_threads,
                  bool emit_fastq) {
    torch::set_num_threads(1);
    std::vector<Runner> runners;

    int num_devices = 1;

    if (device == "cpu") {
        batch_size = batch_size == 0 ? std::thread::hardware_concurrency() : batch_size;
        for (int i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<ModelRunner<CPUDecoder>>(model_path, device,
                                                                        chunk_size, batch_size));
        }
#ifdef __APPLE__
    } else if (device == "metal") {
        batch_size = batch_size == 0 ? auto_gpu_batch_size(model_path.string()) : batch_size;
        auto caller = create_metal_caller(model_path, chunk_size, batch_size);
        for (int i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<MetalModelRunner>(caller, chunk_size, batch_size));
        }
    } else {
        throw std::runtime_error(std::string("Unsupported device: ") + device);
    }
#else   // ifdef __APPLE__
    } else {
        auto devices = parse_cuda_device_string(device);
        num_devices = devices.size();
        batch_size =
                batch_size == 0 ? auto_gpu_batch_size(model_path.string(), devices) : batch_size;
        for (auto device_string : devices) {
            auto caller = create_cuda_caller(model_path, chunk_size, batch_size, device_string);
            for (int i = 0; i < num_runners; i++) {
                runners.push_back(
                        std::make_shared<CudaModelRunner>(caller, chunk_size, batch_size));
            }
        }
    }
#endif  // __APPLE__

    // verify that all runners are using the same stride, in case we allow multiple models in future
    auto model_stride = runners.front()->model_stride();
    assert(std::all_of(runners.begin(), runners.end(), [model_stride](auto runner) {
        return runner->model_stride() == model_stride;
    }));

    if (!remora_models.empty() && emit_fastq) {
        throw std::runtime_error("Modified base models cannot be used with FASTQ output");
    }

    std::vector<std::filesystem::path> remora_model_list;
    std::istringstream stream{remora_models};
    std::string model;
    while (std::getline(stream, model, ',')) {
        remora_model_list.push_back(model);
    }

    // generate model callers before nodes or it affects the speed calculations
    std::vector<std::shared_ptr<RemoraCaller>> remora_callers;
    for (const auto& remora_model : remora_model_list) {
        auto caller = std::make_shared<RemoraCaller>(remora_model, device, remora_batch_size,
                                                     model_stride);
        remora_callers.push_back(caller);
    }

    WriterNode writer_node(std::move(args), emit_fastq, num_devices);

    std::unique_ptr<ModBaseCallerNode> mod_base_caller_node;
    std::unique_ptr<BasecallerNode> basecaller_node;

    if (!remora_model_list.empty()) {
        mod_base_caller_node.reset(new ModBaseCallerNode(writer_node, std::move(remora_callers),
                                                         num_remora_threads, model_stride,
                                                         remora_batch_size));
        basecaller_node =
                std::make_unique<BasecallerNode>(*mod_base_caller_node, std::move(runners),
                                                 batch_size, chunk_size, overlap, model_stride);
    } else {
        basecaller_node = std::make_unique<BasecallerNode>(
                writer_node, std::move(runners), batch_size, chunk_size, overlap, model_stride);
    }
    ScalerNode scaler_node(*basecaller_node, num_devices * 2);
    DataLoader loader(scaler_node, "cpu", num_devices);
    loader.load_reads(data_path);
}

int duplex(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION);

    parser.add_argument("model").help("the basecaller model to run.");

    parser.add_argument("data").help("the data directory.");

    parser.add_argument("-x", "--device")
            .help("device string in format \"cuda:0,...,N\", \"cuda:all\", \"metal\" etc..")
#ifdef __APPLE__
            .default_value(std::string{"metal"});
#else
            .default_value(std::string{"cuda:all"});
#endif

    parser.add_argument("-b", "--batchsize")
            .default_value(0)
            .scan<'i', int>()
            .help("if 0 an optimal batchsize will be selected");

    parser.add_argument("-c", "--chunksize").default_value(10000).scan<'i', int>();

    parser.add_argument("-o", "--overlap").default_value(500).scan<'i', int>();

    parser.add_argument("-r", "--num_runners").default_value(2).scan<'i', int>();

    parser.add_argument("--emit-fastq").default_value(false).implicit_value(true);

    parser.add_argument("--remora-batchsize").default_value(1000).scan<'i', int>();

    parser.add_argument("--remora-threads").default_value(1).scan<'i', int>();

    parser.add_argument("--remora_models")
            .default_value(std::string())
            .help("a comma separated list of remora models");

    std::cerr << "Loading BAM" << std::endl;

    samFile* fp_in = hts_open(
            "/media/groups/machine_learning/active/mvella/duplex_data/kit14_260bps_duplex_test_set/calls.bam",
            "r");                             //open bam file
    bam_hdr_t* bamHdr = sam_hdr_read(fp_in);  //read header
    bam1_t* aln = bam_init1();                //initialize an alignment
    std::cerr << "Header:\n " << bamHdr->text << std::endl;

    auto a = sam_read1(fp_in, bamHdr, aln);
    std::map<std::string, read> reads;
    while (sam_read1(fp_in, bamHdr, aln) >= 0) {
        //int32_t pos = aln->core.pos +1; //left most position of alignment in zero based coordianate (+1)
        //char *chr = bamHdr->target_name[aln->core.tid] ; //contig name (chromosome)
        uint32_t len = aln->core.l_qseq;  //length of the read.

        std::string read_id = bam_get_qname(aln);

        uint8_t* q = bam_get_qual(aln);  //quality string
        uint8_t* s = bam_get_seq(aln);   //sequence string

        std::vector<uint8_t> qualities(len);
        std::vector<char> nucleotides(len);

        // Todo - there is a better way to do this.
        for (int i = 0; i < len; i++) {
            qualities[i] = q[i];
            nucleotides[i] = seq_nt16_str[bam_seqi(s, i)];
        }
        reads[read_id] = {read_id, nucleotides, qualities};
    }
    std::cerr << std::endl;
    std::cerr << "Exit While" << std::endl;

    bam_destroy1(aln);
    sam_close(fp_in);
    std::cerr << "Closing BAM - DONE" << std::endl;

    // Let's also load a pairs file
    std::string pairs_file =
            "/media/groups/machine_learning/active/mvella/duplex_data/kit14_260bps_duplex_test_set/pair_ids_filtered.txt";

    std::ifstream dataFile;
    dataFile.open(pairs_file);

    std::vector<std::string> resultr;

    std::string cell;
    int line = 0;

    std::map<std::string, std::string> t_c_map;
    std::map<std::string, std::string> c_t_map;

    std::getline(dataFile, cell);
    while (!dataFile.eof()) {
        char delim = ' ';
        auto delim_pos = cell.find(delim);

        std::string t = cell.substr(0, delim_pos);
        std::string c = cell.substr(delim_pos + 1, delim_pos * 2 - 1);
        t_c_map[t] = c;
        c_t_map[c] = t;

        line++;
        std::getline(dataFile, cell);
    }

    // Let's now perform alignmnet on all pairs:
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    std::map<char,char> complementary_nucleotides;

    complementary_nucleotides['A'] = 'T';
    complementary_nucleotides['C'] = 'G';
    complementary_nucleotides['G'] = 'C';
    complementary_nucleotides['T'] = 'A';

    int i = 0;
    for (auto key : t_c_map) {
        std::string temp_id = key.first;
        std::string comp_id = key.second;

        std::vector<char> temp_str;
        std::vector<char> comp_str;
        std::vector<uint8_t> temp_q_string;
        if (reads.find(temp_id) == reads.end()) {
        } else {
            auto read = reads.at(temp_id);
            temp_str = read.sequence;
            temp_q_string = read.scores;
            std::cerr << "found a template!" << std::endl;
        }

        auto opts = torch::TensorOptions().dtype(torch::kInt8);
        torch::Tensor t = torch::from_blob(temp_q_string.data(),
                                           {1, (int) temp_q_string.size()}, opts);
        auto t_float = t.to(torch::kFloat32);
        int pool_window = 5;
        t.index({torch::indexing::Slice()}) = -torch::max_pool1d(-t_float,
                                     pool_window,
                                     1,
                                     pool_window / 2);

        //t.index({0, torch::indexing::Slice()}) = t_float.index({0, torch::indexing::Slice()});

        if (reads.find(comp_id) == reads.end()) {
            std::cerr << "Corresponding complement is missing" << std::endl;
        } else if (temp_str.size() != 0) {// We can do the alignment
            auto complement_read = reads.at(comp_id);
            comp_str = complement_read.sequence;
            auto comp_q_scores_reverse = complement_read.scores;
            std::reverse(comp_q_scores_reverse.begin(), comp_q_scores_reverse.end());

            std::vector<char> comp_str_rc = comp_str;
            //compute the RC
            std::reverse(comp_str_rc.begin(), comp_str_rc.end());
            std::for_each(comp_str_rc.begin(), comp_str_rc.end(), [&complementary_nucleotides](char &c){ c=complementary_nucleotides[c]; });

            EdlibAlignResult result = edlibAlign(temp_str.data(), temp_str.size(), comp_str_rc.data(),
                                                 comp_str_rc.size(), align_config);

            //Now - we have to do the actual basespace alignment itself

            std::vector<char> consensus;
            int query_cursor = 0;
            int target_cursor = result.startLocations[0];
            for (int i=0; i<result.alignmentLength; i++){
                if (temp_q_string[target_cursor] >= comp_q_scores_reverse[query_cursor]){
                    consensus.push_back(temp_str[target_cursor]);
                } else{
                    consensus.push_back(comp_str_rc[query_cursor]);
                }

                //Anything but a query insertion and target advances
                if (result.alignment[i] != 2) {
                    target_cursor++;
                }

                //Anything but a target insertion and target advances
                if (result.alignment[i] != 1) {
                    query_cursor++;
                }
            }
            std::cerr<< std::endl;
            for (auto &c: consensus){
                std::cerr << c;
            }
            std::cerr<< std::endl;
            edlibFreeAlignResult(result);
        }

        if (i > 10000) {
            break;
        }

        i++;
    }

    return 0;
}
