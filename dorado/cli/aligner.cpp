#include "Version.h"
#include "minimap.h"
#include "utils/bam_utils.h"
#include "utils/log_utils.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <string>
#include <vector>

namespace dorado {

int aligner(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_argument("index").help("Index in fasta/mmi format.");
    parser.add_argument("reads").help("Reads in BAM/SAM/CRAM format.");
    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    if (parser.get<bool>("--verbose")) {
        spdlog::set_level(spdlog::level::debug);
    }

    auto index(parser.get<std::string>("index"));
    auto reads(parser.get<std::string>("reads"));

    utils::Aligner aligner(index);
    utils::BamReader reader(reads);
    utils::BamWriter writer("-", reader.m_header, aligner.get_idx_records());

    spdlog::info("> input fmt: {} aligned: {}", reader.m_format, reader.m_is_aligned);

    while (reader.next()) {
        // todo: move to reader
        int length = reader.m_record->core.l_qseq;
        auto useq = bam_get_seq(reader.m_record);
        std::vector<char> nucleotides(length);
        for (int i = 0; i < length; i++) {
            nucleotides[i] = seq_nt16_str[bam_seqi(useq, i)];
        }
        auto qname = bam_get_qname(reader.m_record);

        // todo: loop results
        auto alignment = aligner.align(length, nucleotides.data(), qname);
        writer.write_record(reader.m_record, alignment);
    }

    return 0;
}

}  // namespace dorado
