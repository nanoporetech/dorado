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
    parser.add_argument("reads")
            .help("Reads in BAM/SAM/CRAM format.")
            .nargs(argparse::nargs_pattern::any);
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
    auto reads(parser.get<std::vector<std::string>>("reads"));

    if (reads.size() == 0) {
        // todo: check stdin is a pipe and not empty
        reads.push_back("-");
    } else if (reads.size() > 1) {
        spdlog::error("> multi file input not yet handled");
        return 1;
    }

    utils::Aligner aligner(index);
    utils::BamReader reader(reads[0]);
    utils::BamWriter writer("-", reader.m_header, aligner.get_idx_records());

    spdlog::info("> input fmt: {} aligned: {}", reader.m_format, reader.m_is_aligned);

    while (reader.next()) {
        auto [hits, alignment] = aligner.align(reader.seq(), reader.qname());

        spdlog::info("> HITS {}", hits);
        for (int i = 0; i < hits; i++) {
            writer.write_record(reader.m_record, &alignment[i]);
        }
        if (hits == 0) {
            writer.write_record(reader.m_record);
        }
    }

    return 0;
}

}  // namespace dorado
