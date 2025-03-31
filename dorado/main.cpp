#include "cli/cli.h"
#include "dorado_version.h"
#include "utils/locale_utils.h"
#include "utils/log_utils.h"
#include "utils/string_utils.h"

#include <minimap.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <torch/version.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#ifndef __APPLE__
#include <cuda.h>
#endif

#ifdef __linux__
extern "C" {
// There's a bug in GLIBC < 2.25 (Bug 11941) which can trigger an assertion/
// seg fault when a dynamically loaded library is dlclose-d twice - once by ld.so and then
// once by the code that opened the DSO in the first place (more details available
// at https://sourceware.org/legacy-ml/libc-alpha/2016-12/msg00859.html). Dorado
// is seemingly running into this issue transitively through some dependent libraries
// (backtraces indicate libcudnn could be a source). The workaround below bypasses
// the dlclose subroutine entirely by making it a no-op. This will cause a memory
// leak as loaded shared libs won't be closed, but in practice this happens at
// teardown anyway so the leak will be subsumed by termination.
// Fix is borrowed from https://mailman.mit.edu/pipermail/cvs-krb5/2019-October/014884.html
#if !__GLIBC_PREREQ(2, 25)
int dlclose(void*);
int dlclose(void*) { return 0; };
#endif  // __GLIBC_PREREQ
}
#endif  // __linux__

#if TORCH_VERSION_MAJOR < 2 && !defined(TORCH_VERSION)
// One of our static builds doesn't define this, so do it ourself.
#define TORCH_VERSION \
    TORCH_VERSION_MAJOR << '.' << TORCH_VERSION_MINOR << '.' << TORCH_VERSION_PATCH
#endif  // TORCH_VERSION_MAJOR < 2

using entry_ptr = int (*)(int, char**);

namespace {

void usage(const std::vector<std::string>& commands) {
    std::cerr << "Usage: dorado [options] subcommand\n\n"
              << "Positional arguments:\n";

    for (const auto& command : commands) {
        std::cerr << command << '\n';
    }

    std::cerr << "\nOptional arguments:\n"
              << "-h --help               shows help message and exits\n"
              << "-v --version            prints version information and exits\n"
              << "-vv                     prints verbose version information and exits\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    // Load logging settings from environment/command-line.
    spdlog::cfg::load_env_levels();
    dorado::utils::InitLogging();
    dorado::utils::ensure_user_locale_may_be_set();

    const std::map<std::string, entry_ptr> subcommands = {
            {"basecaller", &dorado::basecaller},
            {"duplex", &dorado::duplex},
            {"download", &dorado::download},
            {"aligner", &dorado::aligner},
            {"summary", &dorado::summary},
            {"demux", &dorado::demuxer},
            {"trim", &dorado::trim},
            {"correct", &dorado::correct},
            {"polish", &dorado::polish},
    };

    std::vector<std::string> arguments(argv + 1, argv + argc);
    std::vector<std::string> keys;

    keys.reserve(subcommands.size());
    for (const auto& [key, _] : subcommands) {
        keys.push_back(key);
    }

    if (arguments.size() == 0) {
        usage(keys);
        return EXIT_SUCCESS;
    }

    // Log cmd
    spdlog::info("Running: \"{}\"", dorado::utils::join(arguments, "\" \""));

    const auto& subcommand = arguments[0];

    if (subcommand == "-v" || subcommand == "--version") {
        std::cerr << DORADO_VERSION << '\n';
    } else if (subcommand == "-vv") {
#ifdef __APPLE__
        std::cerr << "dorado:   " << DORADO_VERSION << '\n';
#else
        std::cerr << "dorado:   " << DORADO_VERSION << "+cu" << CUDA_VERSION << '\n';
#endif
        std::cerr << "libtorch: " << TORCH_VERSION << '\n';
        std::cerr << "minimap2: " << MM_VERSION << '\n';

    } else if (subcommand == "-h" || subcommand == "--help") {
        usage(keys);
        return EXIT_SUCCESS;
    } else if (subcommands.find(subcommand) != subcommands.end()) {
        return subcommands.at(subcommand)(--argc, ++argv);
    } else {
        usage(keys);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
