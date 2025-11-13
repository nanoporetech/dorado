#include "cli/cli.h"
#include "dorado_licences/licences.h"
#include "dorado_licences_pod5/licences.h"
#include "dorado_version.h"
#include "torch_utils/torch_utils.h"
#include "utils/crash_handlers.h"
#include "utils/locale_utils.h"
#include "utils/log_utils.h"
#include "utils/string_utils.h"

#include <minimap.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <torch/version.h>

#include <iostream>
#include <iterator>
#include <map>
#include <string_view>
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

using entry_ptr = int (*)(int, char**);

namespace {

void usage(const std::map<std::string_view, entry_ptr>& commands) {
    std::cout << "Usage: dorado [options] subcommand\n\n"
              << "Positional arguments:\n";

    for (const auto& command : commands) {
        std::cout << command.first << '\n';
    }

    std::cout << "\nOptional arguments:\n"
              << "-h --help               shows help message and exits\n"
              << "-l --licences           prints third party licences and exits\n"
              << "-v --version            prints version information and exits\n"
              << "-vv                     prints verbose version information and exits\n";
}

void print_licences() {
    std::cout << "Third party licences:\n";
    auto print = [](const auto& licences) {
        for (const auto [name, licence] : licences) {
            std::fill_n(std::ostream_iterator<char>(std::cout), name.size(), '-');
            std::cout << '\n' << name << '\n';
            std::fill_n(std::ostream_iterator<char>(std::cout), name.size(), '-');
            std::cout << '\n' << licence << '\n';
        }
    };
    print(dorado_licences::licences);
    print(dorado_licences_pod5::licences);
}

}  // namespace

int main(int argc, char* argv[]) {
    // Load logging settings from environment/command-line.
    spdlog::cfg::load_env_levels();
    dorado::utils::InitLogging();
    dorado::utils::ensure_user_locale_may_be_set();

    // Setup crash handlers.
    dorado::utils::install_segfault_handler();
    dorado::utils::install_uncaught_exception_handler();
    dorado::utils::set_stacktrace_getter(dorado::utils::torch_stacktrace);

    const std::map<std::string_view, entry_ptr> subcommands = {
            {"basecaller", &dorado::basecaller},
            {"duplex", &dorado::duplex},
            {"download", &dorado::download},
            {"aligner", &dorado::aligner},
            {"summary", &dorado::summary},
            {"demux", &dorado::demuxer},
            {"trim", &dorado::trim},
            {"correct", &dorado::correct},
            {"polish", &dorado::polish},
            {"variant", &dorado::variant_caller},
    };

    const std::vector<std::string_view> arguments(argv + 1, argv + argc);
    if (arguments.size() == 0) {
        usage(subcommands);
        return EXIT_SUCCESS;
    }

    // Log cmd
    spdlog::info("Running: \"{}\"", dorado::utils::join(arguments, "\" \""));

    const auto& subcommand = arguments[0];

    if (subcommand == "-v" || subcommand == "--version") {
        std::cout << DORADO_VERSION << '\n';
    } else if (subcommand == "-vv") {
#ifdef __APPLE__
        std::cout << "dorado:   " << DORADO_VERSION << '\n';
#else
        std::cout << "dorado:   " << DORADO_VERSION << "+cu" << CUDA_VERSION << '\n';
#endif
        std::cout << "libtorch: " << TORCH_VERSION << '\n';
        std::cout << "minimap2: " << MM_VERSION << '\n';

    } else if (subcommand == "-l" || subcommand == "--licences" || subcommand == "--licenses") {
        print_licences();
        return EXIT_SUCCESS;
    } else if (subcommand == "-h" || subcommand == "--help") {
        usage(subcommands);
        return EXIT_SUCCESS;
    } else if (subcommands.find(subcommand) != subcommands.end()) {
        return subcommands.at(subcommand)(--argc, ++argv);
    } else {
        usage(subcommands);
        return EXIT_FAILURE;
    }
}
