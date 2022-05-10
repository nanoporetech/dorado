#define CPPHTTPLIB_OPENSSL_SUPPORT

#include <iostream>
#include <argparse.hpp>

#include "httplib.h"
#include "Version.h"
#include "../models.h"

int download(int argc, char *argv[]) {

    argparse::ArgumentParser parser("dorado", DORADO_VERSION);

    parser.add_argument("--list")
            .default_value(false)
            .implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    auto list = parser.get<bool>("--list");

    if (list) {
        std::cerr << "> basecaller models" << std::endl;
        for (const auto& [model, _] : basecaller::models) {
	    std::cerr << " - " << model << std::endl;
	}
	
	return 0;
    }

    httplib::Client http(basecaller::URL_ROOT);
    http.set_follow_location(true);

    std::cerr << "> basecaller models" << std::endl;
    for (const auto& [model, path] : basecaller::models) {
      
      std::cerr << " - downloading " << model << " " << path;
      auto res = http.Get(path.c_str());
      std::cout << " [" << res->status << "]" << std::endl;
      return 0;

    }

    return 0;

}


