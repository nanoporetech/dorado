#include "model_downloader.h"

#include <Foundation/Foundation.h>
#include <spdlog/spdlog.h>

namespace dorado::models {

bool ModelDownloader::download_foundation(const std::string& model,
                                          const ModelInfo& info,
                                          const std::filesystem::path& archive) {
    @autoreleasepool {
        spdlog::info(" - downloading {} with foundation", model);

        // Try and download it.
        NSString* path = [NSString stringWithUTF8String:(get_url(model)).c_str()];
        NSURL* url = [NSURL URLWithString:path];
        NSError* error = nil;
        NSData* data = [NSData dataWithContentsOfURL:url
                                             options:NSDataReadingUncached
                                               error:&error];
        if (data == nil) {
            spdlog::warn(" - failed to download {}: {}", model,
                         error.localizedDescription.UTF8String);
            return false;
        }

        // Validate it.
        const std::string_view span(static_cast<const char*>(data.bytes), data.length);
        if (!validate_checksum(span, info)) {
            return false;
        }

        // Save it.
        NSString* output = [NSString stringWithUTF8String:archive.string().c_str()];
        [data writeToFile:output atomically:YES];
        return true;
    }
}

}  // namespace dorado::models
