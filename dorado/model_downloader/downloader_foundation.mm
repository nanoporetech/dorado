#include "downloader.h"

#include <Foundation/Foundation.h>
#include <spdlog/spdlog.h>

namespace dorado::model_downloader {

bool Downloader::download_foundation(const models::ModelInfo& model,
                                     const std::filesystem::path& archive) {
    @autoreleasepool {
        spdlog::info(" - downloading {} with foundation", model.name);

        // Try and download it.
        NSString* path = [NSString stringWithUTF8String:(get_url(model.name)).c_str()];
        NSURL* url = [NSURL URLWithString:path];
        NSError* error = nil;
        NSData* data = [NSData dataWithContentsOfURL:url
                                             options:NSDataReadingUncached
                                               error:&error];
        if (data == nil) {
            spdlog::warn(" - failed to download {}: {}", model.name,
                         error.localizedDescription.UTF8String);
            return false;
        }

        // Validate it.
        const std::string_view span(static_cast<const char*>(data.bytes), data.length);
        if (!validate_checksum(span, model)) {
            return false;
        }

        // Save it.
        NSString* output = [NSString stringWithUTF8String:archive.string().c_str()];
        if ([data writeToFile:output atomically:YES] == NO) {
            spdlog::error("Failed to save downloaded file to disk");
            return false;
        }

        // Check that all of it was saved.
        if (std::filesystem::file_size(archive) != data.length) {
            spdlog::error("Size mismatch between file in memory and file on disk");
            return false;
        }

        return true;
    }
}

}  // namespace dorado::model_downloader
