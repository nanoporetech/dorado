#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <vector>

struct z_stream_s;
typedef z_stream_s z_stream;

namespace dorado::utils {

struct ZstreamDestructor {
    void operator()(z_stream* zlib_stream);
};
using ZstreamPtr = std::unique_ptr<z_stream, ZstreamDestructor>;

class GzipReader {
    const std::string m_gzip_file;
    const std::size_t m_buffer_size;
    std::vector<char> m_compressed_buffer;
    std::vector<char> m_decompressed_buffer;
    std::ifstream m_compressed_stream;
    std::string m_error_message{};
    bool m_is_valid{true};
    ZstreamPtr m_zlib_stream;
    std::size_t m_num_bytes_read{};

    void set_failure(const std::string& error_message);

    bool try_initialise_zlib_stream(z_stream& zlib_stream);

    ZstreamPtr create_zlib_stream();

    ZstreamPtr create_next_zlib_stream(const z_stream& last_zlib_stream);

    bool fetch_next_compressed_chunk();

    bool try_prepare_for_next_decompress();

public:
    GzipReader(std::string gzip_file, std::size_t buffer_size);

    bool is_valid() const;

    const std::string& error_message() const;

    std::size_t num_bytes_read() const;

    // Returns false if reached end of file or an error has occurred.
    // If returns false there will be zero bytes read.
    // N.B. It is possible there are zero bytes read even if returns true,
    // in which case should just continue calling.
    bool read_next();

    std::vector<char>& decompressed_buffer();
};

}  // namespace dorado::utils

#ifdef ZLIB_GZ_FUNCTIONS_MAY_BE_USED_WITHOUT_LINKER_ERROR
struct gzFile_s;
typedef gzFile_s* gzFile;

namespace dorado::utils {

struct gzFileDestructor {
    void operator()(gzFile file);
};
using gzFilePtr = std::unique_ptr<gzFile_s, gzFileDestructor>;

class GzipReader {
    const std::string m_gzip_file;
    const std::size_t m_buffer_size;
    std::vector<char> m_decompressed_buffer;
    gzFilePtr m_gzfile;
    std::string m_error_message{};
    bool m_is_valid{true};
    std::size_t m_num_bytes_read{};

    void set_failure(std::string error_message);

public:
    GzipReader(std::string gzip_file, std::size_t buffer_size);

    bool is_valid() const;

    const std::string& error_message() const;

    std::size_t num_bytes_read() const;

    // Returns false if reached end of file or an error has occurred.
    // If returns false there will be zero bytes read.
    // N.B. It is possible there are zero bytes read even if returns true,
    // in which case should just continue calling.
    bool read_next();

    std::vector<char>& decompressed_buffer();
};

}  // namespace dorado::utils

#endif