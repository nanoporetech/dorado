#pragma once

#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace dorado {

class HtsReader;

class SummaryData {
public:
    using FieldFlags = uint32_t;
    static constexpr FieldFlags GENERAL_FIELDS = 1;
    static constexpr FieldFlags BARCODING_FIELDS = 2;
    static constexpr FieldFlags ALIGNMENT_FIELDS = 4;

    SummaryData();
    SummaryData(FieldFlags flags);

    void set_separator(char s);
    void set_fields(FieldFlags flags);

    /// This will automatically set the fields based on the contents of the file.
    void process_file(const std::string& filename, std::ostream& writer);

    /// For this method the fields must already be set.
    bool process_tree(const std::string& folder, std::ostream& writer);

private:
    char m_separator{'\t'};
    FieldFlags m_field_flags{};

    void write_header(std::ostream& writer);
    void write_rows_from_reader(HtsReader& reader,
                                std::ostream& writer,
                                const std::map<std::string, std::string>& rgst);
};

}  // namespace dorado
