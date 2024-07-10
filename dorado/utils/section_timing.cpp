#include "section_timing.h"

#include <iostream>
#include <map>
#include <mutex>

namespace {
class SectionTimings {
    std::mutex m_mutex;
    std::map<std::string, std::function<std::string()>> m_report_providers;

public:
    static SectionTimings & instance() {
        static SectionTimings the_instance;
        return the_instance;
    }

    void add_report_provider(std::string section_name,
                             std::function<std::string()> report_provider) {
        std::lock_guard lock(m_mutex);
        m_report_providers[section_name] = report_provider;
    }

    void report() {
        std::lock_guard lock(m_mutex);
        for (auto report : m_report_providers) {
            std::cout << report.first << " : " << report.second() << std::endl;
        }
    }
};
}  // namespace

namespace dorado::utils::timings {

namespace details {
void add_report_provider(std::string section_name, std::function<std::string()> report_provider) {
    SectionTimings::instance().add_report_provider(section_name, report_provider);
}
}  // namespace details

void report() { SectionTimings::instance().report(); }

}  // namespace dorado::utils::timings
