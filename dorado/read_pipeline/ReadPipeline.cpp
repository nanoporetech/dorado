#include "ReadPipeline.h"
#include <chrono>

using namespace std::chrono_literals;

void ReadSink::push_read(std::shared_ptr<Read>& read){
    std::unique_lock<std::mutex> push_read_cv_lock(m_push_read_cv_mutex);
    while(!m_push_read_cv.wait_for(push_read_cv_lock, 100ms, [this] {return m_reads.size() < m_max_reads;})) {}
    m_reads.push_back(read);
    m_cv.notify_one();
}