#include "CorrectionProgressTracker.h"

#include "utils/string_utils.h"
#include "utils/tty_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace {

void erase_progress_bar_line() {
    // Don't write escape codes unless it's a TTY.
    if (dorado::utils::is_fd_tty(stderr)) {
        // Erase the current line so that we remove the previous description.
#ifndef _WIN32
        // I would use indicators::erase_progress_bar_line() here, but it hardcodes stdout.
        std::cerr << "\r\033[K";
#endif
    }
}

}  // namespace

namespace dorado {

CorrectionProgressTracker::CorrectionProgressTracker() {}

CorrectionProgressTracker::~CorrectionProgressTracker() = default;

void CorrectionProgressTracker::set_description(const std::string& desc) {
    erase_progress_bar_line();
    m_progress_bar.set_option(indicators::option::PostfixText{desc});
}

void CorrectionProgressTracker::summarize() const {
    erase_progress_bar_line();

    if (m_num_reads_corrected > 0) {
        spdlog::info("Number of reads submitted for correction: {}", m_num_reads_corrected);
    }
}

void CorrectionProgressTracker::update_progress_bar(const stats::NamedStats& stats,
                                                    const stats::NamedStats& aligner_stats) {
    auto fetch_stat = [&stats, &aligner_stats](const std::string& name) {
        auto res_stats = stats.find(name);
        if (res_stats != stats.end()) {
            return res_stats->second;
        }
        auto res_aligner_stats = aligner_stats.find(name);
        if (res_aligner_stats != aligner_stats.end()) {
            return res_aligner_stats->second;
        }
        return 0.;
    };

    auto total_reads_in_input = int64_t(fetch_stat("CorrectionInferenceNode.total_reads_in_input"));
    m_num_reads_corrected = int64_t(fetch_stat("CorrectionInferenceNode.num_reads_corrected"));

    auto index_seqs = int64_t(fetch_stat("index_seqs"));
    auto num_reads_aligned = int64_t(fetch_stat("num_reads_aligned"));
    auto num_reads_to_infer = int64_t(fetch_stat("num_reads_to_infer"));

    auto current_idx = int64_t(fetch_stat("current_idx"));

    // Progress calculation is the following -
    // 1. Calculate total number of indices to process based on total # of reads and
    //    the number of reads per index
    // 2. Number of pipeline stages is one more than number of indices since the last
    //    inference stage is not overlapped.
    // 3. Progress within each pipeline stage is tracked by calculating how many reads
    //    have been aligned in the corresponding indexing stage.
    // 4. Progress of the last pipeline stage is how many of reads are left to be inferred.
    // 5. Overall progress is the sum of steps 3 and 4.
    if (total_reads_in_input > 0 && index_seqs > 0) {
        const int64_t num_indices = (total_reads_in_input / index_seqs) + 1;
        // Since there will be one last inference stage which will not overlap with alignment
        const int64_t num_pipeline_stages = num_indices + 1;

        const float contribution_of_each_stage = 1.f / num_pipeline_stages;

        // Tracks indices completely processed so far.
        float total_indices_processed =
                static_cast<float>(current_idx) * contribution_of_each_stage;

        // Tracks current index being processed.
        float fraction_current_index_processed =
                (static_cast<float>(num_reads_aligned) / total_reads_in_input) *
                contribution_of_each_stage;

        // Approximate tracking of the last stage by using the proportion of number
        // of reads corrected.
        float fraction_last_stage_processed =
                (num_reads_to_infer
                         ? (static_cast<float>(m_num_reads_corrected) / num_reads_to_infer)
                         : 0) *
                contribution_of_each_stage;

        float progress =
                100.f * (std::min(total_indices_processed + fraction_current_index_processed,
                                  1.f - contribution_of_each_stage) +
                         fraction_last_stage_processed);
        if (progress > m_last_progress_written) {
            m_last_progress_written = progress;
            internal_set_progress(progress);
        } else {
            internal_set_progress(m_last_progress_written);
        }
    } else if (num_reads_to_infer > 0) {
        set_description("Correcting");
        const float progress =
                100.f * static_cast<float>(m_num_reads_corrected) / num_reads_to_infer;
        m_last_progress_written = progress;
        internal_set_progress(progress);
    } else {
        set_description("Loading alignments");
        internal_set_progress(0.f);
    }
}

void CorrectionProgressTracker::internal_set_progress(float progress) {
    // The progress bar uses escape sequences that only TTYs understand.
    if (!utils::is_fd_tty(stderr)) {
        return;
    }

    // Sanity clamp.
    progress = std::min(progress, 100.f);

    // Draw it.
#ifdef _WIN32
    m_progress_bar.set_progress(static_cast<size_t>(progress));
#else
    m_progress_bar.set_progress(progress);
#endif
    std::cerr << "\r";
}

}  // namespace dorado
