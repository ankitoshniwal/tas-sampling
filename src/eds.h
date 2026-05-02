#pragma once
/**
 * eds.h — Exponentially Decaying Sketch (EDS)
 *
 * Header-only C++ implementation of the TAS core data structure.
 * Thread-safe for concurrent reads and writes via relaxed atomics.
 *
 * Usage:
 *   EDS sketch;
 *   bool promote = sketch.record_and_check("auth-gateway", 450.0f, 3.0f);
 *
 * Memory: 64KB (4 × 2048 × float32 × 2 tables)
 * Fits within 96KB L1d cache of Intel Sapphire Rapids.
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>

namespace tas {

class EDS {
    static constexpr int   ROWS = 4;
    static constexpr int   COLS = 2048;

    std::atomic<float> sum_[ROWS][COLS];
    std::atomic<float> cnt_[ROWS][COLS];

    float    gamma_;
    int64_t  decay_interval_ns_;
    std::atomic<int64_t>  last_decay_ns_;
    std::atomic<uint32_t> decaying_{0};
    std::atomic<uint64_t> spans_processed_{0};

public:
    /**
     * @param gamma          Decay factor per interval. Default: 0.95
     * @param decay_interval Wall-clock decay schedule. Default: 500ms
     *                       Rule of thumb: deployment_stabilization_time / 10
     */
    explicit EDS(float gamma = 0.95f,
                 std::chrono::milliseconds decay_interval =
                     std::chrono::milliseconds(500))
        : gamma_(gamma)
        , decay_interval_ns_(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  decay_interval).count())
        , last_decay_ns_(now_ns())
    {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) {
                sum_[i][j].store(0.0f, std::memory_order_relaxed);
                cnt_[i][j].store(0.0f, std::memory_order_relaxed);
            }
    }

    /**
     * Record a span's latency and decide whether to promote (sample) it.
     *
     * @param service   Service name or operation key
     * @param latency   Span latency in milliseconds
     * @param beta      Sensitivity multiplier. Recommended: P99/P50 ratio.
     *                  CPU-bound: 1.5 | Standard: 3.0 | Fan-out: 7.0 | DB-heavy: 10.0
     * @return          true if span should be promoted (sampled)
     */
    bool record_and_check(const std::string& service,
                          float latency,
                          float beta) noexcept {
        try_decay();
        spans_processed_.fetch_add(1, std::memory_order_relaxed);

        float min_avg = 1e18f;
        for (int i = 0; i < ROWS; i++) {
            uint32_t idx = hash(service, i);
            float s = sum_[i][idx].load(std::memory_order_relaxed) + latency;
            float c = cnt_[i][idx].load(std::memory_order_relaxed) + 1.0f;
            sum_[i][idx].store(s, std::memory_order_relaxed);
            cnt_[i][idx].store(c, std::memory_order_relaxed);
            if (c > 0.0f) {
                float avg = s / c;
                if (avg < min_avg) min_avg = avg;
            }
        }
        return latency > min_avg * beta;
    }

    /** Returns true once the sketch has processed enough spans to be reliable. */
    bool is_converged(uint64_t warmup_spans = 10000) const noexcept {
        return spans_processed_.load(std::memory_order_relaxed) >= warmup_spans;
    }

    uint64_t spans_processed() const noexcept {
        return spans_processed_.load(std::memory_order_relaxed);
    }

    /** Memory footprint in bytes (constant regardless of traffic volume). */
    static constexpr size_t memory_bytes() noexcept {
        return ROWS * COLS * sizeof(float) * 2;
    }

private:
    static int64_t now_ns() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    /**
     * Lock-free decay leader pattern.
     * One thread wins the CAS and applies decay; all others continue unimpeded.
     */
    void try_decay() noexcept {
        int64_t last = last_decay_ns_.load(std::memory_order_relaxed);
        if (now_ns() - last < decay_interval_ns_) return;

        uint32_t expected = 0;
        if (!decaying_.compare_exchange_strong(expected, 1,
                std::memory_order_acquire,
                std::memory_order_relaxed)) return;

        last_decay_ns_.store(now_ns(), std::memory_order_relaxed);
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) {
                sum_[i][j].store(
                    sum_[i][j].load(std::memory_order_relaxed) * gamma_,
                    std::memory_order_relaxed);
                cnt_[i][j].store(
                    cnt_[i][j].load(std::memory_order_relaxed) * gamma_,
                    std::memory_order_relaxed);
            }

        decaying_.store(0, std::memory_order_release);
    }

    /** FNV-1a inspired hash with row seed for independent hash functions. */
    static uint32_t hash(const std::string& key, int row) noexcept {
        uint32_t h = static_cast<uint32_t>(row * 12345);
        for (unsigned char c : key) h = h * 31u + c;
        return h % COLS;
    }
};

} // namespace tas
