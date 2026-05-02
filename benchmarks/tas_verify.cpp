/**
 * TAS Production — Verification Suite
 *
 * Two checks before full paper rewrite:
 *   1. Multi-seed recall stability (is 99% consistent or a fluke?)
 *   2. Miss rate analysis (are missed outliers at low end of outlier distribution?)
 *
 * Compile:
 *   g++ -O2 -march=native -std=c++17 -pthread -o tas_verify tas_verify.cpp
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <numeric>
#include <algorithm>
#include <map>

// ============================================================
// Zipfian Distribution (identical to production code)
// ============================================================
class ZipfianDistribution {
    int n;
    double alpha;
    std::vector<double> cumulative;

public:
    ZipfianDistribution(int n, double alpha = 1.2) : n(n), alpha(alpha) {
        cumulative.resize(n);
        double sum = 0.0;
        for (int i = 1; i <= n; i++)
            sum += 1.0 / std::pow(i, alpha);
        double running = 0.0;
        for (int i = 1; i <= n; i++) {
            running += (1.0 / std::pow(i, alpha)) / sum;
            cumulative[i - 1] = running;
        }
    }

    int sample(double u) const {
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), u);
        return (int)std::distance(cumulative.begin(), it);
    }
};

// ============================================================
// Production TAS Sketch (identical to production code)
// ============================================================
class ProductionTAS {
    static const int ROWS = 4;
    static const int COLS = 2048;
    std::atomic<float> sum_table[ROWS][COLS];
    std::atomic<float> count_table[ROWS][COLS];
    double decay_factor;
    std::chrono::milliseconds decay_interval_ms;
    std::atomic<bool> decay_in_progress{false};
    std::chrono::steady_clock::time_point last_decay;

public:
    ProductionTAS(double gamma = 0.95,
                  std::chrono::milliseconds decay_interval = std::chrono::milliseconds(500))
        : decay_factor(gamma),
          decay_interval_ms(decay_interval),
          last_decay(std::chrono::steady_clock::now())
    {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) {
                sum_table[i][j].store(0.0f, std::memory_order_relaxed);
                count_table[i][j].store(0.0f, std::memory_order_relaxed);
            }
    }

    uint32_t hash(const std::string& key, int seed) const {
        uint32_t h = (uint32_t)seed;
        for (char c : key) h = h * 31u + (unsigned char)c;
        return h % COLS;
    }

    void try_decay() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_decay < decay_interval_ms) return;
        bool expected = false;
        if (!decay_in_progress.compare_exchange_strong(
                expected, true,
                std::memory_order_acquire,
                std::memory_order_relaxed)) return;
        last_decay = now;
        float f = (float)decay_factor;
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) {
                sum_table[i][j].store(
                    sum_table[i][j].load(std::memory_order_relaxed) * f,
                    std::memory_order_relaxed);
                count_table[i][j].store(
                    count_table[i][j].load(std::memory_order_relaxed) * f,
                    std::memory_order_relaxed);
            }
        decay_in_progress.store(false, std::memory_order_release);
    }

    bool record_and_check(const std::string& service,
                          float latency_ms,
                          double beta) {
        try_decay();
        float min_avg = 1e18f;
        for (int i = 0; i < ROWS; i++) {
            uint32_t idx = hash(service, i * 12345);
            float s = sum_table[i][idx].load(std::memory_order_relaxed) + latency_ms;
            float c = count_table[i][idx].load(std::memory_order_relaxed) + 1.0f;
            sum_table[i][idx].store(s, std::memory_order_relaxed);
            count_table[i][idx].store(c, std::memory_order_relaxed);
            float avg = (c > 0.0f) ? (s / c) : latency_ms;
            if (avg < min_avg) min_avg = avg;
        }
        return latency_ms > (min_avg * (float)beta);
    }
};

// ============================================================
// Shared workload structs
// ============================================================
struct Span {
    std::string service;
    float latency_ms;
    bool is_outlier;
};

std::vector<Span> generate_workload(int n_spans, int n_services,
                                     double outlier_rate, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    ZipfianDistribution zipf(n_services, 1.2);
    std::lognormal_distribution<float> normal_latency(3.5f, 0.3f);
    std::lognormal_distribution<float> outlier_latency(6.0f, 0.3f);

    std::vector<std::string> services;
    for (int i = 0; i < n_services; i++)
        services.push_back("svc_" + std::to_string(i));

    std::vector<Span> spans;
    spans.reserve(n_spans);
    for (int i = 0; i < n_spans; i++) {
        int svc_idx = std::min(zipf.sample(uniform(rng)), n_services - 1);
        bool is_outlier = uniform(rng) < outlier_rate;
        float latency = is_outlier ? outlier_latency(rng) : normal_latency(rng);
        latency = std::max(1.0f, latency);
        spans.push_back({services[svc_idx], latency, is_outlier});
    }
    return spans;
}

// ============================================================
// CHECK 1: Multi-seed recall stability
// Runs 10 seeds, reports mean +/- stddev
// If stddev > 1%, the 99% figure is not stable
// ============================================================
void check_recall_stability() {
    std::cout << "\n=== CHECK 1: Multi-Seed Recall Stability ===" << std::endl;
    std::cout << "Beta=3.0 | 100 Services | 1M Spans | Log-Normal + Zipfian" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    std::cout << std::setw(10) << "Seed"
              << std::setw(16) << "Recall"
              << std::setw(16) << "FPR" << std::endl;
    std::cout << std::string(45, '-') << std::endl;

    std::vector<uint64_t> seeds = {42, 123, 777, 999, 2024, 314, 1337, 8675, 19, 5050};
    std::vector<double> recalls, fprs;

    for (uint64_t seed : seeds) {
        auto spans = generate_workload(1000000, 100, 0.001, seed);

        ProductionTAS sketch;
        int outliers_captured = 0, normal_promoted = 0;
        int total_outliers = 0, total_normals = 0;

        for (auto& span : spans) {
            bool promoted = sketch.record_and_check(span.service, span.latency_ms, 3.0);
            if (span.is_outlier) {
                total_outliers++;
                if (promoted) outliers_captured++;
            } else {
                total_normals++;
                if (promoted) normal_promoted++;
            }
        }

        double recall = (double)outliers_captured / total_outliers * 100.0;
        double fpr    = (double)normal_promoted   / total_normals  * 100.0;
        recalls.push_back(recall);
        fprs.push_back(fpr);

        std::cout << std::setw(10) << seed
                  << std::setw(14) << std::fixed << std::setprecision(2) << recall << "%"
                  << std::setw(14) << std::fixed << std::setprecision(4) << fpr    << "%" 
                  << std::endl;
    }

    // Stats
    auto stats = [](std::vector<double>& v) -> std::pair<double,double> {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sq = 0;
        for (double x : v) sq += (x - mean) * (x - mean);
        return {mean, std::sqrt(sq / v.size())};
    };

    auto [recall_mean, recall_std] = stats(recalls);
    auto [fpr_mean,    fpr_std]    = stats(fprs);

    std::cout << std::string(45, '-') << std::endl;
    std::cout << std::setw(10) << "Mean"
              << std::setw(14) << std::fixed << std::setprecision(2) << recall_mean << "%"
              << std::setw(14) << std::fixed << std::setprecision(4) << fpr_mean    << "%" 
              << std::endl;
    std::cout << std::setw(10) << "Std Dev"
              << std::setw(14) << std::fixed << std::setprecision(2) << recall_std  << "%"
              << std::setw(14) << std::fixed << std::setprecision(4) << fpr_std     << "%" 
              << std::endl;
    std::cout << std::string(45, '-') << std::endl;

    if (recall_std < 1.0)
        std::cout << "VERDICT: Recall is STABLE. Report " 
                  << std::fixed << std::setprecision(1) << recall_mean 
                  << "% +/- " << std::setprecision(1) << recall_std << "% in paper." << std::endl;
    else
        std::cout << "VERDICT: Recall is UNSTABLE (stddev > 1%). Investigate before submitting." << std::endl;
}

// ============================================================
// CHECK 2: Miss rate analysis
// Are missed outliers clustered at low end of outlier 
// distribution (overlap with baseline tail)?
// ============================================================
void check_miss_distribution() {
    std::cout << "\n=== CHECK 2: Miss Rate Distribution Analysis ===" << std::endl;
    std::cout << "Identifies WHERE in the outlier distribution misses occur" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    auto spans = generate_workload(1000000, 100, 0.001, 42);
    ProductionTAS sketch;

    std::vector<float> missed_latencies;
    std::vector<float> captured_latencies;
    float baseline_p99 = 0.0f;

    // Compute baseline P99 from normal spans
    std::vector<float> normal_lats;
    for (auto& s : spans)
        if (!s.is_outlier) normal_lats.push_back(s.latency_ms);
    std::sort(normal_lats.begin(), normal_lats.end());
    if (!normal_lats.empty())
        baseline_p99 = normal_lats[(size_t)(normal_lats.size() * 0.99)];

    for (auto& span : spans) {
        bool promoted = sketch.record_and_check(span.service, span.latency_ms, 3.0);
        if (span.is_outlier) {
            if (!promoted) missed_latencies.push_back(span.latency_ms);
            else           captured_latencies.push_back(span.latency_ms);
        }
    }

    std::sort(missed_latencies.begin(), missed_latencies.end());
    std::sort(captured_latencies.begin(), captured_latencies.end());

    std::cout << "Baseline P99 latency:     " 
              << std::fixed << std::setprecision(1) << baseline_p99 << "ms" << std::endl;
    std::cout << "Total outliers:           " 
              << missed_latencies.size() + captured_latencies.size() << std::endl;
    std::cout << "Captured:                 " << captured_latencies.size() << std::endl;
    std::cout << "Missed:                   " << missed_latencies.size() << std::endl;

    if (!missed_latencies.empty()) {
        std::cout << "\nMissed outlier latency distribution:" << std::endl;
        std::cout << "  Min:    " << missed_latencies.front() << "ms" << std::endl;
        std::cout << "  P25:    " 
                  << missed_latencies[missed_latencies.size()*25/100] << "ms" << std::endl;
        std::cout << "  Median: " 
                  << missed_latencies[missed_latencies.size()*50/100] << "ms" << std::endl;
        std::cout << "  P75:    " 
                  << missed_latencies[missed_latencies.size()*75/100] << "ms" << std::endl;
        std::cout << "  Max:    " << missed_latencies.back() << "ms" << std::endl;

        // Count misses below 3x baseline mean (~105ms)
        float threshold_3x = 105.0f;
        int below_threshold = 0;
        for (float l : missed_latencies)
            if (l < threshold_3x) below_threshold++;

        std::cout << "\nMisses below 3x baseline mean (" 
                  << threshold_3x << "ms): "
                  << below_threshold << " / " << missed_latencies.size()
                  << " (" << std::fixed << std::setprecision(1)
                  << (double)below_threshold/missed_latencies.size()*100 << "%)" << std::endl;

        // Histogram of missed latencies in buckets
        std::cout << "\nMiss latency histogram:" << std::endl;
        std::vector<std::pair<std::string,int>> buckets = {
            {"<100ms  (baseline overlap)", 0},
            {"100-200ms (low outlier)",    0},
            {"200-400ms (mid outlier)",    0},
            {">400ms   (clear outlier)",   0}
        };
        for (float l : missed_latencies) {
            if      (l < 100)  buckets[0].second++;
            else if (l < 200)  buckets[1].second++;
            else if (l < 400)  buckets[2].second++;
            else               buckets[3].second++;
        }
        for (auto& b : buckets)
            std::cout << "  " << std::setw(32) << b.first 
                      << ": " << b.second << std::endl;

        // Verdict
        std::cout << "\nVERDICT: ";
        if (buckets[0].second > (int)missed_latencies.size() * 0.5)
            std::cout << "Confirmed — majority of misses are baseline-overlap outliers." << std::endl;
        else if (!missed_latencies.empty() && 
                 missed_latencies[missed_latencies.size()*50/100] < baseline_p99 * 2)
            std::cout << "Partial overlap — some misses near baseline tail, some cold-start." << std::endl;
        else
            std::cout << "Unexpected — misses are NOT at baseline overlap. Investigate sketch logic." << std::endl;
    } else {
        std::cout << "\nNo missed outliers — 100% recall on this seed." << std::endl;
    }

    // Also check if early-window cold start explains misses
    std::cout << "\n--- Cold-Start Window Analysis ---" << std::endl;
    std::cout << "Rerunning with warm-up period (first 10K spans excluded from measurement)..." 
              << std::endl;

    auto spans2 = generate_workload(1010000, 100, 0.001, 42);
    ProductionTAS sketch2;
    int warmed_captured = 0, warmed_total = 0;

    for (int i = 0; i < (int)spans2.size(); i++) {
        bool promoted = sketch2.record_and_check(
            spans2[i].service, spans2[i].latency_ms, 3.0);
        if (i >= 10000 && spans2[i].is_outlier) { // skip warm-up window
            warmed_total++;
            if (promoted) warmed_captured++;
        }
    }

    double warmed_recall = warmed_total > 0 ?
        (double)warmed_captured / warmed_total * 100.0 : 0.0;
    std::cout << "Post-warmup recall: " 
              << std::fixed << std::setprecision(2) << warmed_recall << "%" << std::endl;

    if (warmed_recall > 99.5)
        std::cout << "VERDICT: Cold-start is the dominant miss cause. "
                  << "State this in paper: 'Misses concentrated in warm-up window.'" << std::endl;
    else
        std::cout << "VERDICT: Cold-start is NOT the only cause. "
                  << "Distribution overlap also contributes." << std::endl;
}

int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << " TAS Verification Suite — Pre-Paper Final Checks       " << std::endl;
    std::cout << "========================================================" << std::endl;

    check_recall_stability();
    check_miss_distribution();

    std::cout << "\n=== VERIFICATION COMPLETE ===" << std::endl;
    std::cout << "Use results above to finalize paper claims." << std::endl;
    return 0;
}
