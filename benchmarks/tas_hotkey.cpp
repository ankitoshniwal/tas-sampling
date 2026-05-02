/**
 * TAS Hot Key Cache Line Bouncing Stress Test
 *
 * Tests throughput and latency under extreme contention:
 *   1. Baseline: 8 threads, 100 services (Zipfian) — normal operation
 *   2. Hot key: 8 threads, ALL hammering same single service key
 *   3. Core-local: 8 threads with per-thread sketches, merged at decay
 *
 * Measures:
 *   - Throughput (Mops/s)
 *   - Per-operation latency (ns)
 *   - Latency percentiles (P50, P99, P999)
 *   - Recall and FPR (accuracy under contention)
 *
 * Compile:
 *   g++ -O2 -march=native -std=c++17 -pthread -o tas_hotkey tas_hotkey.cpp
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <thread>
#include <numeric>
#include <algorithm>
#include <map>
#include <cstring>

// ============================================================
// Shared EDS Sketch (original — shared across threads)
// ============================================================
class SharedEDS {
    static const int ROWS = 4;
    static const int COLS = 2048;
    std::atomic<float> sum_table[ROWS][COLS];
    std::atomic<float> count_table[ROWS][COLS];
    double decay_factor = 0.95;
    std::chrono::steady_clock::time_point last_decay;
    std::atomic<bool> decaying{false};

public:
    SharedEDS() : last_decay(std::chrono::steady_clock::now()) {
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
        if (now - last_decay < std::chrono::milliseconds(500)) return;
        bool expected = false;
        if (!decaying.compare_exchange_strong(expected, true,
                std::memory_order_acquire, std::memory_order_relaxed)) return;
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
        decaying.store(false, std::memory_order_release);
    }

    bool record_and_check(const std::string& service, float latency, double beta) {
        try_decay();
        float min_avg = 1e18f;
        for (int i = 0; i < ROWS; i++) {
            uint32_t idx = hash(service, i * 12345);
            float s = sum_table[i][idx].load(std::memory_order_relaxed) + latency;
            float c = count_table[i][idx].load(std::memory_order_relaxed) + 1.0f;
            sum_table[i][idx].store(s, std::memory_order_relaxed);
            count_table[i][idx].store(c, std::memory_order_relaxed);
            if (c > 0 && s/c < min_avg) min_avg = s/c;
        }
        return latency > min_avg * (float)beta;
    }

    size_t memory_bytes() const {
        return ROWS * COLS * sizeof(float) * 2;
    }
};

// ============================================================
// Core-Local EDS (per-thread sketch, merged at decay)
// Each thread writes to its own sketch — zero contention.
// Merge happens during decay by averaging across thread sketches.
// ============================================================
class CoreLocalEDS {
    static const int ROWS = 4;
    static const int COLS = 2048;
    int n_threads;

    // Per-thread storage: n_threads * ROWS * COLS * 2 arrays
    // Each thread gets its own cache-line-aligned sketch
    struct alignas(64) ThreadSketch {
        float sum[ROWS][COLS];
        float count[ROWS][COLS];
        ThreadSketch() {
            memset(sum, 0, sizeof(sum));
            memset(count, 0, sizeof(count));
        }
    };

    std::vector<ThreadSketch> thread_sketches;

    // Merged sketch for queries (updated during decay)
    float merged_sum[ROWS][COLS];
    float merged_count[ROWS][COLS];

    std::chrono::steady_clock::time_point last_decay;
    std::atomic<bool> decaying{false};
    double decay_factor = 0.95;

public:
    CoreLocalEDS(int n_threads)
        : n_threads(n_threads),
          thread_sketches(n_threads),
          last_decay(std::chrono::steady_clock::now()) {
        memset(merged_sum, 0, sizeof(merged_sum));
        memset(merged_count, 0, sizeof(merged_count));
    }

    uint32_t hash(const std::string& key, int seed) const {
        uint32_t h = (uint32_t)seed;
        for (char c : key) h = h * 31u + (unsigned char)c;
        return h % COLS;
    }

    void try_merge_and_decay() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_decay < std::chrono::milliseconds(500)) return;
        bool expected = false;
        if (!decaying.compare_exchange_strong(expected, true,
                std::memory_order_acquire, std::memory_order_relaxed)) return;
        last_decay = now;

        // Merge all thread sketches into merged arrays
        float f = (float)decay_factor;
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                float sum_total = 0, count_total = 0;
                for (int t = 0; t < n_threads; t++) {
                    sum_total   += thread_sketches[t].sum[i][j];
                    count_total += thread_sketches[t].count[i][j];
                    // Apply decay to each thread sketch
                    thread_sketches[t].sum[i][j]   *= f;
                    thread_sketches[t].count[i][j] *= f;
                }
                merged_sum[i][j]   = sum_total;
                merged_count[i][j] = count_total;
            }
        }

        decaying.store(false, std::memory_order_release);
    }

    // Hot path: thread writes to its own local sketch (no contention)
    // Reads from merged sketch for decision
    bool record_and_check(int thread_id, const std::string& service,
                          float latency, double beta) {
        try_merge_and_decay();

        // Write to thread-local sketch — zero cache line contention
        auto& ts = thread_sketches[thread_id];
        float min_avg = 1e18f;

        for (int i = 0; i < ROWS; i++) {
            uint32_t idx = hash(service, i * 12345);
            ts.sum[i][idx]   += latency;
            ts.count[i][idx] += 1.0f;

            // Use merged if populated; fall back to local pre-first-merge
            float s = (merged_count[i][idx] > 0) ? merged_sum[i][idx]   : ts.sum[i][idx];
            float c = (merged_count[i][idx] > 0) ? merged_count[i][idx] : ts.count[i][idx];
            if (c > 0 && s/c < min_avg) min_avg = s/c;
        }

        return latency > min_avg * (float)beta;
    }

    size_t memory_bytes() const {
        // Per-thread sketches + merged sketch
        return (n_threads + 1) * ROWS * COLS * sizeof(float) * 2;
    }
};

// ============================================================
// Zipfian Distribution
// ============================================================
class ZipfianDist {
    std::vector<double> cumulative;
    int n;
public:
    ZipfianDist(int n, double alpha = 1.2) : n(n) {
        cumulative.resize(n);
        double sum = 0;
        for (int i = 1; i <= n; i++) sum += 1.0/std::pow(i,alpha);
        double running = 0;
        for (int i = 1; i <= n; i++) {
            running += (1.0/std::pow(i,alpha))/sum;
            cumulative[i-1] = running;
        }
    }
    int sample(double u) const {
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), u);
        return std::min((int)std::distance(cumulative.begin(),it), n-1);
    }
};

// ============================================================
// Latency measurement helpers
// ============================================================
struct LatencyStats {
    double mean_ns;
    double p50_ns;
    double p99_ns;
    double p999_ns;
    double max_ns;
};

LatencyStats compute_stats(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    double sum = 0;
    for (double v : samples) sum += v;
    return {
        sum / samples.size(),
        samples[samples.size() * 50 / 100],
        samples[samples.size() * 99 / 100],
        samples[samples.size() * 999 / 1000],
        samples.back()
    };
}

void print_stats(const std::string& label, LatencyStats& s,
                 double mops, double recall, double fpr) {
    std::cout << std::left << std::setw(28) << label
              << std::setw(10) << std::fixed << std::setprecision(2)
                              << mops << " Mops/s"
              << "  P50:" << std::setw(8) << (int)s.p50_ns << "ns"
              << "  P99:" << std::setw(8) << (int)s.p99_ns << "ns"
              << "  P999:" << std::setw(8) << (int)s.p999_ns << "ns"
              << "  Recall:" << std::setw(8)
                             << std::fixed << std::setprecision(2) << recall << "%"
              << "  FPR:" << std::fixed << std::setprecision(4) << fpr << "%"
              << std::endl;
}

// ============================================================
// Benchmark runner
// ============================================================
struct BenchResult {
    double mops;
    LatencyStats latency;
    double recall;
    double fpr;
};

// --- Shared sketch, Zipfian traffic (baseline) ---
BenchResult bench_shared_zipfian(int n_threads, int n_ops) {
    const int N_SERVICES = 100;
    std::vector<std::string> services;
    for (int i = 0; i < N_SERVICES; i++)
        services.push_back("svc_" + std::to_string(i));
    ZipfianDist zipf(N_SERVICES);

    SharedEDS sketch;
    std::atomic<int> outliers_cap{0}, normal_promo{0};
    int total_out = 0, total_norm = 0;
    std::vector<std::vector<double>> thread_latencies(n_threads);
    for (auto& v : thread_latencies) v.reserve(n_ops / n_threads);

    int chunk = n_ops / n_threads;
    std::vector<std::thread> threads;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            std::mt19937_64 rng(42 + t);
            std::uniform_real_distribution<double> ud(0,1);
            std::lognormal_distribution<float> norm(3.5f, 0.3f);
            std::lognormal_distribution<float> out(6.0f, 0.3f);

            for (int i = 0; i < chunk; i++) {
                bool is_out = ud(rng) < 0.001;
                float lat = is_out ? out(rng) : norm(rng);
                std::string svc = services[zipf.sample(ud(rng))];

                auto s0 = std::chrono::high_resolution_clock::now();
                bool p = sketch.record_and_check(svc, lat, 3.0);
                auto s1 = std::chrono::high_resolution_clock::now();

                thread_latencies[t].push_back(
                    std::chrono::duration<double, std::nano>(s1-s0).count());

                if (is_out) { if (p) outliers_cap.fetch_add(1, std::memory_order_relaxed); }
                else        { if (p) normal_promo.fetch_add(1, std::memory_order_relaxed); }
            }
        });
    }
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < chunk * n_threads; i++) {
        if (i % 1000 < 1) total_out++; else total_norm++;
    }

    // Merge latency samples
    std::vector<double> all_lat;
    for (auto& v : thread_latencies)
        all_lat.insert(all_lat.end(), v.begin(), v.end());

    double elapsed = std::chrono::duration<double>(t1-t0).count();
    return {
        (double)n_ops / elapsed / 1e6,
        compute_stats(all_lat),
        (double)outliers_cap / std::max(1, (int)(n_ops * 0.001)) * 100.0,
        (double)normal_promo / std::max(1, (int)(n_ops * 0.999)) * 100.0
    };
}

// --- Shared sketch, HOT KEY (all threads same service) ---
BenchResult bench_shared_hotkey(int n_threads, int n_ops) {
    SharedEDS sketch;
    std::atomic<int> outliers_cap{0}, normal_promo{0};
    std::vector<std::vector<double>> thread_latencies(n_threads);
    for (auto& v : thread_latencies) v.reserve(n_ops / n_threads);

    const std::string HOT_KEY = "auth-gateway"; // single hot key

    int chunk = n_ops / n_threads;
    std::vector<std::thread> threads;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            std::mt19937_64 rng(42 + t);
            std::uniform_real_distribution<double> ud(0,1);
            std::lognormal_distribution<float> norm(3.5f, 0.3f);
            std::lognormal_distribution<float> out(6.0f, 0.3f);

            for (int i = 0; i < chunk; i++) {
                bool is_out = ud(rng) < 0.001;
                float lat = is_out ? out(rng) : norm(rng);

                auto s0 = std::chrono::high_resolution_clock::now();
                bool p = sketch.record_and_check(HOT_KEY, lat, 3.0);
                auto s1 = std::chrono::high_resolution_clock::now();

                thread_latencies[t].push_back(
                    std::chrono::duration<double, std::nano>(s1-s0).count());

                if (is_out) { if (p) outliers_cap.fetch_add(1, std::memory_order_relaxed); }
                else        { if (p) normal_promo.fetch_add(1, std::memory_order_relaxed); }
            }
        });
    }
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<double> all_lat;
    for (auto& v : thread_latencies)
        all_lat.insert(all_lat.end(), v.begin(), v.end());

    double elapsed = std::chrono::duration<double>(t1-t0).count();
    return {
        (double)n_ops / elapsed / 1e6,
        compute_stats(all_lat),
        (double)outliers_cap / std::max(1, (int)(n_ops * 0.001)) * 100.0,
        (double)normal_promo / std::max(1, (int)(n_ops * 0.999)) * 100.0
    };
}

// --- Core-local sketch, HOT KEY ---
BenchResult bench_corelocal_hotkey(int n_threads, int n_ops) {
    CoreLocalEDS sketch(n_threads);
    std::atomic<int> outliers_cap{0}, normal_promo{0};
    std::vector<std::vector<double>> thread_latencies(n_threads);
    for (auto& v : thread_latencies) v.reserve(n_ops / n_threads);

    const std::string HOT_KEY = "auth-gateway";

    int chunk = n_ops / n_threads;
    std::vector<std::thread> threads;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            std::mt19937_64 rng(42 + t);
            std::uniform_real_distribution<double> ud(0,1);
            std::lognormal_distribution<float> norm(3.5f, 0.3f);
            std::lognormal_distribution<float> out(6.0f, 0.3f);

            for (int i = 0; i < chunk; i++) {
                bool is_out = ud(rng) < 0.001;
                float lat = is_out ? out(rng) : norm(rng);

                auto s0 = std::chrono::high_resolution_clock::now();
                bool p = sketch.record_and_check(t, HOT_KEY, lat, 3.0);
                auto s1 = std::chrono::high_resolution_clock::now();

                thread_latencies[t].push_back(
                    std::chrono::duration<double, std::nano>(s1-s0).count());

                if (is_out) { if (p) outliers_cap.fetch_add(1, std::memory_order_relaxed); }
                else        { if (p) normal_promo.fetch_add(1, std::memory_order_relaxed); }
            }
        });
    }
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<double> all_lat;
    for (auto& v : thread_latencies)
        all_lat.insert(all_lat.end(), v.begin(), v.end());

    double elapsed = std::chrono::duration<double>(t1-t0).count();

    std::cout << "  CoreLocal memory: "
              << sketch.memory_bytes() / 1024 << "KB ("
              << n_threads << " thread sketches + 1 merged)" << std::endl;

    return {
        (double)n_ops / elapsed / 1e6,
        compute_stats(all_lat),
        (double)outliers_cap / std::max(1, (int)(n_ops * 0.001)) * 100.0,
        (double)normal_promo / std::max(1, (int)(n_ops * 0.999)) * 100.0
    };
}

// ============================================================
// Main
// ============================================================
int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << " TAS Hot Key Cache Line Bouncing Stress Test            " << std::endl;
    std::cout << " GCP C3 (Intel Xeon Platinum 8481C, Sapphire Rapids)   " << std::endl;
    std::cout << "========================================================" << std::endl;

    const int N_OPS    = 1000000;
    const int N_THREADS = 8;

    std::cout << "\nOps per run: " << N_OPS << " | Threads: " << N_THREADS << std::endl;

    // ---- Test 1: Baseline (Zipfian, shared sketch) ----
    std::cout << "\n--- TEST 1: Baseline (Zipfian traffic, shared sketch) ---" << std::endl;
    auto r1 = bench_shared_zipfian(N_THREADS, N_OPS);
    print_stats("Shared+Zipfian (baseline)", r1.latency, r1.mops, r1.recall, r1.fpr);

    // ---- Test 2: Hot Key (all threads same key, shared sketch) ----
    std::cout << "\n--- TEST 2: Hot Key stress (all threads → auth-gateway) ---" << std::endl;
    auto r2 = bench_shared_hotkey(N_THREADS, N_OPS);
    print_stats("Shared+HotKey", r2.latency, r2.mops, r2.recall, r2.fpr);

    // Latency degradation
    double p99_delta = r2.latency.p99_ns / r1.latency.p99_ns;
    double throughput_delta = r2.mops / r1.mops;
    std::cout << "\nHot Key impact vs baseline:" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(2)
              << r1.mops << " → " << r2.mops << " Mops/s ("
              << (int)((1.0 - throughput_delta)*100) << "% degradation)" << std::endl;
    std::cout << "  P99 latency: " << (int)r1.latency.p99_ns << "ns → "
              << (int)r2.latency.p99_ns << "ns ("
              << std::fixed << std::setprecision(1) << p99_delta << "x increase)" << std::endl;

    // ---- Test 3: Core-Local Sketch (hot key, no contention) ----
    std::cout << "\n--- TEST 3: Core-Local Sketch (hot key, zero contention) ---" << std::endl;
    auto r3 = bench_corelocal_hotkey(N_THREADS, N_OPS);
    print_stats("CoreLocal+HotKey", r3.latency, r3.mops, r3.recall, r3.fpr);

    double recovery_throughput = r3.mops / r1.mops;
    double recovery_p99 = r3.latency.p99_ns / r1.latency.p99_ns;
    std::cout << "\nCore-Local recovery vs baseline:" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(2)
              << r3.mops << " Mops/s ("
              << std::fixed << std::setprecision(0)
              << recovery_throughput * 100 << "% of baseline)" << std::endl;
    std::cout << "  P99 latency: " << (int)r3.latency.p99_ns << "ns ("
              << std::fixed << std::setprecision(1) << recovery_p99 << "x baseline)" << std::endl;

    // ---- Summary ----
    std::cout << "\n========================================================" << std::endl;
    std::cout << " SUMMARY" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << std::left
              << std::setw(30) << "Configuration"
              << std::setw(14) << "Throughput"
              << std::setw(12) << "P50"
              << std::setw(12) << "P99"
              << std::setw(12) << "P999"
              << "Recall" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    auto print_row = [](const std::string& name, BenchResult& r) {
        std::cout << std::left
                  << std::setw(30) << name
                  << std::setw(14) << (std::to_string(r.mops).substr(0,5) + " Mops/s")
                  << std::setw(12) << (std::to_string((int)r.latency.p50_ns) + "ns")
                  << std::setw(12) << (std::to_string((int)r.latency.p99_ns) + "ns")
                  << std::setw(12) << (std::to_string((int)r.latency.p999_ns) + "ns")
                  << std::fixed << std::setprecision(2) << r.recall << "%" << std::endl;
    };

    print_row("Shared + Zipfian (baseline)", r1);
    print_row("Shared + Hot Key (stress)", r2);
    print_row("Core-Local + Hot Key (fix)", r3);

    std::cout << "\nVERDICT:" << std::endl;
    if (r2.latency.p99_ns > r1.latency.p99_ns * 3) {
        std::cout << "  HOT KEY causes significant cache line bouncing." << std::endl;
        std::cout << "  Core-Local sketch is RECOMMENDED for >4 thread deployments." << std::endl;
    } else {
        std::cout << "  HOT KEY degradation is modest. Shared sketch acceptable." << std::endl;
        std::cout << "  Core-Local sketch provides additional headroom." << std::endl;
    }

    std::cout << "\nlscpu cache info for reference:" << std::endl;
    system("lscpu | grep -i cache");

    return 0;
}
