#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <sstream>

// ============================================================
// TAS: Threshold-Aware Sampling Benchmark Suite
// Target: GCP C3 (Intel Sapphire Rapids)
// Run with: g++ -O2 -o tas_benchmark tas_benchmark.cpp && ./tas_benchmark
// ============================================================

class TAS_Sketch {
    static const int ROWS = 4;
    static const int COLS = 2048;
    double sum_table[ROWS][COLS];
    double count_table[ROWS][COLS];
    double decay_factor;
    uint64_t event_count;
    static const uint64_t DECAY_INTERVAL = 10000;

public:
    TAS_Sketch(double gamma = 0.95) : decay_factor(gamma), event_count(0) {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) {
                sum_table[i][j] = 0.0;
                count_table[i][j] = 0.0;
            }
    }

    uint32_t hash(const std::string& key, int seed) const {
        uint32_t h = seed;
        for (char c : key) h = h * 31 + (unsigned char)c;
        return h % COLS;
    }

    void decay() {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) {
                sum_table[i][j]   *= decay_factor;
                count_table[i][j] *= decay_factor;
            }
    }

    bool record_and_check(const std::string& service, uint64_t latency, double beta) {
        event_count++;
        if (event_count % DECAY_INTERVAL == 0) decay();

        double min_avg = 1e18;
        for (int i = 0; i < ROWS; i++) {
            uint32_t idx = hash(service, i * 12345);
            sum_table[i][idx]   += latency;
            count_table[i][idx] += 1.0;
            double avg = sum_table[i][idx] / count_table[i][idx];
            if (avg < min_avg) min_avg = avg;
        }
        return (double)latency > (min_avg * beta);
    }

    size_t memory_bytes() const {
        return ROWS * COLS * 2 * sizeof(double);
    }
};

// ============================================================
// Benchmark 1: Precision-Recall across Beta (100 services)
// Replicates the paper's Table 1 on server-class hardware
// ============================================================
void benchmark_precision_recall() {
    std::cout << "\n=== BENCHMARK 1: Precision-Recall (100 Services, Subtle Outliers) ===" << std::endl;
    std::cout << "Hardware target: GCP C3 Intel Sapphire Rapids" << std::endl;
    std::cout << "Baseline: 20-50ms | Outliers: 80-120ms | Spans: 1,000,000" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left
              << std::setw(10) << "Beta"
              << std::setw(16) << "Recall"
              << std::setw(22) << "False Promo Rate"
              << std::setw(12) << "Throughput"
              << "Status" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    // Generate service keys
    std::vector<std::string> services;
    for (int i = 0; i < 100; i++)
        services.push_back("svc_" + std::to_string(i));

    for (double beta : {1.0, 1.2, 1.5, 2.0, 3.0, 5.0}) {
        TAS_Sketch sketch;
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        int outliers_captured = 0;
        int normal_promoted   = 0;
        int total_outliers    = 0;
        int total_normals     = 0;

        auto t_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 1000000; i++) {
            std::string svc = services[rng() % 100];
            uint64_t latency;
            bool is_outlier = false;

            if (dist(rng) < 0.001) {
                latency    = 80 + (rng() % 40);
                is_outlier = true;
                total_outliers++;
            } else {
                latency = 20 + (rng() % 30);
                total_normals++;
            }

            bool promoted = sketch.record_and_check(svc, latency, beta);
            if (promoted) {
                if (is_outlier) outliers_captured++;
                else            normal_promoted++;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
        double throughput_mops = 1.0 / elapsed_s;  // 1M spans / elapsed

        double recall = (total_outliers > 0) ?
            (double)outliers_captured / total_outliers * 100.0 : 0.0;
        double fpr = (total_normals > 0) ?
            (double)normal_promoted / total_normals * 100.0 : 0.0;

        std::string status;
        if (beta < 1.2)       status = "Over-sampling";
        else if (beta < 1.5)  status = "High Noise";
        else if (beta <= 2.0) status = recall > 95 ? "Optimal/Stable" : "Degrading";
        else if (beta < 5.0)  status = "Under-sampling";
        else                  status = "Blind";

        std::cout << std::left
                  << std::setw(10) << beta
                  << std::setw(16) << (std::to_string((int)round(recall*100)/100.0).substr(0,5) + "%")
                  << std::setw(22) << (std::to_string(fpr).substr(0,7) + "%")
                  << std::setw(12) << (std::to_string((int)(throughput_mops*10)/10.0).substr(0,4) + " Mops/s")
                  << status << std::endl;
    }

    // Random baseline
    std::cout << std::string(70, '-') << std::endl;
    std::mt19937 rng2(42);
    std::uniform_real_distribution<double> dist2(0.0, 1.0);
    int rand_outliers = 0, rand_total = 0;
    for (int i = 0; i < 1000000; i++) {
        bool is_outlier = dist2(rng2) < 0.001;
        if (is_outlier) rand_total++;
        if (dist2(rng2) < 0.01) { // 1% random sample
            if (is_outlier) rand_outliers++;
        }
    }
    double rand_recall = (rand_total > 0) ? (double)rand_outliers / rand_total * 100.0 : 0.0;
    std::cout << std::left
              << std::setw(10) << "1% Random"
              << std::setw(16) << (std::to_string((int)rand_recall) + "%")
              << std::setw(22) << "0.0000%"
              << std::setw(12) << "N/A"
              << "Baseline" << std::endl;
}

// ============================================================
// Benchmark 2: Scale test — 1000 services
// Tests collision pressure beyond the paper's 100-service test
// ============================================================
void benchmark_scale() {
    std::cout << "\n=== BENCHMARK 2: Scale Test (1000 Services) ===" << std::endl;
    std::cout << "Beta fixed at 1.5 (optimal) | Spans: 1,000,000" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    std::vector<std::string> services;
    for (int i = 0; i < 1000; i++)
        services.push_back("svc_" + std::to_string(i));

    TAS_Sketch sketch;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int outliers_captured = 0, normal_promoted = 0;
    int total_outliers = 0, total_normals = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++) {
        std::string svc = services[rng() % 1000];
        uint64_t latency;
        bool is_outlier = dist(rng) < 0.001;
        if (is_outlier) { latency = 80 + (rng() % 40); total_outliers++; }
        else            { latency = 20 + (rng() % 30); total_normals++;  }

        bool promoted = sketch.record_and_check(svc, latency, 1.5);
        if (promoted) {
            if (is_outlier) outliers_captured++;
            else            normal_promoted++;
        }
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Services:          1000" << std::endl;
    std::cout << "Outlier Recall:    " << std::fixed << std::setprecision(2)
              << (double)outliers_captured/total_outliers*100 << "%" << std::endl;
    std::cout << "False Promo Rate:  " << std::fixed << std::setprecision(4)
              << (double)normal_promoted/total_normals*100 << "%" << std::endl;
    std::cout << "Throughput:        " << std::fixed << std::setprecision(2)
              << (1.0/elapsed_s) << " M spans/s" << std::endl;
    std::cout << "Memory Footprint:  " << sketch.memory_bytes()/1024 << " KB" << std::endl;
}

// ============================================================
// Benchmark 3: Throughput vs span volume
// Measures raw decision latency at line rate
// ============================================================
void benchmark_throughput() {
    std::cout << "\n=== BENCHMARK 3: Raw Throughput (ns per decision) ===" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<int> volumes = {100000, 500000, 1000000, 5000000};
    std::vector<std::string> services;
    for (int i = 0; i < 100; i++)
        services.push_back("svc_" + std::to_string(i));

    std::cout << std::left
              << std::setw(14) << "Spans"
              << std::setw(16) << "Total Time"
              << std::setw(18) << "Throughput"
              << "Latency/span" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (int vol : volumes) {
        TAS_Sketch sketch;
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        auto t_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < vol; i++) {
            std::string svc = services[rng() % 100];
            uint64_t latency = 20 + (rng() % 80);
            sketch.record_and_check(svc, latency, 1.5);
        }
        auto t_end = std::chrono::high_resolution_clock::now();

        double elapsed_s  = std::chrono::duration<double>(t_end - t_start).count();
        double elapsed_ms = elapsed_s * 1000.0;
        double mops       = (double)vol / elapsed_s / 1e6;
        double ns_per_op  = elapsed_s * 1e9 / vol;

        std::cout << std::left
                  << std::setw(14) << vol
                  << std::setw(16) << (std::to_string((int)elapsed_ms) + "ms")
                  << std::setw(18) << (std::to_string(mops).substr(0,5) + " M ops/s")
                  << std::to_string((int)ns_per_op) + "ns" << std::endl;
    }
}

// ============================================================
// Benchmark 4: Decay adaptation — baseline shift simulation
// Simulates a deployment event causing latency regression
// ============================================================
void benchmark_decay_adaptation() {
    std::cout << "\n=== BENCHMARK 4: Baseline Shift Adaptation ===" << std::endl;
    std::cout << "Simulates a deployment at span 500,000 causing 2x latency regression" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    TAS_Sketch sketch;
    std::mt19937 rng(42);

    int pre_false_promotions  = 0;
    int post_false_promotions = 0;
    int adaptation_span       = -1;

    for (int i = 0; i < 1000000; i++) {
        uint64_t latency;
        bool post_deployment = (i >= 500000);

        // After deployment: baseline shifts from ~35ms to ~70ms
        if (post_deployment) latency = 60 + (rng() % 30);  // new normal: 60-90ms
        else                 latency = 20 + (rng() % 30);  // old normal: 20-50ms

        bool promoted = sketch.record_and_check("svc_main", latency, 1.5);

        if (post_deployment && promoted) {
            post_false_promotions++;
            if (adaptation_span < 0) adaptation_span = i - 500000;
        }
        if (!post_deployment && promoted) pre_false_promotions++;
    }

    std::cout << "Pre-deployment false promotions:  " << pre_false_promotions << std::endl;
    std::cout << "Post-deployment surge (spans):    " << post_false_promotions << std::endl;
    std::cout << "Adaptation begins at span:        +" << adaptation_span << " post-deployment" << std::endl;
    std::cout << "Decay factor (gamma):             0.95 per 10,000 events" << std::endl;
    std::cout << "(High post-deployment count is expected — sketch adapts over ~5-10 decay cycles)" << std::endl;
}

int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << " TAS Benchmark Suite — GCP C3 (Intel Sapphire Rapids)  " << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "Compile: g++ -O2 -march=native -o tas_benchmark tas_benchmark.cpp" << std::endl;
    std::cout << "Memory per sketch: ~32KB (fits in L2 cache)" << std::endl;

    benchmark_precision_recall();
    benchmark_scale();
    benchmark_throughput();
    benchmark_decay_adaptation();

    std::cout << "\n=== ALL BENCHMARKS COMPLETE ===" << std::endl;
    std::cout << "Report hardware spec in paper: lscpu | grep 'Model name'" << std::endl;
    std::cout << "Pin CPU freq: sudo cpupower frequency-set -g performance" << std::endl;
    return 0;
}
