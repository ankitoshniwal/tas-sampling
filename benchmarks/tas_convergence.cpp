/**
 * TAS Convergence Curve Generator
 * Measures recall at every 100 spans from 0 to 50,000
 * Output: CSV for plotting
 *
 * Compile:
 *   g++ -O2 -march=native -std=c++17 -pthread -o tas_convergence tas_convergence.cpp
 * Run:
 *   ./tas_convergence > convergence.csv
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <iomanip>
#include <atomic>
#include <chrono>
#include <algorithm>

class TAS_Sketch {
    static const int ROWS = 4;
    static const int COLS = 2048;
    std::atomic<float> sum_table[ROWS][COLS];
    std::atomic<float> count_table[ROWS][COLS];
    double decay_factor = 0.95;
    std::chrono::steady_clock::time_point last_decay;
    std::atomic<bool> decaying{false};

public:
    TAS_Sketch() : last_decay(std::chrono::steady_clock::now()) {
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
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) {
                sum_table[i][j].store(
                    sum_table[i][j].load(std::memory_order_relaxed) * 0.95f,
                    std::memory_order_relaxed);
                count_table[i][j].store(
                    count_table[i][j].load(std::memory_order_relaxed) * 0.95f,
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
};

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

int main() {
    const int TOTAL_SPANS = 1000000;
    const int N_SERVICES = 100;
    const int SAMPLE_EVERY = 100; // measure recall every 100 spans
    const double BETA = 3.0;
    const double OUTLIER_RATE = 0.001;

    // Generate services
    std::vector<std::string> services;
    for (int i = 0; i < N_SERVICES; i++)
        services.push_back("svc_" + std::to_string(i));

    ZipfianDist zipf(N_SERVICES);

    // Run across 5 seeds, average results
    const int N_SEEDS = 5;
    std::vector<uint64_t> seeds = {42, 123, 777, 999, 2024};

    // Storage: for each measurement point, store (captured, total) per seed
    int n_points = TOTAL_SPANS / SAMPLE_EVERY;
    std::vector<std::vector<double>> recalls(n_points);

    for (uint64_t seed : seeds) {
        TAS_Sketch sketch;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> ud(0,1);
        std::lognormal_distribution<float> norm(3.5f, 0.3f);
        std::lognormal_distribution<float> out(6.0f, 0.3f);

        int window_captured = 0;
        int window_total_out = 0;
        int cumulative_captured = 0;
        int cumulative_total_out = 0;

        for (int i = 0; i < TOTAL_SPANS; i++) {
            bool is_out = ud(rng) < OUTLIER_RATE;
            float lat = is_out ? out(rng) : norm(rng);
            std::string svc = services[zipf.sample(ud(rng))];

            bool promoted = sketch.record_and_check(svc, lat, BETA);

            if (is_out) {
                cumulative_total_out++;
                if (promoted) cumulative_captured++;
            }

            // Record cumulative recall at each measurement point
            if ((i+1) % SAMPLE_EVERY == 0) {
                int idx = i / SAMPLE_EVERY;
                double recall = cumulative_total_out > 0 ?
                    (double)cumulative_captured / cumulative_total_out * 100.0 : 0.0;
                recalls[idx].push_back(recall);
            }
        }
    }

    // Output CSV
    std::cout << "span_count,recall_mean,recall_min,recall_max" << std::endl;
    for (int i = 0; i < n_points; i++) {
        int span_count = (i+1) * SAMPLE_EVERY;
        auto& v = recalls[i];
        double mean = 0;
        for (double r : v) mean += r;
        mean /= v.size();
        double mn = *std::min_element(v.begin(), v.end());
        double mx = *std::max_element(v.begin(), v.end());

        // Only output up to 50K for the convergence figure
        // then every 10K after that
        if (span_count <= 50000 || span_count % 10000 == 0) {
            std::cout << span_count << ","
                      << std::fixed << std::setprecision(4) << mean << ","
                      << std::fixed << std::setprecision(4) << mn << ","
                      << std::fixed << std::setprecision(4) << mx << std::endl;
        }
    }

    return 0;
}
