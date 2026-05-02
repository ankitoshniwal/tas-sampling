#pragma once
/**
 * eds_hierarchical.h — Hierarchical EDS with Cold-Start Mitigation
 *
 * Extends EDS with a cluster-wide static baseline fallback.
 * During the warmup window, uses the cluster baseline for threshold
 * decisions instead of the (unreliable) local sketch estimate.
 *
 * Use when:
 *   - Services have tight latency SLOs requiring outlier capture from span 0
 *   - Subtle outliers (2-5x baseline) must be captured within the first
 *     second of a pod restart
 *
 * Baseline staleness tolerance (empirically verified):
 *   ±50% of true P50 → no recall degradation, no FPR inflation
 *   3x too high      → recall drops to ~94.7%
 *   3x too low       → FPR inflates to ~1%
 *
 * In practice, a cluster baseline updated hourly via config or gossip
 * is sufficient for correct cold-start behavior.
 */

#include "eds.h"
#include <unordered_map>

namespace tas {

class HierarchicalEDS : public EDS {
    std::unordered_map<std::string, float> cluster_baseline_;
    uint64_t warmup_threshold_;

public:
    /**
     * @param warmup_threshold  Spans before switching to local baseline.
     *                          Formula: max(500, convergence_seconds × expected_rps)
     * @param gamma             Decay factor. Default: 0.95
     * @param decay_interval    Decay schedule. Default: 500ms
     */
    explicit HierarchicalEDS(
        uint64_t warmup_threshold = 10000,
        float gamma = 0.95f,
        std::chrono::milliseconds decay_interval =
            std::chrono::milliseconds(500))
        : EDS(gamma, decay_interval)
        , warmup_threshold_(warmup_threshold)
    {}

    /**
     * Register a cluster-wide baseline for a service.
     * Call at pod startup with values from config service or gossip protocol.
     *
     * @param service      Service name
     * @param baseline_ms  Cluster P50 latency in milliseconds
     */
    void set_cluster_baseline(const std::string& service,
                               float baseline_ms) {
        cluster_baseline_[service] = baseline_ms;
    }

    /**
     * Record and check with hierarchical baseline fallback.
     *
     * Pre-convergence:  uses cluster_baseline[service] × beta as threshold
     * Post-convergence: uses local EDS baseline × beta as threshold
     */
    bool record_and_check(const std::string& service,
                          float latency,
                          float beta) noexcept {
        // Always record into sketch to build baseline
        // Use parent implementation but intercept the decision
        bool local_decision = EDS::record_and_check(service, latency, beta);

        // If converged, trust local decision
        if (is_converged(warmup_threshold_)) return local_decision;

        // Pre-convergence: use cluster baseline if available
        auto it = cluster_baseline_.find(service);
        if (it != cluster_baseline_.end()) {
            return latency > it->second * beta;
        }

        // No cluster baseline available — fall back to local decision
        return local_decision;
    }
};

} // namespace tas
