# TAS: Threshold-Aware Sampling

[![Paper](https://img.shields.io/badge/arXiv-TAS-red)](https://arxiv.org/abs/PENDING)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![OTel](https://img.shields.io/badge/OTel%20Collector-0.96-orange)](https://opentelemetry.io)

An edge-native distributed tracing sampler that replaces value-blind random sampling
with intelligent, latency-aware trace promotion using a 64KB L1-resident
Exponentially Decaying Sketch (EDS).

## The Problem

Standard 1% random sampling means a critical 1000ms latency spike has only a **1%
chance of being captured**. When a P99 incident occurs, you need the traces — not
a lottery.

## Key Results

Benchmarked on **GCP C3 (Intel Xeon Platinum 8481C, Sapphire Rapids, 96KB L1d)**:

| Metric | Value |
|--------|-------|
| Decision latency | 63ns/span |
| Raw sketch lookup | 13ns/span |
| Throughput (single thread) | 15.67 M spans/s |
| Memory footprint | 64KB (L1-resident) |
| Outlier recall (β=3.0, 10 seeds) | 99.96% ± 0.05% |
| False promotion rate | 0.0062% |
| Memory vs OTel tail-sampling (1M RPS) | 31,250x smaller |

## How It Works

TAS maintains a **64KB Exponentially Decaying Sketch** per node. For each span:

1. Record the span's latency into the sketch for its service key
2. Retrieve the current baseline estimate (conservative minimum average)
3. If `latency > baseline × β` → **promote** (sample the trace)
4. Otherwise → drop

The sketch decays on a wall-clock schedule (γ=0.95, every 500ms), autonomously
adapting to baseline shifts after deployments without manual intervention.

## β Calibration

A key finding: optimal β depends on the service's latency distribution shape.

| Service Type | P99/P50 Ratio | Recommended β |
|-------------|---------------|---------------|
| CPU-bound | ~1.3x | 1.5 |
| Standard | ~2.0x | 3.0 |
| Fan-out | ~10x | 7.0 |
| DB-heavy | ~20x | 10.0 |

**Heuristic: β ≈ P99/P50 ratio of your service.**

## Repository Structure

```
tas-sampling/
├── src/
│   ├── eds.h                    # Core EDS sketch (header-only C++)
│   ├── eds_corelocal.h          # Core-Local EDS for high-contention deployments
│   └── eds_hierarchical.h       # Hierarchical EDS with cold-start mitigation
├── benchmarks/
│   ├── tas_benchmark.cpp        # Main throughput + precision-recall benchmark
│   ├── tas_reviewer.cpp         # Reviewer response experiment suite (6 experiments)
│   ├── tas_hotkey.cpp           # Hot-key cache line bouncing stress test
│   ├── tas_coldstart.cpp        # Cold-start ghost mitigation test
│   ├── tas_convergence.cpp      # Convergence curve data generator
│   └── tas_verify.cpp           # Multi-seed recall stability verification
├── paper/
│   └── TAS.pdf                  # Full paper (5 pages)
└── demo/
    ├── docker-compose.yaml      # OTel Collector + Jaeger + span generator
    ├── config.yaml              # Collector config with tas_sampler processor
    └── spangenerator/
        ├── main.go              # Realistic log-normal + Zipfian traffic generator
        └── Dockerfile
```

## Quick Start

```bash
git clone https://github.com/ankitoshniwal/tas-sampling
cd tas-sampling/demo
docker compose up
```

Open **http://localhost:16686** (Jaeger UI). Set Min Duration to `100ms` to see
only promoted outlier traces.

## Replicating Paper Results

All benchmarks target GCP C3 (Intel Sapphire Rapids). Setup:

```bash
# Provision GCP C3 instance
gcloud compute instances create tas-benchmark \
  --machine-type=c3-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --zone=us-central1-a

# Install dependencies
sudo apt-get install -y g++ build-essential

# Run main benchmark
g++ -O2 -march=native -std=c++17 -pthread -o tas_benchmark benchmarks/tas_benchmark.cpp
./tas_benchmark | tee results.txt
```

## OTel Collector Integration

TAS is implemented as a custom OTel Collector processor (`tas_sampler`).
See `demo/config.yaml` for configuration reference.

```yaml
processors:
  tas_sampler:
    beta: 3.0                      # P99/P50 ratio of your service
    decay_factor: 0.95             # Baseline forgetting rate
    decay_interval: 500ms          # Wall-clock decay schedule
    warmup_spans: 10000            # Cold-start window
    cold_start_sampling_rate: 0.01 # Fallback during warmup (matches head-sampling)
    latency_attribute: "http.server.duration"
```

## Architecture

```
Span arrives
     │
     ▼
Cold-start? ──YES──► 1% random sampling (fallback)
     │
     │ NO (post-convergence)
     ▼
EDS Sketch (64KB, L1-resident)
RecordAndCheck(service, latency_ms, β)
     │
latency > baseline × β?
     │
YES ─┤─ NO
     │      │
  PROMOTE  DROP
```

For high-contention deployments (>4 threads or hot-key services):
use **Core-Local EDS** — per-thread sketches merged at decay,
eliminating cache line contention at the cost of L2 vs L1 residency.

## Citation

```bibtex
@article{toshniwal2026tas,
  title={TAS: Adaptive Trace Sampling using L1-Resident Exponentially Decaying Sketches},
  author={Toshniwal, Ankit},
  journal={arXiv preprint},
  year={2026}
}
```

## License

Apache 2.0
