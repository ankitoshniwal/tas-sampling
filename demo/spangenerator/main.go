// spangenerator simulates realistic microservice traffic for TAS demo.
// Generates log-normal latency distribution with Zipfian service selection
// and injects P99 outliers at configurable rate.
package main

import (
	"context"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var services = []string{
	"auth-gateway", "user-service", "payment-service", "inventory-service",
	"recommendation-engine", "search-service", "notification-service",
	"order-service", "catalog-service", "review-service",
	"shipping-service", "analytics-service", "cache-service",
	"session-service", "config-service", "audit-service",
	"media-service", "billing-service", "fraud-detection", "api-gateway",
}

func getEnvFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func getEnvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return def
}

// logNormalLatency generates a realistic service latency in milliseconds.
// mu=3.5, sigma=0.3 gives P50~33ms, P95~54ms, P99~67ms — matches paper.
func logNormalLatency(rng *rand.Rand, isOutlier bool) float64 {
	if isOutlier {
		// Outlier: log-normal centered at ~400ms
		return math.Exp(6.0 + 0.3*rng.NormFloat64())
	}
	// Normal: log-normal centered at ~33ms
	return math.Exp(3.5 + 0.3*rng.NormFloat64())
}

// zipfianService selects a service with power-law probability.
// alpha=1.2 means auth-gateway gets ~30% of traffic.
func zipfianService(rng *rand.Rand, n int, alpha float64) int {
	// Simple rejection sampling for Zipfian
	sum := 0.0
	weights := make([]float64, n)
	for i := 1; i <= n; i++ {
		weights[i-1] = 1.0 / math.Pow(float64(i), alpha)
		sum += weights[i-1]
	}
	r := rng.Float64() * sum
	cumulative := 0.0
	for i, w := range weights {
		cumulative += w
		if r <= cumulative {
			return i
		}
	}
	return n - 1
}

func main() {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		endpoint = "http://localhost:4317"
	}
	// Strip http:// prefix for grpc dialer
	grpcEndpoint := endpoint
	if len(grpcEndpoint) > 7 && grpcEndpoint[:7] == "http://" {
		grpcEndpoint = grpcEndpoint[7:]
	}

	outlierRate  := getEnvFloat("OUTLIER_RATE", 0.001)
	numServices  := getEnvInt("NUM_SERVICES", 20)
	spansPerSec  := getEnvInt("SPANS_PER_SECOND", 1000)

	log.Printf("TAS Span Generator starting: endpoint=%s outlier_rate=%.3f services=%d rate=%d/s",
		endpoint, outlierRate, numServices, spansPerSec)

	// Set up OTel exporter
	ctx := context.Background()
	conn, err := grpc.DialContext(ctx, grpcEndpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		log.Fatalf("Failed to connect to collector: %v", err)
	}

	exp, err := otlptracegrpc.New(ctx, otlptracegrpc.WithGRPCConn(conn))
	if err != nil {
		log.Fatalf("Failed to create exporter: %v", err)
	}

	if numServices > len(services) {
		numServices = len(services)
	}
	activeServices := services[:numServices]

	// Create a tracer provider per service for realistic resource attribution
	providers := make(map[string]*sdktrace.TracerProvider)
	for _, svc := range activeServices {
		res := resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceName(svc),
		)
		tp := sdktrace.NewTracerProvider(
			sdktrace.WithBatcher(exp),
			sdktrace.WithResource(res),
			// Always-on sampler — TAS processor makes the decision
			sdktrace.WithSampler(sdktrace.AlwaysSample()),
		)
		providers[svc] = tp
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	interval := time.Duration(float64(time.Second) / float64(spansPerSec))

	totalSpans := 0
	outlierCount := 0
	ticker := time.NewTicker(interval)
	statsTimer := time.NewTicker(10 * time.Second)

	log.Printf("Generating spans every %v (target: %d/s)", interval, spansPerSec)
	log.Printf("P99 latency outliers injected at %.1f%% rate", outlierRate*100)

	for {
		select {
		case <-ticker.C:
			svcIdx := zipfianService(rng, numServices, 1.2)
			svcName := activeServices[svcIdx]
			isOutlier := rng.Float64() < outlierRate
			latencyMs := logNormalLatency(rng, isOutlier)

			tp := providers[svcName]
			tracer := tp.Tracer("spangenerator")

			spanCtx, span := tracer.Start(ctx, "handle_request")
			// Simulate the actual duration
			time.Sleep(time.Duration(latencyMs) * time.Millisecond)

			// Attach latency as attribute for TAS processor
			span.SetAttributes(
				attribute.Float64("http.server.duration", latencyMs),
				attribute.Bool("injected.outlier", isOutlier),
				attribute.String("service.name", svcName),
			)
			span.End()
			_ = spanCtx

			totalSpans++
			if isOutlier {
				outlierCount++
			}

		case <-statsTimer.C:
			log.Printf("Stats: total=%d outliers=%d (%.2f%%)",
				totalSpans, outlierCount,
				float64(outlierCount)/float64(totalSpans)*100)
		}
	}
}
