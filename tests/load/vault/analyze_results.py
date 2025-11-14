#!/usr/bin/env python3
"""
Vault Load Test Results Analyzer

Analyzes k6 JSON output and generates performance reports.

Usage:
    python analyze_results.py results/vault_read_20251113.json
    python analyze_results.py results/*.json --output report.md
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dateutil import parser as dateutil_parser
import argparse


class LoadTestAnalyzer:
    """Analyzes k6 load test results and generates reports."""

    def __init__(self, result_file: Path):
        self.result_file = result_file
        self.metrics = self._load_results()

    def _load_results(self) -> List[Dict[str, Any]]:
        """Load k6 JSON results file."""
        metrics = []
        with open(self.result_file, "r") as f:
            for line in f:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return metrics

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate key statistics from test results."""
        http_reqs = [
            m
            for m in self.metrics
            if m.get("type") == "Point" and m.get("metric") == "http_reqs"
        ]
        http_durations = [
            m
            for m in self.metrics
            if m.get("type") == "Point"
            and m.get("metric") == "http_req_duration"
        ]
        errors = [
            m
            for m in self.metrics
            if m.get("type") == "Point"
            and m.get("metric") == "http_req_failed"
        ]

        total_requests = len(http_reqs)
        total_errors = sum(
            1 for e in errors if e.get("data", {}).get("value", 0) > 0
        )

        # Calculate latency percentiles
        durations = [m.get("data", {}).get("value", 0) for m in http_durations]
        durations.sort()

        def percentile(data, p):
            if not data:
                return 0
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (data[c] - data[f]) * (k - f)

        stats = {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (
                (total_errors / total_requests * 100)
                if total_requests > 0
                else 0
            ),
            "latency": {
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "avg": sum(durations) / len(durations) if durations else 0,
                "p50": percentile(durations, 0.50),
                "p95": percentile(durations, 0.95),
                "p99": percentile(durations, 0.99),
            },
            "duration_seconds": self._calculate_duration(),
            "throughput_rps": (
                total_requests / self._calculate_duration()
                if self._calculate_duration() > 0
                else 0
            ),
        }

        return stats

    def _calculate_duration(self) -> float:
        """Calculate test duration in seconds."""
        timestamps = [
            m.get("data", {}).get("time")
            for m in self.metrics
            if m.get("data", {}).get("time")
        ]
        if not timestamps:
            return 0
        # Use dateutil.parser to handle various ISO formats including
        # timezone-aware timestamps
        timestamps = [dateutil_parser.isoparse(t) for t in timestamps if t]
        if len(timestamps) < 2:
            return 0
        return (max(timestamps) - min(timestamps)).total_seconds()

    def assess_performance(self, stats: Dict[str, Any]) -> Dict[str, str]:
        """Assess performance against targets."""
        assessment = {}

        # Latency assessment
        p95_target = 100  # ms
        p99_target = 200  # ms

        if stats["latency"]["p95"] < p95_target:
            assessment["p95_latency"] = (
                f"✓ PASS ({stats['latency']['p95']:.2f}ms < {p95_target}ms)"
            )
        else:
            assessment["p95_latency"] = (
                f"✗ FAIL ({stats['latency']['p95']:.2f}ms > {p95_target}ms)"
            )

        if stats["latency"]["p99"] < p99_target:
            assessment["p99_latency"] = (
                f"✓ PASS ({stats['latency']['p99']:.2f}ms < {p99_target}ms)"
            )
        else:
            assessment["p99_latency"] = (
                f"✗ FAIL ({stats['latency']['p99']:.2f}ms > {p99_target}ms)"
            )

        # Error rate assessment
        error_rate_target = 1.0  # percent
        if stats["error_rate"] < error_rate_target:
            assessment["error_rate"] = (
                f"✓ PASS ({stats['error_rate']:.2f}% < {error_rate_target}%)"
            )
        else:
            assessment["error_rate"] = (
                f"✗ FAIL ({stats['error_rate']:.2f}% > {error_rate_target}%)"
            )

        # Throughput assessment (vary by test type)
        throughput_target = 100  # RPS minimum
        if stats["throughput_rps"] > throughput_target:
            rps = stats["throughput_rps"]
            assessment["throughput"] = (
                f"✓ PASS ({rps:.2f} RPS > {throughput_target} RPS)"
            )
        else:
            rps = stats["throughput_rps"]
            assessment["throughput"] = (
                f"✗ FAIL ({rps:.2f} RPS < {throughput_target} RPS)"
            )

        return assessment

    def generate_report(self) -> str:
        """Generate markdown report."""
        stats = self.calculate_statistics()
        assessment = self.assess_performance(stats)

        report = f"""# Vault Load Test Results

**Test File**: {self.result_file.name}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Requests | {stats['total_requests']:,} |
| Total Errors | {stats['total_errors']:,} |
| Error Rate | {stats['error_rate']:.2f}% |
| Duration | {stats['duration_seconds']:.2f}s |
| Throughput | {stats['throughput_rps']:.2f} RPS |

## Latency Metrics

| Percentile | Value |
|------------|-------|
| Min | {stats['latency']['min']:.2f}ms |
| Average | {stats['latency']['avg']:.2f}ms |
| P50 (Median) | {stats['latency']['p50']:.2f}ms |
| P95 | {stats['latency']['p95']:.2f}ms |
| P99 | {stats['latency']['p99']:.2f}ms |
| Max | {stats['latency']['max']:.2f}ms |

## Performance Assessment

| Test | Result |
|------|--------|
| P95 Latency | {assessment['p95_latency']} |
| P99 Latency | {assessment['p99_latency']} |
| Error Rate | {assessment['error_rate']} |
| Throughput | {assessment['throughput']} |

## Recommendations

"""
        # Add recommendations based on results
        if stats["error_rate"] > 1.0:
            report += (
                "- ⚠️  **High error rate detected** - "
                "Investigate Vault logs for root cause\n"
            )

        if stats["latency"]["p95"] > 100:
            report += (
                "- ⚠️  **High P95 latency** - "
                "Consider Vault performance tuning or scaling\n"
            )

        if stats["throughput_rps"] < 100:
            report += (
                "- ⚠️  **Low throughput** - "
                "Review Vault backend performance (Consul/etcd)\n"
            )

        if all("PASS" in v for v in assessment.values()):
            report += (
                "- ✓ **All performance targets met** - "
                "Vault is performing within acceptable limits\n"
            )

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Vault load test results"
    )
    parser.add_argument(
        "files", nargs="+", help="JSON result files from k6"
    )
    parser.add_argument(
        "--output", "-o", help="Output markdown file", default=None
    )

    args = parser.parse_args()

    reports = []
    for file_path in args.files:
        result_file = Path(file_path)
        if not result_file.exists():
            print(f"Error: File not found: {result_file}", file=sys.stderr)
            continue

        print(f"Analyzing {result_file}...")
        analyzer = LoadTestAnalyzer(result_file)
        report = analyzer.generate_report()
        reports.append(report)

    # Combine reports
    full_report = "\n\n---\n\n".join(reports)

    # Output report
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(full_report)
        print(f"Report saved to {output_path}")
    else:
        print(full_report)


if __name__ == "__main__":
    main()
