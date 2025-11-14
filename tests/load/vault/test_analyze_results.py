"""Unit tests for analyze_results.py load test analyzer.

Tests the LoadTestAnalyzer class and its methods for calculating statistics,
assessing performance, and generating reports from k6 JSON output.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import mock_open, patch
from analyze_results import LoadTestAnalyzer


class TestLoadTestAnalyzer:
    """Test suite for LoadTestAnalyzer class."""

    @pytest.fixture
    def sample_metrics(self):
        """Sample k6 metrics for testing."""
        return [
            {
                "type": "Point",
                "metric": "http_reqs",
                "data": {"time": "2025-11-14T10:00:00Z", "value": 1},
            },
            {
                "type": "Point",
                "metric": "http_req_duration",
                "data": {"time": "2025-11-14T10:00:01Z", "value": 10.5},
            },
            {
                "type": "Point",
                "metric": "http_req_duration",
                "data": {"time": "2025-11-14T10:00:02Z", "value": 20.0},
            },
            {
                "type": "Point",
                "metric": "http_req_duration",
                "data": {"time": "2025-11-14T10:00:03Z", "value": 15.5},
            },
            {
                "type": "Point",
                "metric": "http_req_failed",
                "data": {"time": "2025-11-14T10:00:04Z", "value": 0},
            },
        ]

    @pytest.fixture
    def sample_json_file(self, tmp_path, sample_metrics):
        """Create a temporary JSON file with sample metrics."""
        json_file = tmp_path / "test_results.json"
        with open(json_file, "w") as f:
            for metric in sample_metrics:
                f.write(json.dumps(metric) + "\n")
        return json_file

    def test_load_results(self, sample_json_file):
        """Test loading metrics from JSON file."""
        analyzer = LoadTestAnalyzer(sample_json_file)
        assert len(analyzer.metrics) == 5
        assert analyzer.metrics[0]["metric"] == "http_reqs"

    def test_load_results_invalid_json(self, tmp_path):
        """Test handling of invalid JSON lines."""
        json_file = tmp_path / "invalid.json"
        with open(json_file, "w") as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')

        analyzer = LoadTestAnalyzer(json_file)
        # Should skip invalid line
        assert len(analyzer.metrics) == 2

    def test_calculate_statistics_basic(self, sample_json_file):
        """Test basic statistics calculation."""
        analyzer = LoadTestAnalyzer(sample_json_file)
        stats = analyzer.calculate_statistics()

        assert stats["total_requests"] == 1
        assert stats["total_errors"] == 0
        assert stats["error_rate"] == 0.0
        assert "latency" in stats
        assert "throughput_rps" in stats

    def test_calculate_statistics_latency_percentiles(
        self, sample_json_file
    ):
        """Test percentile calculations for latency."""
        analyzer = LoadTestAnalyzer(sample_json_file)
        stats = analyzer.calculate_statistics()

        latency = stats["latency"]
        # We have durations: [10.5, 15.5, 20.0]
        assert latency["min"] == 10.5
        assert latency["max"] == 20.0
        assert latency["avg"] == pytest.approx(15.33, rel=0.1)
        # P50 should be around median (15.5)
        assert 15 < latency["p50"] < 16

    def test_calculate_statistics_with_errors(self, tmp_path):
        """Test statistics calculation with errors present."""
        metrics = [
            {
                "type": "Point",
                "metric": "http_reqs",
                "data": {"value": 1},
            },
            {
                "type": "Point",
                "metric": "http_reqs",
                "data": {"value": 1},
            },
            {
                "type": "Point",
                "metric": "http_req_failed",
                "data": {"value": 1},
            },  # Error
            {
                "type": "Point",
                "metric": "http_req_failed",
                "data": {"value": 0},
            },
        ]

        json_file = tmp_path / "with_errors.json"
        with open(json_file, "w") as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")

        analyzer = LoadTestAnalyzer(json_file)
        stats = analyzer.calculate_statistics()

        assert stats["total_requests"] == 2
        assert stats["total_errors"] == 1
        assert stats["error_rate"] == 50.0

    def test_calculate_duration_valid_timestamps(self, tmp_path):
        """Test duration calculation with valid timestamps."""
        metrics = [
            {
                "type": "Point",
                "data": {"time": "2025-11-14T10:00:00Z"},
            },
            {
                "type": "Point",
                "data": {"time": "2025-11-14T10:05:00Z"},
            },
        ]

        json_file = tmp_path / "duration_test.json"
        with open(json_file, "w") as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")

        analyzer = LoadTestAnalyzer(json_file)
        duration = analyzer._calculate_duration()

        # 5 minutes = 300 seconds
        assert duration == 300.0

    def test_calculate_duration_timezone_aware(self, tmp_path):
        """Test duration calculation with timezone-aware timestamps."""
        # This tests the bug fix for timezone parsing
        metrics = [
            {
                "type": "Point",
                "data": {"time": "2025-11-14T10:00:00+01:00"},
            },
            {
                "type": "Point",
                "data": {"time": "2025-11-14T10:05:00+01:00"},
            },
        ]

        json_file = tmp_path / "tz_test.json"
        with open(json_file, "w") as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")

        analyzer = LoadTestAnalyzer(json_file)
        duration = analyzer._calculate_duration()

        # Should still be 5 minutes
        assert duration == 300.0

    def test_calculate_duration_no_timestamps(self, tmp_path):
        """Test duration calculation with no timestamps."""
        metrics = [{"type": "Point", "data": {}}]

        json_file = tmp_path / "no_time.json"
        with open(json_file, "w") as f:
            f.write(json.dumps(metrics[0]) + "\n")

        analyzer = LoadTestAnalyzer(json_file)
        duration = analyzer._calculate_duration()

        assert duration == 0

    def test_assess_performance_all_pass(self, sample_json_file):
        """Test performance assessment when all targets met."""
        analyzer = LoadTestAnalyzer(sample_json_file)
        stats = {
            "latency": {"p95": 50, "p99": 150},
            "error_rate": 0.5,
            "throughput_rps": 200,
        }

        assessment = analyzer.assess_performance(stats)

        assert "PASS" in assessment["p95_latency"]
        assert "PASS" in assessment["p99_latency"]
        assert "PASS" in assessment["error_rate"]
        assert "PASS" in assessment["throughput"]

    def test_assess_performance_all_fail(self, sample_json_file):
        """Test performance assessment when all targets failed."""
        analyzer = LoadTestAnalyzer(sample_json_file)
        stats = {
            "latency": {"p95": 150, "p99": 300},
            "error_rate": 2.0,
            "throughput_rps": 50,
        }

        assessment = analyzer.assess_performance(stats)

        assert "FAIL" in assessment["p95_latency"]
        assert "FAIL" in assessment["p99_latency"]
        assert "FAIL" in assessment["error_rate"]
        assert "FAIL" in assessment["throughput"]

    def test_generate_report_structure(self, sample_json_file):
        """Test that generated report contains all required sections."""
        analyzer = LoadTestAnalyzer(sample_json_file)
        report = analyzer.generate_report()

        # Check for required sections
        assert "# Vault Load Test Results" in report
        assert "## Summary Statistics" in report
        assert "## Latency Metrics" in report
        assert "## Performance Assessment" in report
        assert "## Recommendations" in report

    def test_generate_report_with_warnings(self, tmp_path):
        """Test report generation includes warnings for issues."""
        # Create metrics with high error rate
        metrics = [
            {
                "type": "Point",
                "metric": "http_reqs",
                "data": {"time": "2025-11-14T10:00:00Z", "value": 1},
            },
            {
                "type": "Point",
                "metric": "http_req_duration",
                "data": {"time": "2025-11-14T10:00:01Z", "value": 150},
            },
            {
                "type": "Point",
                "metric": "http_req_failed",
                "data": {"value": 1},
            },
        ]

        json_file = tmp_path / "warnings.json"
        with open(json_file, "w") as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")

        analyzer = LoadTestAnalyzer(json_file)
        report = analyzer.generate_report()

        # Should include warning about high error rate
        assert "High error rate detected" in report
        # Should include warning about high latency
        assert "High P95 latency" in report

    def test_generate_report_all_targets_met(self, sample_json_file):
        """Test report when all performance targets are met."""
        analyzer = LoadTestAnalyzer(sample_json_file)
        # Mock assess_performance to return all PASS
        with patch.object(
            analyzer,
            "assess_performance",
            return_value={
                "p95_latency": "✓ PASS",
                "p99_latency": "✓ PASS",
                "error_rate": "✓ PASS",
                "throughput": "✓ PASS",
            },
        ):
            report = analyzer.generate_report()
            assert "All performance targets met" in report

    def test_percentile_edge_cases(self):
        """Test percentile calculation edge cases."""
        analyzer = LoadTestAnalyzer.__new__(LoadTestAnalyzer)
        analyzer.metrics = []

        # Empty list
        stats = analyzer.calculate_statistics()
        assert stats["latency"]["p50"] == 0
        assert stats["latency"]["p95"] == 0

    def test_empty_metrics_file(self, tmp_path):
        """Test handling of empty metrics file."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("")

        analyzer = LoadTestAnalyzer(json_file)
        stats = analyzer.calculate_statistics()

        assert stats["total_requests"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["throughput_rps"] == 0.0


class TestMain:
    """Test main function and CLI interface."""

    def test_main_single_file(self, tmp_path, capsys, monkeypatch):
        """Test main with single result file."""
        # Create test file
        json_file = tmp_path / "test.json"
        metrics = [
            {
                "type": "Point",
                "metric": "http_reqs",
                "data": {"time": "2025-11-14T10:00:00Z", "value": 1},
            },
            {
                "type": "Point",
                "metric": "http_req_duration",
                "data": {"time": "2025-11-14T10:00:01Z", "value": 10},
            },
        ]
        with open(json_file, "w") as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")

        # Mock sys.argv
        monkeypatch.setattr(
            "sys.argv", ["analyze_results.py", str(json_file)]
        )

        # Run main
        from analyze_results import main

        main()

        # Check output
        captured = capsys.readouterr()
        assert "Vault Load Test Results" in captured.out

    def test_main_output_file(self, tmp_path, monkeypatch):
        """Test main with output file specified."""
        # Create test file
        json_file = tmp_path / "test.json"
        output_file = tmp_path / "report.md"
        metrics = [
            {
                "type": "Point",
                "metric": "http_reqs",
                "data": {"value": 1},
            }
        ]
        with open(json_file, "w") as f:
            f.write(json.dumps(metrics[0]) + "\n")

        # Mock sys.argv
        monkeypatch.setattr(
            "sys.argv",
            ["analyze_results.py", str(json_file), "-o", str(output_file)],
        )

        # Run main
        from analyze_results import main

        main()

        # Check output file was created
        assert output_file.exists()
        content = output_file.read_text()
        assert "Vault Load Test Results" in content
