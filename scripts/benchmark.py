"""
SmartShield — System Benchmarking Script
Measures latency, memory, CPU, temperature across the full pipeline.

Usage:
    python scripts/benchmark.py
    
Output:
    benchmarks/benchmark_results.json
"""

import os
import sys
import json
import time
import csv
import platform
import subprocess

try:
    import psutil
except ImportError:
    psutil = None
    print("WARNING: psutil not installed. Some metrics will be unavailable.")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.rule_engine import RuleEngine

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def get_cpu_temperature():
    """Get CPU temperature (Raspberry Pi or Android)."""
    try:
        # Raspberry Pi
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            temp_str = result.stdout.strip().replace("temp=", "").replace("'C", "")
            return float(temp_str)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    try:
        # Linux thermal zone
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, PermissionError):
        pass
    
    return None


def benchmark_rule_engine(engine, test_messages, iterations=500):
    """Benchmark Stage 1: Rule Engine."""
    print("\n--- Stage 1: Rule Engine Benchmark ---")
    
    latencies = []
    for _ in range(iterations):
        for msg in test_messages:
            start = time.perf_counter_ns()
            engine.analyze(msg)
            elapsed_us = (time.perf_counter_ns() - start) / 1000
            latencies.append(elapsed_us)
    
    results = {
        "total_runs": len(latencies),
        "avg_latency_us": round(sum(latencies) / len(latencies), 2),
        "min_latency_us": round(min(latencies), 2),
        "max_latency_us": round(max(latencies), 2),
        "p50_latency_us": round(sorted(latencies)[len(latencies) // 2], 2),
        "p95_latency_us": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "p99_latency_us": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
    }
    
    print(f"  Iterations: {results['total_runs']}")
    print(f"  Avg: {results['avg_latency_us']:.1f} µs")
    print(f"  P50: {results['p50_latency_us']:.1f} µs")
    print(f"  P95: {results['p95_latency_us']:.1f} µs")
    print(f"  P99: {results['p99_latency_us']:.1f} µs")
    
    return results


def benchmark_memory():
    """Measure current process memory usage."""
    if psutil is None:
        return {"rss_mb": "N/A", "vms_mb": "N/A"}
    
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        "rss_mb": round(mem.rss / (1024 * 1024), 2),
        "vms_mb": round(mem.vms / (1024 * 1024), 2),
    }


def benchmark_sustained(engine, test_messages, duration_sec=60):
    """Run sustained test for memory stability and thermal monitoring."""
    print(f"\n--- Sustained Test ({duration_sec}s) ---")
    
    start_time = time.time()
    msg_count = 0
    mem_readings = []
    temp_readings = []
    
    while (time.time() - start_time) < duration_sec:
        for msg in test_messages:
            engine.analyze(msg)
            msg_count += 1
        
        # Sample memory and temp every second
        mem = benchmark_memory()
        mem_readings.append(mem["rss_mb"])
        
        temp = get_cpu_temperature()
        if temp:
            temp_readings.append(temp)
    
    elapsed = time.time() - start_time
    
    results = {
        "duration_sec": round(elapsed, 1),
        "messages_processed": msg_count,
        "throughput_msgs_per_sec": round(msg_count / elapsed, 1),
        "memory_start_mb": mem_readings[0] if mem_readings else "N/A",
        "memory_end_mb": mem_readings[-1] if mem_readings else "N/A",
        "memory_max_mb": max(mem_readings) if mem_readings else "N/A",
        "memory_leak_detected": False,
    }
    
    # Check for memory leak (>10% growth)
    if mem_readings and isinstance(mem_readings[0], (int, float)):
        growth = (mem_readings[-1] - mem_readings[0]) / max(mem_readings[0], 1)
        results["memory_leak_detected"] = growth > 0.10
        results["memory_growth_pct"] = round(growth * 100, 2)
    
    if temp_readings:
        results["temp_start_c"] = temp_readings[0]
        results["temp_end_c"] = temp_readings[-1]
        results["temp_max_c"] = max(temp_readings)
    
    print(f"  Messages: {msg_count} in {elapsed:.1f}s")
    print(f"  Throughput: {results['throughput_msgs_per_sec']} msg/s")
    print(f"  Memory: {results['memory_start_mb']} → {results['memory_end_mb']} MB")
    if results.get("memory_leak_detected"):
        print(f"  ⚠ Memory leak detected: {results['memory_growth_pct']:.1f}% growth")
    else:
        print(f"  ✓ No memory leak detected")
    
    return results


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("=" * 60)
    print("SmartShield — Full System Benchmark")
    print("=" * 60)
    
    # System info
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }
    
    if psutil:
        system_info["cpu_count"] = psutil.cpu_count()
        system_info["total_ram_gb"] = round(
            psutil.virtual_memory().total / (1024**3), 2
        )
    
    print(f"\n  Platform: {system_info['platform']}")
    print(f"  Arch: {system_info['architecture']}")
    
    temp = get_cpu_temperature()
    if temp:
        system_info["cpu_temp_c"] = temp
        print(f"  CPU Temp: {temp}°C")
    
    # Load test messages
    test_messages = []
    test_file = os.path.join(PROJECT_ROOT, "data", "test.csv")
    with open(test_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_messages.append(row["text"])
    
    print(f"  Test messages: {len(test_messages)}")
    
    # Initialize
    engine = RuleEngine()
    
    # Benchmarks
    rule_results = benchmark_rule_engine(engine, test_messages, iterations=100)
    memory_results = benchmark_memory()
    sustained_results = benchmark_sustained(engine, test_messages, duration_sec=30)
    
    # Model sizes
    model_sizes = {}
    models_dir = os.path.join(PROJECT_ROOT, "models")
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            path = os.path.join(models_dir, f)
            if os.path.isfile(path):
                model_sizes[f] = round(os.path.getsize(path) / (1024 * 1024), 2)
    
    rules_size = os.path.getsize(
        os.path.join(PROJECT_ROOT, "rules", "scam_patterns.json")
    ) / 1024
    model_sizes["scam_patterns.json"] = f"{rules_size:.1f} KB"
    
    # Compile results
    all_results = {
        "system_info": system_info,
        "rule_engine_benchmark": rule_results,
        "memory_usage": memory_results,
        "sustained_test": sustained_results,
        "model_sizes": model_sizes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save
    output_path = os.path.join(PROJECT_ROOT, "benchmarks", "benchmark_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Benchmark complete! Results: {output_path}")
    print(f"{'=' * 60}")
    
    return all_results


if __name__ == "__main__":
    run_full_benchmark()
