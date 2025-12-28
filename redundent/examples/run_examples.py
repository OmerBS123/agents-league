"""
Example runner script to demonstrate all Pydantic V2 patterns for distributed systems.

This script runs all demonstrations and provides a comprehensive overview
of the patterns implemented in the guide.
"""

import sys
import time
from typing import Dict, Any

def run_with_timing(func, description: str) -> Dict[str, Any]:
    """Run a function and time its execution."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print('='*60)

    start_time = time.time()

    try:
        func()
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
        print(f"Error: {error}")

    end_time = time.time()
    duration = end_time - start_time

    result = {
        'description': description,
        'success': success,
        'duration': duration,
        'error': error
    }

    print(f"\nCompleted in {duration:.3f}s - {'SUCCESS' if success else 'FAILED'}")
    return result

def main():
    """Run all demonstrations with timing and error handling."""
    print("Pydantic V2 Distributed Systems Comprehensive Guide")
    print("="*60)
    print("This demonstration covers:")
    print("✓ Strict typing and validation patterns")
    print("✓ Advanced validation with context awareness")
    print("✓ High-performance optimization techniques")
    print("✓ Custom serialization and error handling")
    print("✓ JSON-RPC protocol implementation")
    print("✓ Performance benchmarking and analysis")

    results = []

    # Import and run main guide demonstrations
    try:
        from distributed_system_pydantic_guide import (
            demonstrate_validation_patterns,
            demonstrate_serialization_patterns,
            demonstrate_message_protocol,
            demonstrate_performance_patterns
        )

        results.append(run_with_timing(
            demonstrate_validation_patterns,
            "Core Validation Patterns"
        ))

        results.append(run_with_timing(
            demonstrate_serialization_patterns,
            "Serialization & Custom Serializers"
        ))

        results.append(run_with_timing(
            demonstrate_message_protocol,
            "Distributed Message Protocol (JSON-RPC)"
        ))

        results.append(run_with_timing(
            demonstrate_performance_patterns,
            "Performance Optimization Patterns"
        ))

    except ImportError as e:
        print(f"Could not import main guide: {e}")
        return 1

    # Run advanced validation demonstrations
    try:
        from advanced_validation_patterns import (
            demonstrate_context_aware_validation,
            demonstrate_conditional_validation,
            demonstrate_cross_model_validation,
            demonstrate_external_dependencies,
            demonstrate_business_rules
        )

        results.append(run_with_timing(
            demonstrate_context_aware_validation,
            "Context-Aware Validation"
        ))

        results.append(run_with_timing(
            demonstrate_conditional_validation,
            "Conditional Validation Logic"
        ))

        results.append(run_with_timing(
            demonstrate_cross_model_validation,
            "Cross-Model Reference Validation"
        ))

        results.append(run_with_timing(
            demonstrate_external_dependencies,
            "External Service Dependencies"
        ))

        results.append(run_with_timing(
            demonstrate_business_rules,
            "Business Rules & Compliance"
        ))

    except ImportError as e:
        print(f"Could not import advanced validation patterns: {e}")
        print("Skipping advanced validation demonstrations...")

    # Run performance demonstrations
    try:
        from performance_optimization_patterns import (
            demonstrate_high_throughput_processing,
            demonstrate_caching,
            PerformanceBenchmark
        )

        results.append(run_with_timing(
            demonstrate_high_throughput_processing,
            "High-Throughput Batch Processing"
        ))

        results.append(run_with_timing(
            demonstrate_caching,
            "Model Caching Performance"
        ))

        results.append(run_with_timing(
            lambda: PerformanceBenchmark.benchmark_model_types(1000),
            "Model Type Performance Comparison"
        ))

        results.append(run_with_timing(
            lambda: PerformanceBenchmark.benchmark_validation_methods(1000),
            "Validation Method Benchmarks"
        ))

        results.append(run_with_timing(
            lambda: PerformanceBenchmark.benchmark_serialization_methods(1000),
            "Serialization Method Benchmarks"
        ))

    except ImportError as e:
        print(f"Could not import performance optimization patterns: {e}")
        print("Skipping performance demonstrations...")

    # Print summary
    print(f"\n{'='*60}")
    print("DEMONSTRATION SUMMARY")
    print('='*60)

    total_duration = sum(r['duration'] for r in results)
    successful_runs = sum(1 for r in results if r['success'])
    failed_runs = len(results) - successful_runs

    print(f"Total demonstrations: {len(results)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Total execution time: {total_duration:.3f}s")
    print(f"Average time per demo: {total_duration/len(results):.3f}s")

    if failed_runs > 0:
        print(f"\nFailed demonstrations:")
        for result in results:
            if not result['success']:
                print(f"  - {result['description']}: {result['error']}")

    print(f"\n{'='*60}")
    print("KEY PATTERNS DEMONSTRATED")
    print('='*60)
    print("✓ BaseModel with ConfigDict optimization")
    print("✓ Field validators (before, after, wrap, plain)")
    print("✓ Model validators with cross-field validation")
    print("✓ Context-aware and conditional validation")
    print("✓ Custom serializers (field and model level)")
    print("✓ Comprehensive error handling with custom messages")
    print("✓ High-performance configurations and caching")
    print("✓ Batch processing and parallel validation")
    print("✓ JSON-RPC protocol with polymorphic payloads")
    print("✓ Memory optimization and performance tuning")

    print(f"\n{'='*60}")
    print("DISTRIBUTED SYSTEM CONSIDERATIONS")
    print('='*60)
    print("• Thread-safe frozen models for concurrent access")
    print("• Strict protocol compliance with extra='forbid'")
    print("• Structured error propagation across services")
    print("• Context-aware validation for security levels")
    print("• High-throughput batch processing patterns")
    print("• Custom serialization for complex data types")
    print("• Performance optimization for low-latency messaging")
    print("• Discriminated unions for polymorphic message handling")

    print(f"\nAll demonstrations completed successfully!" if failed_runs == 0 else f"\nSome demonstrations failed. Check logs above.")

    return 0 if failed_runs == 0 else 1

if __name__ == "__main__":
    sys.exit(main())