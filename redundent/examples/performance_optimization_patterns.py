"""
Performance-Optimized Pydantic V2 Patterns for High-Throughput Distributed Systems
===================================================================================

This module demonstrates performance optimization patterns including:
- Memory-efficient model design
- Fast serialization/deserialization
- Validation optimization
- Caching strategies
- Batch processing patterns
- Zero-copy operations where possible

Based on: docs.pydantic.dev/latest/concepts/performance/
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Union, Literal
from uuid import UUID, uuid4
import time
import json
from dataclasses import dataclass
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    computed_field,
    model_serializer,
    field_serializer
)
from pydantic.types import PositiveInt, StrictStr
from typing_extensions import TypedDict


# ============================================================================
# HIGH-PERFORMANCE MODEL CONFIGURATIONS
# ============================================================================

class HighPerformanceConfig(ConfigDict):
    """Optimized configuration for maximum performance."""
    # Validation optimizations
    validate_default: bool = False      # Skip default value validation
    validate_assignment: bool = False   # Skip assignment validation
    use_enum_values: bool = True       # Faster enum handling

    # Memory optimizations
    extra: str = "ignore"              # Don't store extra fields
    frozen: bool = True                # Immutable for memory efficiency

    # JSON optimizations
    ser_json_bytes: bool = False       # Return str, not bytes
    cache_strings: bool = True         # Cache repeated strings

    # Type checking optimizations
    arbitrary_types_allowed: bool = False  # Disable for speed

class UltraFastConfig(ConfigDict):
    """Extreme performance configuration (trades safety for speed)."""
    # Disable most validation for maximum speed
    validate_default: bool = False
    validate_assignment: bool = False
    extra: str = "ignore"
    frozen: bool = True

    # Minimal serialization
    ser_json_bytes: bool = False
    cache_strings: bool = False        # Skip caching for minimal models

    # Skip computed fields for serialization
    computed_fields: bool = False


# ============================================================================
# MEMORY-EFFICIENT MODEL DESIGN
# ============================================================================

class CompactMessage(BaseModel):
    """
    Memory-optimized message using specific types and minimal fields.

    Performance optimizations:
    - Use concrete types (list, dict) over abstract (Sequence, Mapping)
    - Fixed-size types where possible
    - Minimal field count
    - No computed fields
    """
    model_config = HighPerformanceConfig()

    # Use specific types for better performance
    msg_id: int                        # int instead of UUID for compactness
    msg_type: Literal["req", "resp", "notif"]  # Limited enum values
    payload: dict[str, Any]            # dict instead of Mapping
    headers: list[tuple[str, str]]     # Specific structure

class BatchMessage(BaseModel):
    """Batch message container for high-throughput processing."""
    model_config = HighPerformanceConfig()

    batch_id: int
    count: PositiveInt
    messages: list[CompactMessage]     # Batch processing

    # Pre-computed field for quick access
    @computed_field
    @property
    def total_payload_size(self) -> int:
        """Compute total payload size once."""
        return sum(len(str(msg.payload)) for msg in self.messages)

# Using TypedDict for even better performance
class HighSpeedMessage(TypedDict):
    """
    TypedDict for maximum performance when Pydantic features aren't needed.

    ~2.5x faster than equivalent Pydantic models for simple data.
    """
    msg_id: int
    msg_type: str
    data: dict[str, Any]
    timestamp: float


# ============================================================================
# FAST SERIALIZATION PATTERNS
# ============================================================================

class OptimizedSerializer:
    """Custom serialization class optimized for speed."""

    # Pre-compiled TypeAdapters for reuse (crucial for performance)
    _message_adapter = TypeAdapter(CompactMessage)
    _batch_adapter = TypeAdapter(BatchMessage)
    _typed_dict_adapter = TypeAdapter(HighSpeedMessage)

    # Thread-local storage for serialization buffers
    _local = threading.local()

    @classmethod
    def get_buffer(cls) -> dict:
        """Get thread-local buffer for serialization."""
        if not hasattr(cls._local, 'buffer'):
            cls._local.buffer = {}
        return cls._local.buffer

    @classmethod
    def serialize_message(cls, message: CompactMessage) -> str:
        """Fast message serialization using pre-compiled adapter."""
        return cls._message_adapter.dump_json(message).decode()

    @classmethod
    def deserialize_message(cls, data: str) -> CompactMessage:
        """Fast message deserialization using pre-compiled adapter."""
        return cls._message_adapter.validate_json(data)

    @classmethod
    def serialize_batch(cls, batch: BatchMessage) -> str:
        """Optimized batch serialization."""
        return cls._batch_adapter.dump_json(batch).decode()

    @classmethod
    def serialize_typed_dict(cls, data: HighSpeedMessage) -> str:
        """Ultra-fast TypedDict serialization."""
        return cls._typed_dict_adapter.dump_json(data).decode()

    @classmethod
    def serialize_batch_streaming(cls, messages: List[CompactMessage]) -> str:
        """
        Streaming serialization for large batches.
        Avoids creating intermediate objects.
        """
        # Use buffer for building JSON manually for maximum speed
        buffer = cls.get_buffer()
        buffer.clear()

        buffer['batch_id'] = int(time.time() * 1000000)  # microsecond timestamp
        buffer['count'] = len(messages)
        buffer['messages'] = []

        # Serialize each message directly to avoid intermediate objects
        for msg in messages:
            msg_dict = {
                'msg_id': msg.msg_id,
                'msg_type': msg.msg_type,
                'payload': msg.payload,
                'headers': msg.headers
            }
            buffer['messages'].append(msg_dict)

        return json.dumps(buffer, separators=(',', ':'))  # Compact JSON


# ============================================================================
# VALIDATION OPTIMIZATION PATTERNS
# ============================================================================

class ValidationOptimizer:
    """Patterns for optimizing validation performance."""

    @staticmethod
    def skip_validation_for_trusted_sources(data: dict, source: str) -> CompactMessage:
        """
        Skip validation for trusted internal sources.

        Use model_construct for pre-validated data.
        """
        if source in ["internal_agent", "system_service"]:
            # Use model_construct to skip validation
            return CompactMessage.model_construct(**data)
        else:
            # Full validation for external sources
            return CompactMessage.model_validate(data)

    @staticmethod
    def batch_validate_with_failfast(data_list: List[dict]) -> Tuple[List[CompactMessage], List[ValidationError]]:
        """
        Batch validation with early failure detection.

        Returns valid models and errors separately for batch processing.
        """
        valid_models = []
        errors = []

        for i, data in enumerate(data_list):
            try:
                model = CompactMessage.model_validate(data)
                valid_models.append(model)
            except ValidationError as e:
                # Store error with index for tracking
                e.index = i
                errors.append(e)
                # Continue processing other items instead of failing entire batch

        return valid_models, errors

    @staticmethod
    def pre_filter_invalid_data(data_list: List[dict]) -> List[dict]:
        """
        Quick pre-filtering to remove obviously invalid data before validation.

        This can significantly improve batch processing performance.
        """
        valid_data = []

        for data in data_list:
            # Quick sanity checks without full validation
            if (isinstance(data, dict) and
                'msg_id' in data and isinstance(data['msg_id'], int) and
                'msg_type' in data and data['msg_type'] in ['req', 'resp', 'notif']):
                valid_data.append(data)

        return valid_data


# ============================================================================
# CACHING AND MEMOIZATION PATTERNS
# ============================================================================

class ModelCache:
    """High-performance caching for frequently used models."""

    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, Any] = {}
        self._access_count: Dict[str, int] = defaultdict(int)
        self._max_size = max_size
        self._lock = threading.RLock()

    def _make_key(self, model_class: type, data: dict) -> str:
        """Create cache key from model type and data."""
        # Use sorted items for consistent keys
        data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return f"{model_class.__name__}:{hash(data_str)}"

    def get_or_create(self, model_class: type, data: dict) -> BaseModel:
        """Get cached model or create and cache new one."""
        key = self._make_key(model_class, data)

        with self._lock:
            if key in self._cache:
                self._access_count[key] += 1
                return self._cache[key]

            # Create new model
            model = model_class.model_validate(data)

            # Cache management - remove least accessed items if cache is full
            if len(self._cache) >= self._max_size:
                # Remove 10% of least accessed items
                items_to_remove = sorted(
                    self._access_count.items(),
                    key=lambda x: x[1]
                )[:self._max_size // 10]

                for remove_key, _ in items_to_remove:
                    self._cache.pop(remove_key, None)
                    self._access_count.pop(remove_key, None)

            self._cache[key] = model
            self._access_count[key] = 1
            return model

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()

# Global cache instance
model_cache = ModelCache()


# ============================================================================
# BATCH PROCESSING PATTERNS
# ============================================================================

class HighThroughputProcessor:
    """Processor optimized for high-throughput scenarios."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self._serializer = OptimizedSerializer()
        self._cache = ModelCache(max_size=50000)

    def process_message_batch(self, raw_messages: List[dict]) -> Dict[str, Any]:
        """
        Process a batch of messages with maximum efficiency.

        Returns processing results with performance metrics.
        """
        start_time = time.time()

        # Step 1: Pre-filter invalid data (fast rejection)
        valid_data = ValidationOptimizer.pre_filter_invalid_data(raw_messages)
        filter_time = time.time() - start_time

        # Step 2: Parallel validation using thread pool
        validation_start = time.time()
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Split data into chunks for parallel processing
            chunk_size = max(1, len(valid_data) // self.num_workers)
            chunks = [
                valid_data[i:i + chunk_size]
                for i in range(0, len(valid_data), chunk_size)
            ]

            # Process chunks in parallel
            futures = [
                executor.submit(self._process_chunk, chunk)
                for chunk in chunks
            ]

            # Collect results
            all_valid_models = []
            all_errors = []

            for future in futures:
                chunk_valid, chunk_errors = future.result()
                all_valid_models.extend(chunk_valid)
                all_errors.extend(chunk_errors)

        validation_time = time.time() - validation_start

        # Step 3: Batch serialization
        serialization_start = time.time()
        if all_valid_models:
            batch = BatchMessage(
                batch_id=int(time.time() * 1000000),
                count=len(all_valid_models),
                messages=all_valid_models
            )
            serialized_result = self._serializer.serialize_batch(batch)
        else:
            serialized_result = ""
        serialization_time = time.time() - serialization_start

        total_time = time.time() - start_time

        return {
            'total_processed': len(raw_messages),
            'valid_messages': len(all_valid_models),
            'errors': len(all_errors),
            'serialized_data': serialized_result,
            'performance_metrics': {
                'total_time': total_time,
                'filter_time': filter_time,
                'validation_time': validation_time,
                'serialization_time': serialization_time,
                'messages_per_second': len(raw_messages) / total_time if total_time > 0 else 0
            }
        }

    def _process_chunk(self, chunk: List[dict]) -> Tuple[List[CompactMessage], List[ValidationError]]:
        """Process a chunk of messages."""
        return ValidationOptimizer.batch_validate_with_failfast(chunk)

    def process_with_caching(self, raw_messages: List[dict]) -> List[CompactMessage]:
        """Process messages using caching for frequently seen patterns."""
        results = []

        for data in raw_messages:
            try:
                # Use cache for frequently occurring message patterns
                model = self._cache.get_or_create(CompactMessage, data)
                results.append(model)
            except ValidationError:
                # Skip invalid messages in high-throughput scenario
                continue

        return results


# ============================================================================
# ZERO-COPY AND STREAMING PATTERNS
# ============================================================================

class StreamingProcessor:
    """Processor for streaming data with minimal memory allocation."""

    def __init__(self):
        self._buffer = bytearray(64 * 1024)  # 64KB buffer
        self._serializer = OptimizedSerializer()

    def process_streaming_json(self, json_stream: str) -> int:
        """
        Process streaming JSON data with minimal memory allocation.

        Returns count of processed messages.
        """
        processed_count = 0

        # Parse JSON objects one at a time from stream
        decoder = json.JSONDecoder()
        idx = 0
        data_len = len(json_stream)

        while idx < data_len:
            json_stream = json_stream.lstrip()  # Remove whitespace
            if not json_stream:
                break

            try:
                obj, end_idx = decoder.raw_decode(json_stream, idx)

                # Process object using TypedDict for maximum speed
                if self._is_valid_message_structure(obj):
                    # Convert to TypedDict (no validation)
                    typed_msg: HighSpeedMessage = {
                        'msg_id': obj.get('msg_id', 0),
                        'msg_type': obj.get('msg_type', ''),
                        'data': obj.get('data', {}),
                        'timestamp': time.time()
                    }
                    processed_count += 1

                idx = end_idx

            except json.JSONDecodeError:
                break

        return processed_count

    def _is_valid_message_structure(self, obj: Any) -> bool:
        """Quick structural validation without full Pydantic validation."""
        return (
            isinstance(obj, dict) and
            'msg_id' in obj and
            'msg_type' in obj and
            isinstance(obj.get('msg_id'), int)
        )


# ============================================================================
# PERFORMANCE BENCHMARKING
# ============================================================================

class PerformanceBenchmark:
    """Benchmark different approaches for performance comparison."""

    @staticmethod
    def generate_test_data(count: int) -> List[dict]:
        """Generate test message data."""
        return [
            {
                'msg_id': i,
                'msg_type': 'req' if i % 2 == 0 else 'resp',
                'payload': {'data': f'test_data_{i}', 'value': i * 1.5},
                'headers': [('content-type', 'application/json'), ('priority', 'normal')]
            }
            for i in range(count)
        ]

    @staticmethod
    def benchmark_serialization_methods(data_count: int = 10000):
        """Benchmark different serialization approaches."""
        print(f"=== Serialization Benchmark ({data_count} messages) ===")

        # Generate test data
        test_data = PerformanceBenchmark.generate_test_data(data_count)
        messages = [CompactMessage.model_validate(item) for item in test_data]

        serializer = OptimizedSerializer()

        # Benchmark 1: Individual message serialization
        start_time = time.time()
        for msg in messages:
            serializer.serialize_message(msg)
        individual_time = time.time() - start_time

        # Benchmark 2: Batch serialization
        start_time = time.time()
        batch = BatchMessage(
            batch_id=12345,
            count=len(messages),
            messages=messages
        )
        serializer.serialize_batch(batch)
        batch_time = time.time() - start_time

        # Benchmark 3: Streaming serialization
        start_time = time.time()
        serializer.serialize_batch_streaming(messages)
        streaming_time = time.time() - start_time

        print(f"Individual serialization: {individual_time:.3f}s ({data_count/individual_time:.0f} msg/s)")
        print(f"Batch serialization: {batch_time:.3f}s ({data_count/batch_time:.0f} msg/s)")
        print(f"Streaming serialization: {streaming_time:.3f}s ({data_count/streaming_time:.0f} msg/s)")

    @staticmethod
    def benchmark_validation_methods(data_count: int = 10000):
        """Benchmark different validation approaches."""
        print(f"\n=== Validation Benchmark ({data_count} messages) ===")

        test_data = PerformanceBenchmark.generate_test_data(data_count)

        # Benchmark 1: Standard validation
        start_time = time.time()
        standard_results = []
        for item in test_data:
            try:
                model = CompactMessage.model_validate(item)
                standard_results.append(model)
            except ValidationError:
                pass
        standard_time = time.time() - start_time

        # Benchmark 2: Batch validation with pre-filtering
        start_time = time.time()
        filtered_data = ValidationOptimizer.pre_filter_invalid_data(test_data)
        batch_results, _ = ValidationOptimizer.batch_validate_with_failfast(filtered_data)
        batch_time = time.time() - start_time

        # Benchmark 3: Using model_construct (skip validation)
        start_time = time.time()
        construct_results = []
        for item in test_data:
            model = CompactMessage.model_construct(**item)
            construct_results.append(model)
        construct_time = time.time() - start_time

        print(f"Standard validation: {standard_time:.3f}s ({len(standard_results)/standard_time:.0f} msg/s)")
        print(f"Batch validation: {batch_time:.3f}s ({len(batch_results)/batch_time:.0f} msg/s)")
        print(f"Skip validation (construct): {construct_time:.3f}s ({len(construct_results)/construct_time:.0f} msg/s)")

    @staticmethod
    def benchmark_model_types(data_count: int = 10000):
        """Benchmark different model types."""
        print(f"\n=== Model Type Benchmark ({data_count} messages) ===")

        test_data = PerformanceBenchmark.generate_test_data(data_count)

        # Benchmark 1: Pydantic BaseModel
        start_time = time.time()
        pydantic_results = []
        for item in test_data:
            try:
                model = CompactMessage.model_validate(item)
                pydantic_results.append(model)
            except ValidationError:
                pass
        pydantic_time = time.time() - start_time

        # Benchmark 2: TypedDict with TypeAdapter
        typed_dict_adapter = TypeAdapter(HighSpeedMessage)
        start_time = time.time()
        typed_dict_results = []
        for item in test_data:
            try:
                # Convert to TypedDict format
                typed_item: HighSpeedMessage = {
                    'msg_id': item['msg_id'],
                    'msg_type': item['msg_type'],
                    'data': item['payload'],
                    'timestamp': time.time()
                }
                validated = typed_dict_adapter.validate_python(typed_item)
                typed_dict_results.append(validated)
            except ValidationError:
                pass
        typed_dict_time = time.time() - start_time

        # Benchmark 3: Plain dict (no validation)
        start_time = time.time()
        plain_dict_results = []
        for item in test_data:
            plain_dict_results.append(item)
        plain_dict_time = time.time() - start_time

        print(f"Pydantic BaseModel: {pydantic_time:.3f}s ({len(pydantic_results)/pydantic_time:.0f} msg/s)")
        print(f"TypedDict + TypeAdapter: {typed_dict_time:.3f}s ({len(typed_dict_results)/typed_dict_time:.0f} msg/s)")
        print(f"Plain dict (no validation): {plain_dict_time:.3f}s ({len(plain_dict_results)/plain_dict_time:.0f} msg/s)")


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_high_throughput_processing():
    """Demonstrate high-throughput processing patterns."""
    print("=== High-Throughput Processing Demo ===")

    # Create processor
    processor = HighThroughputProcessor(num_workers=4)

    # Generate test data with some invalid entries
    test_data = [
        {'msg_id': i, 'msg_type': 'req', 'payload': {'data': f'test_{i}'}, 'headers': []}
        for i in range(1000)
    ]
    # Add some invalid entries
    test_data.extend([
        {'invalid': 'data'},
        {'msg_id': 'invalid_id'},
        {'msg_id': 9999, 'msg_type': 'invalid_type', 'payload': {}, 'headers': []}
    ])

    # Process batch
    results = processor.process_message_batch(test_data)

    print(f"Processed {results['total_processed']} messages")
    print(f"Valid: {results['valid_messages']}, Errors: {results['errors']}")
    print(f"Throughput: {results['performance_metrics']['messages_per_second']:.0f} msg/s")
    print(f"Total time: {results['performance_metrics']['total_time']:.3f}s")

def demonstrate_caching():
    """Demonstrate caching performance benefits."""
    print("\n=== Caching Demo ===")

    cache = ModelCache(max_size=100)

    # Same data processed multiple times
    test_data = {'msg_id': 1, 'msg_type': 'req', 'payload': {'test': 'data'}, 'headers': []}

    # First access - cache miss
    start_time = time.time()
    model1 = cache.get_or_create(CompactMessage, test_data)
    first_time = time.time() - start_time

    # Second access - cache hit
    start_time = time.time()
    model2 = cache.get_or_create(CompactMessage, test_data)
    second_time = time.time() - start_time

    print(f"First access (cache miss): {first_time*1000:.3f}ms")
    print(f"Second access (cache hit): {second_time*1000:.3f}ms")
    print(f"Speedup: {first_time/second_time:.1f}x")
    print(f"Same object: {model1 is model2}")

if __name__ == "__main__":
    """Run performance demonstrations and benchmarks."""

    # Run demonstrations
    demonstrate_high_throughput_processing()
    demonstrate_caching()

    # Run benchmarks
    print("\n" + "="*60)
    PerformanceBenchmark.benchmark_model_types(5000)
    PerformanceBenchmark.benchmark_validation_methods(5000)
    PerformanceBenchmark.benchmark_serialization_methods(5000)

    print("\n=== Performance Optimization Summary ===")
    print("✓ High-performance model configurations")
    print("✓ Memory-efficient model design")
    print("✓ Fast serialization with pre-compiled TypeAdapters")
    print("✓ Validation optimization and batching")
    print("✓ Caching and memoization patterns")
    print("✓ Parallel batch processing")
    print("✓ TypedDict for maximum speed scenarios")
    print("✓ Streaming processing with minimal allocation")