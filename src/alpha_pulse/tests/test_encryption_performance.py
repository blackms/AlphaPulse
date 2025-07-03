"""
Performance tests for encryption operations.

Run with: pytest -v -m performance
"""
import pytest
import time
import random
import string
import statistics
from typing import List, Tuple
import psutil
import os

from alpha_pulse.utils.encryption import (
    AESCipher,
    PerformanceOptimizedCipher,
    get_cipher,
    get_optimized_cipher
)
from alpha_pulse.models.encrypted_fields import EncryptedString, EncryptedJSON
from alpha_pulse.config.database import get_db_config

# Mark all tests as performance tests
pytestmark = pytest.mark.performance


class TestEncryptionPerformance:
    """Performance benchmarks for encryption operations."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for performance tests."""
        self.sample_sizes = [10, 100, 1000, 10000]
        self.data_sizes = [100, 1000, 10000]  # bytes
        
    def generate_random_string(self, length: int) -> str:
        """Generate random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def measure_memory(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_single_encryption_performance(self):
        """Benchmark single encryption operations."""
        cipher = get_cipher()
        results = []
        
        print("\n=== Single Encryption Performance ===")
        print("Data Size | Avg Time (ms) | Throughput (MB/s)")
        print("-" * 50)
        
        for data_size in self.data_sizes:
            data = self.generate_random_string(data_size)
            times = []
            
            # Warm up
            for _ in range(10):
                cipher.encrypt(data)
            
            # Measure
            for _ in range(100):
                start = time.perf_counter()
                encrypted = cipher.encrypt(data)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = statistics.mean(times) * 1000  # Convert to ms
            throughput = (data_size / 1024 / 1024) / statistics.mean(times)  # MB/s
            
            print(f"{data_size:9} | {avg_time:13.2f} | {throughput:17.2f}")
            results.append((data_size, avg_time, throughput))
        
        # Assert performance requirements
        # Should encrypt 1KB in under 1ms
        assert results[0][1] < 1.0
        # Should achieve at least 10 MB/s throughput
        assert results[-1][2] > 10.0
    
    def test_batch_encryption_performance(self):
        """Benchmark batch encryption operations."""
        optimized = get_optimized_cipher()
        
        print("\n=== Batch Encryption Performance ===")
        print("Batch Size | Total Time (s) | Items/sec | Speedup")
        print("-" * 60)
        
        base_data = [self.generate_random_string(1000) for _ in range(1000)]
        
        # Measure single encryption baseline
        cipher = get_cipher()
        start = time.perf_counter()
        for item in base_data[:100]:
            cipher.encrypt(item)
        single_time = time.perf_counter() - start
        
        for batch_size in [10, 50, 100, 500, 1000]:
            data = base_data[:batch_size]
            
            # Measure batch encryption
            start = time.perf_counter()
            encrypted = optimized.batch_encrypt(data)
            batch_time = time.perf_counter() - start
            
            items_per_sec = batch_size / batch_time
            speedup = (single_time * batch_size / 100) / batch_time
            
            print(f"{batch_size:10} | {batch_time:14.3f} | {items_per_sec:9.0f} | {speedup:7.1f}x")
            
            # Verify all items encrypted
            assert len(encrypted) == batch_size
    
    def test_decryption_performance(self):
        """Benchmark decryption operations."""
        cipher = get_cipher()
        
        print("\n=== Decryption Performance ===")
        print("Operation    | Encrypt (ms) | Decrypt (ms) | Ratio")
        print("-" * 55)
        
        for data_size in self.data_sizes:
            data = self.generate_random_string(data_size)
            
            # Encrypt first
            encrypted = cipher.encrypt(data)
            
            # Measure encryption time
            enc_times = []
            for _ in range(100):
                start = time.perf_counter()
                cipher.encrypt(data)
                enc_times.append(time.perf_counter() - start)
            
            # Measure decryption time
            dec_times = []
            for _ in range(100):
                start = time.perf_counter()
                decrypted = cipher.decrypt(encrypted)
                dec_times.append(time.perf_counter() - start)
            
            avg_enc = statistics.mean(enc_times) * 1000
            avg_dec = statistics.mean(dec_times) * 1000
            ratio = avg_dec / avg_enc
            
            print(f"{data_size:5} bytes | {avg_enc:12.2f} | {avg_dec:12.2f} | {ratio:5.2f}")
            
            # Decryption should be comparable to encryption (within 2x)
            assert ratio < 2.0
    
    def test_memory_usage(self):
        """Benchmark memory usage during encryption."""
        cipher = get_cipher()
        
        print("\n=== Memory Usage Analysis ===")
        print("Data Size | Count | Memory Before | Memory After | Increase")
        print("-" * 70)
        
        initial_memory = self.measure_memory()
        
        for data_size in [1000, 10000]:
            for count in [100, 1000]:
                # Force garbage collection
                import gc
                gc.collect()
                
                before = self.measure_memory()
                
                # Encrypt multiple items
                encrypted_items = []
                for i in range(count):
                    data = self.generate_random_string(data_size)
                    encrypted = cipher.encrypt(data)
                    encrypted_items.append(encrypted)
                
                after = self.measure_memory()
                increase = after - before
                
                print(f"{data_size:9} | {count:5} | {before:13.1f} | {after:12.1f} | {increase:8.1f} MB")
                
                # Memory increase should be reasonable (< 100MB for 1000 items)
                if count == 1000:
                    assert increase < 100
    
    def test_concurrent_encryption(self):
        """Test encryption performance under concurrent load."""
        import concurrent.futures
        import threading
        
        print("\n=== Concurrent Encryption Performance ===")
        print("Threads | Time (s) | Throughput | Efficiency")
        print("-" * 50)
        
        def encrypt_batch(cipher, data_list):
            """Encrypt a batch of data."""
            results = []
            for data in data_list:
                encrypted = cipher.encrypt(data)
                results.append(encrypted)
            return results
        
        # Prepare test data
        total_items = 10000
        data_list = [self.generate_random_string(1000) for _ in range(total_items)]
        
        # Single thread baseline
        cipher = get_cipher()
        start = time.perf_counter()
        encrypt_batch(cipher, data_list)
        single_thread_time = time.perf_counter() - start
        
        for num_threads in [1, 2, 4, 8]:
            # Split data among threads
            chunk_size = total_items // num_threads
            chunks = [data_list[i:i + chunk_size] 
                     for i in range(0, total_items, chunk_size)]
            
            start = time.perf_counter()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for chunk in chunks:
                    # Each thread gets its own cipher instance
                    future = executor.submit(encrypt_batch, get_cipher(), chunk)
                    futures.append(future)
                
                # Wait for all to complete
                concurrent.futures.wait(futures)
            
            multi_thread_time = time.perf_counter() - start
            
            throughput = total_items / multi_thread_time
            efficiency = (single_thread_time / multi_thread_time) / num_threads * 100
            
            print(f"{num_threads:7} | {multi_thread_time:8.2f} | {throughput:10.0f} | {efficiency:10.1f}%")
    
    def test_key_derivation_performance(self):
        """Test key derivation and caching performance."""
        from alpha_pulse.utils.encryption import EncryptionKeyManager
        
        print("\n=== Key Derivation Performance ===")
        print("Operation         | Time (μs) | Cache Hit")
        print("-" * 45)
        
        manager = EncryptionKeyManager()
        contexts = ["trading_data", "user_pii", "api_credentials", "audit_logs"]
        
        # Clear cache
        manager._key_cache.clear()
        
        for context in contexts:
            # First derivation (cache miss)
            start = time.perf_counter()
            key1, version1 = manager.derive_data_key(context)
            time1 = (time.perf_counter() - start) * 1_000_000  # Convert to microseconds
            
            # Second derivation (cache hit)
            start = time.perf_counter()
            key2, version2 = manager.derive_data_key(context)
            time2 = (time.perf_counter() - start) * 1_000_000
            
            print(f"{context:17} | {time1:9.1f} | No")
            print(f"{context:17} | {time2:9.1f} | Yes")
            
            # Cache hit should be at least 100x faster
            assert time2 < time1 / 100
    
    def test_database_field_performance(self):
        """Test performance of encrypted database fields."""
        from sqlalchemy import create_engine, Column, Integer, String
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker
        
        Base = declarative_base()
        
        class TestModel(Base):
            __tablename__ = "perf_test"
            id = Column(Integer, primary_key=True)
            normal_field = Column(String)
            encrypted_field = Column(EncryptedString())
            encrypted_json = Column(EncryptedJSON())
        
        # Create in-memory database
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        print("\n=== Database Field Performance ===")
        print("Operation        | Normal (ms) | Encrypted (ms) | Overhead")
        print("-" * 60)
        
        # Test data
        test_string = "test" * 100
        test_json = {"key": "value", "list": list(range(100))}
        
        # Benchmark writes
        normal_times = []
        encrypted_times = []
        
        for i in range(100):
            # Normal field
            start = time.perf_counter()
            obj = TestModel(id=i*3, normal_field=test_string)
            session.add(obj)
            session.commit()
            normal_times.append(time.perf_counter() - start)
            
            # Encrypted field
            start = time.perf_counter()
            obj = TestModel(id=i*3+1, encrypted_field=test_string)
            session.add(obj)
            session.commit()
            encrypted_times.append(time.perf_counter() - start)
        
        avg_normal = statistics.mean(normal_times) * 1000
        avg_encrypted = statistics.mean(encrypted_times) * 1000
        overhead = (avg_encrypted / avg_normal - 1) * 100
        
        print(f"Write String     | {avg_normal:11.2f} | {avg_encrypted:14.2f} | {overhead:7.1f}%")
        
        # Benchmark reads
        normal_read_times = []
        encrypted_read_times = []
        
        for i in range(100):
            # Normal field
            start = time.perf_counter()
            obj = session.query(TestModel).filter_by(id=i*3).first()
            _ = obj.normal_field
            normal_read_times.append(time.perf_counter() - start)
            
            # Encrypted field
            start = time.perf_counter()
            obj = session.query(TestModel).filter_by(id=i*3+1).first()
            _ = obj.encrypted_field
            encrypted_read_times.append(time.perf_counter() - start)
        
        avg_normal_read = statistics.mean(normal_read_times) * 1000
        avg_encrypted_read = statistics.mean(encrypted_read_times) * 1000
        read_overhead = (avg_encrypted_read / avg_normal_read - 1) * 100
        
        print(f"Read String      | {avg_normal_read:11.2f} | {avg_encrypted_read:14.2f} | {read_overhead:7.1f}%")
        
        # Overhead should be less than 50%
        assert overhead < 50
        assert read_overhead < 50


def run_performance_suite():
    """Run the complete performance test suite."""
    print("=" * 70)
    print("AlphaPulse Encryption Performance Test Suite")
    print("=" * 70)
    
    test = TestEncryptionPerformance()
    test.setup()
    
    test.test_single_encryption_performance()
    test.test_batch_encryption_performance()
    test.test_decryption_performance()
    test.test_memory_usage()
    test.test_concurrent_encryption()
    test.test_key_derivation_performance()
    test.test_database_field_performance()
    
    print("\n" + "=" * 70)
    print("Performance tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_performance_suite()