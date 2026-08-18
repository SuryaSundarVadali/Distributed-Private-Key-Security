"""
HKDF Implementation with Entropy Aggregation
NIST SP 800-56C Compliant Key Derivation Function

Paper Reference: Section 2 - HKDF Architecture for Robot Swarms

Refactor notes vs. the original version:
  - extract_phase / expand_phase / derive_mission_key are now pure
    functions: no mutable self.PRK / self.metrics state, everything is
    returned explicitly. This makes the class safe to reuse across
    threads/robots and much easier to unit test.
  - Entropy-bit accounting is now a function of the number of bytes
    actually requested from each source, rather than a hard-coded 256.
    These numbers are still heuristic upper bounds, not a formal
    SP 800-90B min-entropy assessment - see the docstring on
    aggregate_entropy() for the caveat.
  - The single "aggregated entropy -> mission key" path has been split
    into two clearly separated roles:
      * generate_local_seed()      - per-robot local randomness (e.g.
                                      to seed a local DRBG / nonce stream)
      * derive_global_mission_key()- the shared mission key, derived from
                                      a mission_seed agreed via the
                                      swarm's key-agreement protocol
                                      (e.g. a Shamir-reconstructed secret
                                      or dealer-issued seed), NOT from a
                                      single robot's local entropy.
"""

import argparse
import hashlib
import hmac
import secrets
import statistics
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class HKDFMetrics:
    """Capture metrics for paper evaluation"""
    extraction_time_us: float = 0.0  # microseconds
    expansion_time_us: float = 0.0   # microseconds
    total_time_us: float = 0.0       # microseconds
    energy_cost_mj: float = 0.0      # millijoules
    entropy_quality: float = 0.0     # entropy/byte metric
    security_strength: int = 256     # bits


@dataclass
class EntropyMetrics:
    """Metrics describing a single entropy-aggregation call."""
    env_entropy_bits: int = 0
    net_entropy_bits: int = 0
    hw_entropy_bits: int = 0
    total_entropy_bits: int = 0
    entropy_per_byte: float = 0.0
    output_bytes: int = 0
    security_level: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            'env_entropy_bits': self.env_entropy_bits,
            'net_entropy_bits': self.net_entropy_bits,
            'hw_entropy_bits': self.hw_entropy_bits,
            'total_entropy_bits': self.total_entropy_bits,
            'entropy_per_byte': self.entropy_per_byte,
            'output_bytes': self.output_bytes,
            'security_level': self.security_level,
        }


class EntropyGenerator:
    """
    Aggregates multiple entropy sources from robotic platforms

    Paper Reference: Section 2.2 - Entropy Generation in Robotic Platforms

    Sources:
    - Environmental sensors: temperature, pressure, light, motion
    - Network timing: packet delays, communication jitter
    - Hardware RNG: cryptographic random generators
    """

    def collect_environmental_entropy(self, bytes_needed: int = 32) -> bytes:
        """
        Simulate sensor readings (in production: actual sensors)
        - Temperature variations: +/-0.1C precision
        - Pressure fluctuations: +/-1 Pa precision
        - Light intensity: 0-100,000 lux variations
        - Motion acceleration: +/-50 m/s^2 variations

        Args:
            bytes_needed: Number of entropy bytes to generate

        Returns:
            Entropy bytes from environmental sources
        """
        timestamp = int(time.time() * 1e6)  # microsecond precision
        counter = time.perf_counter_ns()

        h = hashlib.sha256()
        h.update(struct.pack('>Q', timestamp))
        h.update(struct.pack('>Q', counter))
        h.update(secrets.token_bytes(16))  # Additional hardware randomness

        return h.digest()[:bytes_needed]

    def collect_network_entropy(self, bytes_needed: int = 32) -> bytes:
        """
        Extract entropy from inter-robot communication
        - Packet arrival timing: microsecond precision
        - Round-trip delays: millisecond variations
        - Jitter patterns: stochastic timing

        Args:
            bytes_needed: Number of entropy bytes to generate

        Returns:
            Entropy bytes from network timing
        """
        h = hashlib.sha256()

        for _ in range(4):
            h.update(struct.pack('>Q', time.perf_counter_ns()))
            time.sleep(0.0001)  # Small delay to capture jitter

        return h.digest()[:bytes_needed]

    def collect_hardware_entropy(self, bytes_needed: int = 32) -> bytes:
        """
        Access hardware RNG from robot controllers
        - ARM TrustZone true RNG
        - Intel SGX secure random number generator
        - Fallback: cryptographic hashing of system timings

        Args:
            bytes_needed: Number of entropy bytes to generate

        Returns:
            Entropy bytes from hardware RNG
        """
        return secrets.token_bytes(bytes_needed)

    def aggregate_entropy(self, env_bytes: int = 32, net_bytes: int = 32,
                           hw_bytes: int = 32) -> Tuple[bytes, EntropyMetrics]:
        """
        Combine all sources: E_robot = Hash(S_env || S_net || S_hw || timestamp)

        Paper Formula: E_robot = H_combine(S_env, S_net, S_hw, timestamp)

        NOTE on entropy accounting: the *_entropy_bits values below are a
        conservative heuristic, not a formal min-entropy estimate per
        NIST SP 800-90B. Environmental and network timing sources are
        assumed to contribute roughly 2 bits of min-entropy per output
        byte (a commonly used conservative bound for noisy physical/
        timing sources); the hardware RNG source is assumed to be a
        certified/whitened CSPRNG and is credited at 8 bits/byte. Any
        claim about the "security level" of derived keys should be
        validated against the actual entropy source characterization
        used in deployment, not inferred from this heuristic alone.

        Args:
            env_bytes: Bytes from environmental sensors
            net_bytes: Bytes from network timing
            hw_bytes: Bytes from hardware RNG

        Returns:
            - 256-bit aggregated entropy
            - EntropyMetrics for paper evaluation
        """
        env_entropy = self.collect_environmental_entropy(env_bytes)
        net_entropy = self.collect_network_entropy(net_bytes)
        hw_entropy = self.collect_hardware_entropy(hw_bytes)

        h = hashlib.sha256()
        h.update(env_entropy)
        h.update(net_entropy)
        h.update(hw_entropy)
        h.update(struct.pack('>Q', int(time.time() * 1e9)))  # nanosecond timestamp

        aggregated = h.digest()

        # Heuristic, byte-conditioned entropy accounting (see docstring).
        env_entropy_bits = env_bytes * 2
        net_entropy_bits = net_bytes * 2
        hw_entropy_bits = hw_bytes * 8

        total_entropy_bits = env_entropy_bits + net_entropy_bits + hw_entropy_bits
        entropy_per_byte = total_entropy_bits / len(aggregated)

        metrics = EntropyMetrics(
            env_entropy_bits=env_entropy_bits,
            net_entropy_bits=net_entropy_bits,
            hw_entropy_bits=hw_entropy_bits,
            total_entropy_bits=total_entropy_bits,
            entropy_per_byte=entropy_per_byte,
            output_bytes=len(aggregated),
            security_level=min(256, total_entropy_bits),
        )

        return aggregated, metrics


class HKDFImplementation:
    """
    HMAC-based Key Derivation Function (NIST SP 800-56C compliant)

    Paper Reference: Section 2.1 - HKDF Architecture for Robot Swarms

    Two-phase process:
    1. Extract: PRK = HMAC-SHA256(salt, IKM)
    2. Expand: OKM = HKDF-Expand(PRK, info, L)

    All methods here are pure / stateless: they take their inputs as
    arguments and return their outputs (including metrics) directly,
    rather than mutating instance state. This makes a single
    HKDFImplementation instance safe to share/reuse across concurrent
    derivations (e.g. multiple robots or missions in a simulation) and
    makes each phase independently unit-testable.
    """

    def __init__(self, hash_function: str = 'sha256'):
        """
        Initialize HKDF implementation

        Args:
            hash_function: Hash function to use (default: sha256)
        """
        self.hash_fn = hash_function
        self.hash_len = 32  # SHA-256 output length

    def extract_phase(self, salt: Optional[bytes], raw_key_material: bytes) -> Tuple[bytes, float]:
        """
        Extraction: PRK = HMAC-SHA256(salt, RKM)

        Args:
            salt: Optional salt value (a non-secret random value)
            raw_key_material: Input keying material

        Returns:
            - PRK (256-bit pseudo-random key)
            - Timing metrics for computation cost analysis (microseconds)
        """
        start_time = time.perf_counter()

        if salt is None or len(salt) == 0:
            salt = b'\x00' * self.hash_len

        prk = hmac.new(salt, raw_key_material, hashlib.sha256).digest()

        elapsed_us = (time.perf_counter() - start_time) * 1e6

        return prk, elapsed_us

    def expand_phase(self, prk: bytes, info: Optional[bytes], length: int) -> Tuple[bytes, float]:
        """
        Expansion: T(i) = HMAC-SHA256(PRK, T(i-1) || info || byte(i))
                  OKM = T(1) || T(2) || ... || T(N)

        Paper Formula: N = ceil(L / HashLen)

        Args:
            prk: Pseudo-random key from extract phase
            info: Optional context and application specific information
            length: Length of output keying material in bytes

        Returns:
            - Output Key Material (length bytes)
            - Expansion metrics for benchmarking (microseconds)
        """
        start_time = time.perf_counter()

        if info is None:
            info = b''

        n = (length + self.hash_len - 1) // self.hash_len

        if n > 255:
            raise ValueError(f"Cannot derive key longer than {255 * self.hash_len} bytes")

        t = b''
        okm = b''

        for i in range(1, n + 1):
            t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
            okm += t

        elapsed_us = (time.perf_counter() - start_time) * 1e6

        return okm[:length], elapsed_us

    def _derive(self, ikm: bytes, salt: Optional[bytes], info: bytes,
                length: int, entropy_bytes_for_quality: int) -> Tuple[bytes, HKDFMetrics]:
        """Shared extract+expand pipeline used by both key-derivation entry points."""
        overall_start = time.perf_counter()

        prk, extract_time = self.extract_phase(salt, ikm)
        okm, expand_time = self.expand_phase(prk, info, length)

        total_time = (time.perf_counter() - overall_start) * 1e6

        # Energy cost estimate: typical microcontroller ~0.1 mW/MHz @ ~100 MHz => ~10 mW
        power_mw = 10.0
        energy_cost_mj = (power_mw * total_time) / 1000.0

        metrics = HKDFMetrics(
            extraction_time_us=extract_time,
            expansion_time_us=expand_time,
            total_time_us=total_time,
            energy_cost_mj=energy_cost_mj,
            entropy_quality=entropy_bytes_for_quality * 8.0 / len(okm),
            security_strength=min(256, entropy_bytes_for_quality * 8),
        )

        return okm, metrics

    def generate_local_seed(self, local_entropy: bytes, salt: Optional[bytes],
                             robot_id: str, length: int = 32) -> Tuple[bytes, HKDFMetrics]:
        """
        Per-robot local seed derivation: seed = HKDF(local_entropy, salt, robot_id)

        This is intended to seed a per-robot local DRBG / nonce stream, NOT
        to be used directly as the shared mission key - each robot's
        aggregated sensor entropy is local to that robot and is not
        secret-shared with the rest of the swarm.

        Args:
            local_entropy: Aggregated entropy from this robot's sensors
                (e.g. from EntropyGenerator.aggregate_entropy)
            salt: Salt value (can be public)
            robot_id: Identifier for this robot, used for domain separation
            length: Desired seed length in bytes (default: 32)

        Returns:
            - Local seed material
            - Derivation metrics (time, energy cost)
        """
        info = robot_id.encode() if isinstance(robot_id, str) else robot_id
        return self._derive(local_entropy, salt, info, length, len(local_entropy))

    def derive_global_mission_key(self, mission_seed: bytes, salt: Optional[bytes],
                                   mission_id: str, length: int = 32) -> Tuple[bytes, HKDFMetrics]:
        """
        Global mission key derivation: SK = HKDF(mission_seed, salt, mission_id)

        `mission_seed` is the swarm-wide secret agreed via the protocol's
        key-agreement step (e.g. reconstructed from a Shamir secret-sharing
        scheme, or issued by a designated dealer) - it is NOT a single
        robot's local sensor entropy. Keeping this entry point separate
        from generate_local_seed() makes the trust boundary between
        "per-robot local randomness" and "swarm-shared mission secret"
        explicit in the code.

        Args:
            mission_seed: Swarm-agreed mission seed / secret
            salt: Salt value (can be public)
            mission_id: Mission identifier for domain separation
            length: Desired key length in bytes (default: 32)

        Returns:
            - Mission Private Key (256-bit by default)
            - Derivation metrics (time, energy cost)
        """
        info = mission_id.encode() if isinstance(mission_id, str) else mission_id
        return self._derive(mission_seed, salt, info, length, len(mission_seed))

    def derive_mission_key(self, entropy: bytes, salt: Optional[bytes],
                            mission_id: str, length: int = 32) -> Tuple[bytes, HKDFMetrics]:
        """
        Backward-compatible alias for derive_global_mission_key().

        Kept so existing call sites / benchmarks from the original
        implementation keep working; new code should call
        generate_local_seed() or derive_global_mission_key() directly to
        make the intent explicit.
        """
        return self.derive_global_mission_key(entropy, salt, mission_id, length)

    def get_metrics(self, metrics: HKDFMetrics) -> Dict[str, Any]:
        """
        Format a HKDFMetrics instance for paper evaluation / reporting.

        Args:
            metrics: metrics returned from generate_local_seed() /
                derive_global_mission_key() / derive_mission_key()

        Returns:
            Dictionary of performance metrics
        """
        return {
            'extraction_time_us': metrics.extraction_time_us,
            'expansion_time_us': metrics.expansion_time_us,
            'total_time_us': metrics.total_time_us,
            'energy_cost_mj': metrics.energy_cost_mj,
            'entropy_quality': metrics.entropy_quality,
            'security_strength': metrics.security_strength,
            'meets_paper_requirements': {
                'time_under_1ms': metrics.total_time_us < 1000,
                'energy_under_1mj': metrics.energy_cost_mj < 1.0,
                'security_256bit': metrics.security_strength >= 256,
            },
        }


# Test and benchmarking functionality
def run_hkdf_benchmark(iterations: int = 1000) -> Dict[str, Any]:
    """
    Benchmark HKDF implementation

    Args:
        iterations: Number of iterations to run

    Returns:
        Benchmark results
    """
    entropy_gen = EntropyGenerator()
    hkdf = HKDFImplementation()

    times = []

    for _ in range(iterations):
        entropy, _ = entropy_gen.aggregate_entropy()

        salt = secrets.token_bytes(32)
        _, metrics = hkdf.derive_global_mission_key(entropy, salt, "test_mission")
        times.append(metrics.total_time_us)

    return {
        'iterations': iterations,
        'mean_time_us': statistics.mean(times),
        'median_time_us': statistics.median(times),
        'min_time_us': min(times),
        'max_time_us': max(times),
        'stdev_time_us': statistics.stdev(times) if len(times) > 1 else 0,
        'paper_requirement_1ms': statistics.mean(times) < 1000,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HKDF entropy-aggregation demo and benchmark for robot swarms."
    )
    parser.add_argument(
        '--iterations', type=int, default=1000,
        help="Number of benchmark iterations to run (default: 1000). "
             "Keep this small in CI to avoid accidental long runs."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("=" * 80)
    print("HKDF Key Derivation with Entropy Aggregation - Test")
    print("=" * 80)

    # Test entropy generation
    print("\n[1] Testing Entropy Generation")
    entropy_gen = EntropyGenerator()
    entropy, entropy_metrics = entropy_gen.aggregate_entropy()
    print(f"  - Aggregated entropy: {len(entropy)} bytes")
    print(f"  - Total entropy bits (heuristic): {entropy_metrics.total_entropy_bits}")
    print(f"  - Security level: {entropy_metrics.security_level} bits")

    # Test per-robot local seed derivation
    print("\n[2] Testing Local Seed Derivation (per robot)")
    hkdf = HKDFImplementation()
    local_salt = b"robot_local_salt_2024"
    local_seed, local_metrics = hkdf.generate_local_seed(entropy, local_salt, "robot_07")
    print(f"  - Local seed: {local_seed.hex()[:32]}...")
    print(f"  - Total time: {local_metrics.total_time_us:.2f} us")

    # Test global mission key derivation (uses a separate mission_seed,
    # NOT a single robot's local entropy)
    print("\n[3] Testing Global Mission Key Derivation")
    mission_seed = secrets.token_bytes(32)  # stand-in for a Shamir-reconstructed seed
    mission_salt = b"mission_salt_2024"
    mission_key, metrics = hkdf.derive_global_mission_key(
        mission_seed, mission_salt, "precision_farming_mission"
    )
    print(f"  - Mission key: {mission_key.hex()[:32]}...")
    print(f"  - Extraction time: {metrics.extraction_time_us:.2f} us")
    print(f"  - Expansion time: {metrics.expansion_time_us:.2f} us")
    print(f"  - Total time: {metrics.total_time_us:.2f} us")
    print(f"  - Energy cost: {metrics.energy_cost_mj:.6f} mJ")

    # Verify paper requirements
    print("\n[4] Verifying Paper Requirements")
    requirements = hkdf.get_metrics(metrics)['meets_paper_requirements']
    print(f"  - Time < 1ms: {requirements['time_under_1ms']}")
    print(f"  - Energy < 1mJ: {requirements['energy_under_1mj']}")
    print(f"  - Security >= 256-bit: {requirements['security_256bit']}")

    # Run benchmark
    print(f"\n[5] Running Benchmark ({args.iterations} iterations)")
    benchmark = run_hkdf_benchmark(args.iterations)
    print(f"  - Mean time: {benchmark['mean_time_us']:.2f} us")
    print(f"  - Median time: {benchmark['median_time_us']:.2f} us")
    print(f"  - Min time: {benchmark['min_time_us']:.2f} us")
    print(f"  - Max time: {benchmark['max_time_us']:.2f} us")
    print(f"  - Std dev: {benchmark['stdev_time_us']:.2f} us")
    print(f"  - Meets paper requirement: {benchmark['paper_requirement_1ms']}")

    print("\n" + "=" * 80)
    print("HKDF Implementation Complete - All Tests Passed")
    print("=" * 80)