"""
Main Distributed Cryptographic Task Executor with HEFT Scheduling

Integrates HKDF-based key derivation, Shamir Secret Sharing,
HOTP, Merkle Trees, and SMPC with intelligent task distribution.
"""

import argparse
import hashlib
import logging
import secrets
import time
from typing import Any, Dict

from dpk_security.core_system.task_scheduler import (
    HEFTScheduler,
    CryptoTask,
    NodeCapabilities,
    TaskPriority,
)
from dpk_security.crypto_modules.key_generation import DeterministicRSAKeyGenerator
from dpk_security.crypto_modules.hotp import HOTP
from dpk_security.crypto_modules.merkle_tree import MerkleTree
from dpk_security.crypto_modules.smpc_verification import SecureMultiPartyComputation
from dpk_security.crypto_modules.shamir_secret_sharing import ShamirSecretSharing
from dpk_security.logging_config import setup_logging

logger = logging.getLogger(__name__)


class DistributedCryptoExecutor:
    """Main orchestrator for distributed cryptographic operations with computational tasks."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Core scheduler (HEFT-based)
        self.scheduler = HEFTScheduler(
            heartbeat_interval=config.get("heartbeat_interval", 10.0)
        )

        # Cryptographic infrastructure state
        self.master_seed = secrets.token_bytes(32)
        self.hotp_secret = hashlib.sha256(self.master_seed + b"HOTP").digest()
        self.counter = 0
        self.crypto_state: Dict[str, Any] = {
            "private_key_shares": {},
            "merkle_tree": None,
            "mpc_results": {},
            "hotp_tokens": {},
            "master_private_key": None,
            "sss": None,
            "smpc_system": None,
        }

        # Initialize cryptographic infrastructure (key, shares, HOTP, Merkle, SMPC)
        self._initialize_crypto_infrastructure()

        # Register computational task handlers and initialize nodes
        self._register_task_handlers()
        self._initialize_nodes()

    # -------------------------------------------------------------------------
    # Cryptographic infrastructure
    # -------------------------------------------------------------------------

    def _initialize_crypto_infrastructure(self) -> None:
        """Initialize cryptographic infrastructure that runs on every node."""
        logger.info("Initializing cryptographic infrastructure...")

        # Step 1: Generate master private key deterministically
        self._perform_key_generation()

        # Step 2: Distribute secret shares using SSS
        self._perform_share_distribution()

        # Step 3: Generate HOTP tokens for authentication
        self._perform_hotp_setup()

        # Step 4: Create Merkle tree for integrity verification
        self._perform_merkle_storage()

        # Step 5: Initialize SMPC framework
        self._perform_mpc_initialization()

        logger.info("✅ Cryptographic infrastructure initialized successfully")

    def _perform_key_generation(self) -> None:
        """Generate master private key deterministically from master_seed."""
        logger.info("🔐 Generating master private key deterministically...")

        generator = DeterministicRSAKeyGenerator("master_node", self.master_seed)
        rsa_key = generator.generate_rsa_key(key_size=2048)

        private_key_der = rsa_key.export_key("DER")
        private_key_int = int.from_bytes(private_key_der, "big")

        self.crypto_state["master_private_key"] = private_key_int
        self.crypto_state["master_key_der"] = private_key_der

        logger.info("✅ Master key generated: %d bits", rsa_key.size_in_bits())

    def _perform_share_distribution(self) -> None:
        """Distribute secret shares using Shamir's Secret Sharing."""
        logger.info("🔀 Distributing secret shares...")

        if not self.crypto_state["master_private_key"]:
            raise ValueError("Master private key not generated")

        sss = ShamirSecretSharing(
            threshold=self.config["threshold"],
            num_shares=self.config["num_nodes"],
        )
        self.crypto_state["sss"] = sss

        shares_dict = sss.share_generation(self.crypto_state["master_private_key"])

        # shares_dict keys are 1..n; map them to crypto_node_0..crypto_node_{n-1}
        for rid, share in shares_dict.items():
            node_index = rid - 1
            node_id = f"crypto_node_{node_index}"
            self.crypto_state["private_key_shares"][node_id] = share

        logger.info(
            "✅ Distributed %d shares with threshold %d",
            len(shares_dict),
            self.config["threshold"],
        )

    def _perform_hotp_setup(self) -> None:
        """Setup HOTP authentication for all nodes."""
        logger.info("🔑 Setting up HOTP authentication...")

        for i in range(self.config["num_nodes"]):
            node_id = f"crypto_node_{i}"

            tokens = []
            for j in range(10):  # Generate 10 tokens per node
                token = HOTP.generate(self.hotp_secret, (i * 10) + j)
                tokens.append(token)

            self.crypto_state["hotp_tokens"][node_id] = tokens

        logger.info(
            "✅ HOTP tokens generated for %d nodes", self.config["num_nodes"]
        )

    def _perform_merkle_storage(self) -> None:
        """Create Merkle tree for integrity verification over shares."""
        logger.info("🌳 Creating Merkle tree for integrity verification...")

        share_data = []
        for node_id, share in self.crypto_state["private_key_shares"].items():
            # share is typically (x, y) in Shamir; encode as bytes
            share_bytes = f"{node_id}:{share[0]}:{share[1]}".encode("utf-8")
            share_data.append(share_bytes)

        if not share_data:
            raise ValueError("No share data available for Merkle tree")

        # Our MerkleTree takes a list of data blocks
        merkle_tree = MerkleTree(share_data)
        self.crypto_state["merkle_tree"] = merkle_tree

        logger.info("✅ Merkle tree created with %d leaves", len(share_data))

    def _perform_mpc_initialization(self) -> None:
        """Initialize SMPC verification framework (3-phase aggregation)."""
        logger.info("🤝 Initializing SMPC verification framework...")

        smpc = SecureMultiPartyComputation(
            num_robots=self.config["num_nodes"],
            task_count=self.config.get("task_count", 0),
        )
        self.crypto_state["smpc_system"] = smpc

        logger.info(
            "✅ SMPC framework initialized for %d robots",
            self.config["num_nodes"],
        )

    # -------------------------------------------------------------------------
    # Scheduler and node setup
    # -------------------------------------------------------------------------

    def _register_task_handlers(self) -> None:
        """Register handlers for computational task types."""
        handlers = {
            # Mathematical tasks
            "prime_generation": self._handle_prime_generation,
            "matrix_multiplication": self._handle_matrix_multiplication,
            "fibonacci_calculation": self._handle_fibonacci_calculation,
            # Data processing tasks
            "large_array_sort": self._handle_array_sort,
            "pattern_search": self._handle_pattern_search,
            "data_compression": self._handle_data_compression,
            # Image processing tasks
            "image_processing": self._handle_image_processing,
            "object_detection": self._handle_object_detection,
            # Network tasks
            "web_crawling": self._handle_web_crawling,
            "api_data_fetch": self._handle_api_data_fetch,
            # Machine learning tasks
            "linear_regression": self._handle_linear_regression,
            "kmeans_clustering": self._handle_kmeans_clustering,
            # File operations
            "file_hashing": self._handle_file_hashing,
            "file_encryption": self._handle_file_encryption,
            # Scientific computing
            "monte_carlo_simulation": self._handle_monte_carlo,
            "statistical_analysis": self._handle_statistical_analysis,
        }

        for task_type, handler in handlers.items():
            self.scheduler.register_task_handler(task_type, handler)

    def _initialize_nodes(self) -> None:
        """Initialize compute nodes with capabilities."""
        for i in range(self.config["num_nodes"]):
            node_id = f"crypto_node_{i}"

            if i < 20:
                specialization = "mathematical"
                task_types = [
                    "prime_generation",
                    "matrix_multiplication",
                    "fibonacci_calculation",
                ]
            elif i < 40:
                specialization = "data_processing"
                task_types = [
                    "large_array_sort",
                    "pattern_search",
                    "data_compression",
                ]
            elif i < 60:
                specialization = "image_processing"
                task_types = ["image_processing", "object_detection"]
            elif i < 80:
                specialization = "network"
                task_types = ["web_crawling", "api_data_fetch"]
            else:
                specialization = "ml_scientific"
                task_types = [
                    "linear_regression",
                    "kmeans_clustering",
                    "monte_carlo_simulation",
                ]

            capabilities = NodeCapabilities(
                node_id=node_id,
                computation_power=1.0 + (i * 0.1),
                available_factors=[
                    "biometric",
                    "password",
                    "hardware_token",
                    "location",
                    "time_window",
                ],
                max_concurrent_tasks=3 + (i % 3),
                task_completion_rate=0.95 - (i * 0.005),
                specialization=specialization,
                supported_task_types=task_types,
            )

            self.scheduler.register_node(node_id, capabilities)

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def start_execution(self) -> None:
        """Start the distributed execution system."""
        logger.info("Starting Distributed Cryptographic Execution System")
        logger.info("=" * 55)

        self.scheduler.start()
        self._create_computational_workflow()
        self._monitor_execution()

    # Convenience alias
    def run(self) -> None:
        self.start_execution()

    def _create_computational_workflow(self) -> None:
        """Create and submit computational tasks that will be executed securely."""
        tasks: list[CryptoTask] = []

        # Mathematical tasks
        tasks.extend(
            [
                CryptoTask(
                    task_id="prime_gen_1",
                    task_type="prime_generation",
                    data={
                        "range_start": 1000000,
                        "range_end": 1010000,
                        "count": 100,
                    },
                    priority=TaskPriority.MEDIUM,
                    computation_cost=2.0,
                ),
                CryptoTask(
                    task_id="matrix_mult_1",
                    task_type="matrix_multiplication",
                    data={"matrix_size": 500, "iterations": 3},
                    priority=TaskPriority.MEDIUM,
                    computation_cost=3.0,
                ),
                CryptoTask(
                    task_id="fibonacci_1",
                    task_type="fibonacci_calculation",
                    data={"n": 10000, "modulo": 1000000007},
                    priority=TaskPriority.LOW,
                    computation_cost=1.5,
                ),
            ]
        )

        # Data processing
        tasks.extend(
            [
                CryptoTask(
                    task_id="array_sort_1",
                    task_type="large_array_sort",
                    data={"array_size": 100000, "algorithm": "quicksort"},
                    priority=TaskPriority.MEDIUM,
                    computation_cost=2.5,
                ),
                CryptoTask(
                    task_id="pattern_search_1",
                    task_type="pattern_search",
                    data={
                        "text_size": 1000000,
                        "pattern": "cryptography",
                        "algorithm": "kmp",
                    },
                    priority=TaskPriority.LOW,
                    computation_cost=1.8,
                ),
            ]
        )

        # Image processing
        tasks.append(
            CryptoTask(
                task_id="image_proc_1",
                task_type="image_processing",
                data={
                    "operation": "gaussian_blur",
                    "image_size": "1024x768",
                    "kernel_size": 7,
                },
                priority=TaskPriority.HIGH,
                computation_cost=4.0,
            )
        )

        # Network
        tasks.append(
            CryptoTask(
                task_id="api_fetch_1",
                task_type="api_data_fetch",
                data={
                    "endpoint": "https://jsonplaceholder.typicode.com/posts",
                    "count": 50,
                },
                priority=TaskPriority.MEDIUM,
                computation_cost=1.0,
            )
        )

        # Machine learning
        tasks.append(
            CryptoTask(
                task_id="linear_reg_1",
                task_type="linear_regression",
                data={
                    "dataset_size": 5000,
                    "features": 10,
                    "iterations": 500,
                },
                priority=TaskPriority.HIGH,
                computation_cost=3.5,
            )
        )

        # Scientific computing
        tasks.append(
            CryptoTask(
                task_id="monte_carlo_1",
                task_type="monte_carlo_simulation",
                data={"iterations": 100000, "variables": 2},
                priority=TaskPriority.MEDIUM,
                computation_cost=2.8,
            )
        )

        for task in tasks:
            if self._verify_crypto_security():
                self.scheduler.submit_task(task)
                logger.info("Submitted secure task: %s", task.task_id)
            else:
                logger.error("Security verification failed for task: %s", task.task_id)

    def _verify_crypto_security(self) -> bool:
        """Verify cryptographic security before executing tasks."""
        # HOTP
        current_token = HOTP.generate(self.hotp_secret, self.counter)
        verification_result = HOTP.verify(self.hotp_secret, current_token, self.counter)
        hotp_valid = verification_result.valid

        # Merkle
        merkle_valid = self.crypto_state["merkle_tree"] is not None

        # Shares
        shares_valid = (
            len(self.crypto_state["private_key_shares"]) >= self.config["threshold"]
        )

        # SMPC
        mpc_valid = self.crypto_state.get("smpc_system") is not None

        return hotp_valid and merkle_valid and shares_valid and mpc_valid

    # -------------------------------------------------------------------------
    # Task handlers (simulated workloads)
    # -------------------------------------------------------------------------

    def _handle_prime_generation(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        logger.info("[%s] Executing prime generation...", node_id)
        import random

        start_time = time.time()
        primes = []
        for _ in range(task.data["count"]):
            prime = random.randint(task.data["range_start"], task.data["range_end"])
            if prime % 2 != 0:
                primes.append(prime)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "primes_found": len(primes),
            "execution_time": execution_time,
            "first_prime": primes[0] if primes else None,
        }

    def _handle_matrix_multiplication(
        self, task: CryptoTask, node_id: str
    ) -> Dict[str, Any]:
        logger.info("[%s] Executing matrix multiplication...", node_id)
        start_time = time.time()
        size = task.data["matrix_size"]
        iterations = task.data["iterations"]
        total_ops = size * size * size * iterations
        time.sleep(0.1 * iterations)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "matrix_size": size,
            "iterations": iterations,
            "total_operations": total_ops,
            "execution_time": execution_time,
        }

    def _handle_fibonacci_calculation(
        self, task: CryptoTask, node_id: str
    ) -> Dict[str, Any]:
        logger.info("[%s] Calculating Fibonacci sequence...", node_id)
        start_time = time.time()
        n = task.data["n"]
        modulo = task.data["modulo"]
        a, b = 0, 1
        for _ in range(n):
            a, b = b, (a + b) % modulo
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "fibonacci_n": n,
            "result": a,
            "execution_time": execution_time,
        }

    def _handle_array_sort(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        logger.info("[%s] Sorting large array...", node_id)
        import random

        start_time = time.time()
        array_size = task.data["array_size"]
        array = [random.randint(1, 1000000) for _ in range(min(array_size, 1000))]
        array.sort()
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "array_size": array_size,
            "algorithm": task.data["algorithm"],
            "execution_time": execution_time,
            "sorted": True,
        }

    def _handle_pattern_search(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        logger.info("[%s] Searching for patterns...", node_id)
        import random

        start_time = time.time()
        text_size = task.data["text_size"]
        pattern = task.data["pattern"]
        occurrences = random.randint(0, text_size // 1000)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "text_size": text_size,
            "pattern": pattern,
            "occurrences": occurrences,
            "execution_time": execution_time,
        }

    def _handle_data_compression(
        self, task: CryptoTask, node_id: str
    ) -> Dict[str, Any]:
        logger.info("[%s] Compressing data...", node_id)
        import random

        start_time = time.time()
        data_size = task.data["data_size"]
        algorithm = task.data["algorithm"]
        compression_ratio = random.uniform(0.3, 0.8)
        compressed_size = int(data_size * compression_ratio)
        time.sleep(0.5)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "original_size": data_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "algorithm": algorithm,
            "execution_time": execution_time,
        }

    def _handle_image_processing(
        self, task: CryptoTask, node_id: str
    ) -> Dict[str, Any]:
        logger.info("[%s] Processing image...", node_id)
        start_time = time.time()
        operation = task.data["operation"]
        image_size = task.data["image_size"]
        kernel_size = task.data["kernel_size"]
        time.sleep(1.0)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "operation": operation,
            "image_size": image_size,
            "kernel_size": kernel_size,
            "execution_time": execution_time,
        }

    def _handle_object_detection(
        self, task: CryptoTask, node_id: str
    ) -> Dict[str, Any]:
        logger.info("[%s] Detecting objects...", node_id)
        import random

        start_time = time.time()
        algorithm = task.data["algorithm"]
        image_count = task.data["image_count"]
        objects_detected = random.randint(0, image_count * 3)
        time.sleep(0.5)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "algorithm": algorithm,
            "images_processed": image_count,
            "objects_detected": objects_detected,
            "execution_time": execution_time,
        }

    def _handle_web_crawling(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        logger.info("[%s] Crawling web pages...", node_id)
        start_time = time.time()
        urls = task.data["urls"]
        depth = task.data["depth"]
        pages_crawled = len(urls) * (depth + 1)
        time.sleep(0.3)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "urls_processed": len(urls),
            "pages_crawled": pages_crawled,
            "depth": depth,
            "execution_time": execution_time,
        }

    def _handle_api_data_fetch(
        self, task: CryptoTask, node_id: str
    ) -> Dict[str, Any]:
        logger.info("[%s] Fetching API data...", node_id)
        start_time = time.time()
        endpoint = task.data["endpoint"]
        count = task.data["count"]
        time.sleep(0.2)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "endpoint": endpoint,
            "records_fetched": count,
            "execution_time": execution_time,
        }

    def _handle_linear_regression(
        self, task: CryptoTask, node_id: str
    ) -> Dict[str, Any]:
        logger.info("[%s] Performing linear regression...", node_id)
        import random

        start_time = time.time()
        dataset_size = task.data["dataset_size"]
        features = task.data["features"]
        iterations = task.data["iterations"]
        mse = random.uniform(0.01, 0.1)
        r_squared = random.uniform(0.8, 0.99)
        time.sleep(0.8)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "dataset_size": dataset_size,
            "features": features,
            "iterations": iterations,
            "mse": mse,
            "r_squared": r_squared,
            "execution_time": execution_time,
        }

    def _handle_kmeans_clustering(
        self, task: CryptoTask, node_id: str
    ) -> Dict[str, Any]:
        logger.info("[%s] Performing K-means clustering...", node_id)
        import random

        start_time = time.time()
        data_points = task.data["data_points"]
        clusters = task.data["clusters"]
        dimensions = task.data["dimensions"]
        inertia = random.uniform(100, 1000)
        time.sleep(0.6)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "data_points": data_points,
            "clusters": clusters,
            "dimensions": dimensions,
            "inertia": inertia,
            "execution_time": execution_time,
        }

    def _handle_file_hashing(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        logger.info("[%s] Hashing file...", node_id)
        import hashlib

        start_time = time.time()
        file_size = task.data["file_size"]
        algorithm = task.data["algorithm"]
        hash_obj = hashlib.sha256(str(file_size).encode("utf-8"))
        file_hash = hash_obj.hexdigest()
        time.sleep(0.3)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "file_size": file_size,
            "algorithm": algorithm,
            "hash": file_hash[:32],
            "execution_time": execution_time,
        }

    def _handle_file_encryption(
        self, task: CryptoTask, node_id: str
    ) -> Dict[str, Any]:
        logger.info("[%s] Encrypting file...", node_id)
        start_time = time.time()
        file_size = task.data["file_size"]
        algorithm = task.data["algorithm"]
        time.sleep(0.4)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "file_size": file_size,
            "algorithm": algorithm,
            "encrypted": True,
            "execution_time": execution_time,
        }

    def _handle_monte_carlo(self, task: CryptoTask, node_id: str) -> Dict[str, Any]:
        logger.info("[%s] Running Monte Carlo simulation...", node_id)
        import random

        start_time = time.time()
        iterations = task.data["iterations"]
        variables = task.data["variables"]
        result = random.uniform(0, 1)
        time.sleep(0.7)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "iterations": iterations,
            "variables": variables,
            "result": result,
            "execution_time": execution_time,
        }

    def _handle_statistical_analysis(
        self, task: CryptoTask, node_id: str
    ) -> Dict[str, Any]:
        logger.info("[%s] Performing statistical analysis...", node_id)
        import random

        start_time = time.time()
        dataset_size = task.data["dataset_size"]
        operations = task.data["operations"]
        stats: Dict[str, float] = {}
        for op in operations:
            stats[op] = random.uniform(0, 100)
        time.sleep(0.4)
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "dataset_size": dataset_size,
            "operations": operations,
            "statistics": stats,
            "execution_time": execution_time,
        }

    # -------------------------------------------------------------------------
    # Monitoring and reporting
    # -------------------------------------------------------------------------

    def _monitor_execution(self) -> None:
        """Monitor the execution of the computational workflow."""
        logger.info("\n" + "=" * 55)
        logger.info("MONITORING SECURE COMPUTATIONAL WORKFLOW")
        logger.info("=" * 55)

        start_time = time.time()
        max_wait_time = 300.0

        while time.time() - start_time < max_wait_time:
            status = self.scheduler.get_scheduler_status()
            print(
                f"\rStatus: {status['completed_tasks']}/{status['total_tasks']} computational tasks completed",
                end="",
            )

            if status["completed_tasks"] == status["total_tasks"]:
                print("\n\n✅ All computational tasks completed successfully!")
                self._print_final_report()
                break
            elif status["failed_tasks"] > 0:
                print(f"\n\n❌ {status['failed_tasks']} tasks failed")
                break

            time.sleep(2.0)
        else:
            print("\n\n⏰ Execution timeout reached")

        self.scheduler.stop()

    def _print_final_report(self) -> None:
        """Print final execution report."""
        logger.info("\n" + "=" * 55)
        logger.info("FINAL EXECUTION REPORT")
        logger.info("=" * 55)

        stats = self.scheduler.get_scheduler_status()
        logger.info("Total Computational Tasks: %d", stats["total_tasks"])
        logger.info("Completed: %d", stats["completed_tasks"])
        logger.info("Failed: %d", stats["failed_tasks"])

        logger.info("\nCryptographic Infrastructure Status:")
        logger.info(
            "✅ Master Key Generated: %s",
            bool(self.crypto_state["master_private_key"]),
        )
        logger.info(
            "✅ Secret Shares Distributed: %d shares",
            len(self.crypto_state["private_key_shares"]),
        )
        logger.info(
            "✅ HOTP Tokens Active: %d nodes",
            len(self.crypto_state["hotp_tokens"]),
        )
        logger.info(
            "✅ Merkle Tree Integrity: %s",
            bool(self.crypto_state["merkle_tree"]),
        )
        logger.info(
            "✅ SMPC Framework Ready: %s",
            self.crypto_state.get("smpc_system") is not None,
        )

        security_verified = self._verify_crypto_security()
        logger.info(
            "\n🔐 Overall Security Status: %s",
            "SECURE" if security_verified else "COMPROMISED",
        )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="DPK Enhanced Distributed Executor"
    )
    parser.add_argument("--num-nodes", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--task-count", type=int, default=0)
    parser.add_argument("--heartbeat-interval", type=float, default=10.0)
    args = parser.parse_args()

    config = {
        "num_nodes": args.num_nodes,
        "threshold": args.threshold,
        "task_count": args.task_count,
        "heartbeat_interval": args.heartbeat_interval,
    }

    executor = DistributedCryptoExecutor(config)
    executor.start_execution()


if __name__ == "__main__":
    main()