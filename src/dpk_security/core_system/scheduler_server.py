"""
Central Scheduler Server
Coordinates task distribution using HEFT algorithm.
"""

import argparse
import logging
import secrets
import threading
import time
from typing import Any, Dict

from flask import Flask, jsonify, request

from dpk_security.core_system.task_scheduler import (
    HEFTScheduler,
    CryptoTask,
    TaskStatus,
    TaskPriority,
    NodeCapabilities,
)
from dpk_security.logging_config import setup_logging

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Will be set in main()
scheduler_instance: "SchedulerServer | None" = None


class SchedulerServer:
    """Central scheduler server managing distributed computational tasks."""

    def __init__(self, port: int = 8000) -> None:
        self.port = port
        self.scheduler = HEFTScheduler()
        self.master_seed = secrets.token_bytes(32)
        self.task_counter = 0
        self.is_running = False

        threading.Thread(target=self._monitoring_loop, daemon=True).start()

        logger.info("Scheduler server initialized on port %d", port)

    def _monitoring_loop(self) -> None:
        """Monitor tasks and trigger rescheduling."""
        while True:
            try:
                if hasattr(self.scheduler, "monitor_tasks"):
                    self.scheduler.monitor_tasks()
                time.sleep(10)
            except Exception as e:
                logger.error("Monitoring loop error: %s", e)
                time.sleep(10)

    def create_sample_tasks(self) -> Dict[str, CryptoTask]:
        """Create sample computational tasks for testing."""
        logger.info("Creating sample computational tasks")

        tasks = [
            # Mathematical computation tasks
            CryptoTask(
                task_id="math_prime_1",
                task_type="prime_generation",
                data={"range_start": 1000000, "range_end": 1010000, "count": 100},
                priority=TaskPriority.HIGH,
                computation_cost=45.0,
                timeout=180.0,
            ),
            CryptoTask(
                task_id="math_matrix_1",
                task_type="matrix_multiplication",
                data={"matrix_size": 1000, "iterations": 5},
                priority=TaskPriority.MEDIUM,
                computation_cost=30.0,
                dependencies=["math_prime_1"],
            ),
            CryptoTask(
                task_id="math_fibonacci_1",
                task_type="fibonacci_calculation",
                data={"n": 50000, "modulo": 1000000007},
                priority=TaskPriority.LOW,
                computation_cost=20.0,
            ),
            # Data processing tasks
            CryptoTask(
                task_id="data_sort_1",
                task_type="large_array_sort",
                data={"array_size": 1000000, "algorithm": "quicksort"},
                priority=TaskPriority.MEDIUM,
                computation_cost=25.0,
            ),
            CryptoTask(
                task_id="data_search_1",
                task_type="pattern_search",
                data={
                    "text_size": 10000000,
                    "pattern": "cryptography",
                    "algorithm": "kmp",
                },
                priority=TaskPriority.HIGH,
                computation_cost=35.0,
                dependencies=["data_sort_1"],
            ),
            CryptoTask(
                task_id="data_compress_1",
                task_type="data_compression",
                data={"data_size": 5000000, "algorithm": "lz77"},
                priority=TaskPriority.MEDIUM,
                computation_cost=40.0,
            ),
            # Image processing tasks
            CryptoTask(
                task_id="image_filter_1",
                task_type="image_processing",
                data={
                    "operation": "gaussian_blur",
                    "image_size": "1920x1080",
                    "kernel_size": 15,
                },
                priority=TaskPriority.HIGH,
                computation_cost=50.0,
                timeout=240.0,
            ),
            CryptoTask(
                task_id="image_detect_1",
                task_type="object_detection",
                data={"algorithm": "edge_detection", "image_count": 50},
                priority=TaskPriority.CRITICAL,
                computation_cost=60.0,
                dependencies=["image_filter_1"],
            ),
            # Network and web tasks
            CryptoTask(
                task_id="web_crawl_1",
                task_type="web_crawling",
                data={
                    "urls": ["https://example.com", "https://httpbin.org"],
                    "depth": 2,
                },
                priority=TaskPriority.MEDIUM,
                computation_cost=30.0,
                timeout=300.0,
            ),
            CryptoTask(
                task_id="api_fetch_1",
                task_type="api_data_fetch",
                data={
                    "endpoint": "https://jsonplaceholder.typicode.com/posts",
                    "count": 100,
                },
                priority=TaskPriority.LOW,
                computation_cost=15.0,
            ),
            # Machine learning tasks
            CryptoTask(
                task_id="ml_linear_regression_1",
                task_type="linear_regression",
                data={"dataset_size": 10000, "features": 20, "iterations": 1000},
                priority=TaskPriority.HIGH,
                computation_cost=55.0,
                dependencies=["data_sort_1", "math_matrix_1"],
            ),
            CryptoTask(
                task_id="ml_clustering_1",
                task_type="kmeans_clustering",
                data={"data_points": 50000, "clusters": 10, "dimensions": 5},
                priority=TaskPriority.MEDIUM,
                computation_cost=45.0,
                dependencies=["ml_linear_regression_1"],
            ),
            # File processing tasks
            CryptoTask(
                task_id="file_hash_1",
                task_type="file_hashing",
                data={"file_size": 100000000, "algorithm": "sha256"},
                priority=TaskPriority.LOW,
                computation_cost=25.0,
            ),
            CryptoTask(
                task_id="file_encrypt_1",
                task_type="file_encryption",
                data={"file_size": 50000000, "algorithm": "aes256"},
                priority=TaskPriority.HIGH,
                computation_cost=35.0,
                dependencies=["file_hash_1"],
            ),
            # Scientific computation tasks
            CryptoTask(
                task_id="science_monte_carlo_1",
                task_type="monte_carlo_simulation",
                data={"iterations": 1000000, "variables": 3},
                priority=TaskPriority.MEDIUM,
                computation_cost=40.0,
            ),
            CryptoTask(
                task_id="science_statistics_1",
                task_type="statistical_analysis",
                data={
                    "dataset_size": 100000,
                    "operations": ["mean", "std", "correlation"],
                },
                priority=TaskPriority.LOW,
                computation_cost=20.0,
                dependencies=["science_monte_carlo_1"],
            ),
        ]

        for task in tasks:
            if hasattr(self.scheduler, "add_task"):
                self.scheduler.add_task(task)
            elif hasattr(self.scheduler, "submit_task"):
                self.scheduler.submit_task(task)

        logger.info("Created %d computational tasks", len(tasks))
        return {t.task_id: t for t in tasks}

def _require_scheduler() -> SchedulerServer:
    if scheduler_instance is None:
        raise RuntimeError("Scheduler not initialized")
    return scheduler_instance

@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    """Receive heartbeat from nodes."""
    try:
        server = _require_scheduler()
        data = request.get_json(force=True)
        node_id = data["node_id"]

        if node_id not in server.scheduler.nodes:
            node_capability = NodeCapabilities(
                node_id=node_id,
                computation_power=1.0,
                max_concurrent_tasks=5,
                available_factors=["biometric", "password", "hardware_token"],
                task_completion_rate=0.95,
                communication_latency={},
            )
            if hasattr(server.scheduler, "add_node"):
                server.scheduler.add_node(node_capability)
            elif hasattr(server.scheduler, "register_node"):
                server.scheduler.register_node(node_id, node_capability)

            logger.info("New node registered: %s", node_id)
        else:
            node = server.scheduler.nodes[node_id]
            if hasattr(node, "update_status"):
                node.update_status(
                    current_load=data.get("current_load", 0),
                    last_heartbeat=time.time(),
                )

        return jsonify({"status": "success", "message": "Heartbeat received"}), 200

    except Exception as e:
        logger.error("Heartbeat processing error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/get_task/<node_id>", methods=["GET"])
def get_task(node_id: str):
    """Get next task assignment for a node."""
    try:
        server = _require_scheduler()
        task = None
        if hasattr(server.scheduler, "get_task_assignment"):
            task = server.scheduler.get_task_assignment(node_id)
        elif hasattr(server.scheduler, "get_next_task"):
            task = server.scheduler.get_next_task(node_id)

        if not task:
            return jsonify({}), 200

        task_dict: Dict[str, Any] = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "data": task.data,
            "priority": task.priority.value,
            "computation_cost": task.computation_cost,
            "communication_cost": task.communication_cost,
            "dependencies": task.dependencies,
            "timeout": task.timeout,
            "max_retries": task.max_retries,
            "created_at": task.created_at,
            "assigned_at": task.assigned_at,
            "assigned_node": task.assigned_node,
            "status": task.status.value,
            "retries": task.retries,
            "required_factors": task.required_factors,
        }

        logger.info("Assigned task %s to node %s", task.task_id, node_id)
        return jsonify(task_dict), 200

    except Exception as e:
        logger.error("Task assignment error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/update_task_status", methods=["POST"])
def update_task_status():
    """Update task execution status."""
    try:
        server = _require_scheduler()
        data = request.get_json(force=True)
        task_id = data["task_id"]
        status = TaskStatus(data["status"])
        execution_time = data.get("execution_time")

        if hasattr(server.scheduler, "update_task_status"):
            server.scheduler.update_task_status(task_id, status, execution_time)
        elif hasattr(server.scheduler, "report_task_completion"):
            server.scheduler.report_task_completion(task_id, status, execution_time)

        logger.info("Task %s status updated to %s", task_id, status.value)
        return jsonify({"status": "success", "message": "Status updated"}), 200

    except Exception as e:
        logger.error("Status update error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/statistics", methods=["GET"])
def get_statistics():
    """Get scheduling statistics."""
    try:
        server = _require_scheduler()
        if hasattr(server.scheduler, "get_scheduling_statistics"):
            stats = server.scheduler.get_scheduling_statistics()
        elif hasattr(server.scheduler, "get_stats"):
            stats = server.scheduler.get_stats()
        else:
            stats = {
                "total_tasks": len(getattr(server.scheduler, "tasks", {})),
                "total_nodes": len(getattr(server.scheduler, "nodes", {})),
            }

        return jsonify(stats), 200

    except Exception as e:
        logger.error("Statistics error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/nodes", methods=["GET"])
def get_nodes():
    """Get information about all nodes."""
    try:
        server = _require_scheduler()
        nodes_info: Dict[str, Any] = {}
        for node_id, node in getattr(server.scheduler, "nodes", {}).items():
            nodes_info[node_id] = {
                "computation_power": getattr(node, "computation_power", 1.0),
                "max_concurrent_tasks": getattr(node, "max_concurrent_tasks", 5),
                "available_factors": getattr(node, "available_factors", []),
                "task_completion_rate": getattr(node, "task_completion_rate", 0.95),
                "communication_latency": getattr(node, "communication_latency", {}),
            }

        return jsonify(nodes_info), 200

    except Exception as e:
        logger.error("Nodes info error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/tasks", methods=["GET"])
def get_tasks():
    """Get information about all tasks."""
    try:
        server = _require_scheduler()
        tasks_info: Dict[str, Any] = {}
        for task_id, task in getattr(server.scheduler, "tasks", {}).items():
            tasks_info[task_id] = {
                "task_type": task.task_type,
                "status": task.status.value,
                "assigned_node": task.assigned_node,
                "priority": task.priority.value,
                "computation_cost": task.computation_cost,
                "dependencies": task.dependencies,
                "retries": task.retries,
                "created_at": task.created_at,
                "assigned_at": task.assigned_at,
                "completed_at": task.completed_at,
            }

        return jsonify(tasks_info), 200

    except Exception as e:
        logger.error("Tasks info error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/reschedule", methods=["POST"])
def manual_reschedule():
    """Manually trigger rescheduling."""
    try:
        server = _require_scheduler()
        if hasattr(server.scheduler, "heft_schedule"):
            schedule = server.scheduler.heft_schedule()
        elif hasattr(server.scheduler, "reschedule"):
            schedule = server.scheduler.reschedule()
        else:
            schedule = {}

        logger.info("Manual rescheduling completed: %d assignments", len(schedule))
        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Rescheduled {len(schedule)} tasks",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error("Manual reschedule error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "healthy"}), 200

def main() -> None:
    global scheduler_instance
    setup_logging()
    parser = argparse.ArgumentParser(description="DPK Scheduler Server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    scheduler_instance = SchedulerServer(port=args.port)

    # Optional: pre-populate tasks and run initial schedule
    tasks = scheduler_instance.create_sample_tasks()
    if hasattr(scheduler_instance.scheduler, "heft_schedule"):
        schedule = scheduler_instance.scheduler.heft_schedule()
        logger.info("Initial HEFT schedule with %d assignments", len(schedule))
    elif hasattr(scheduler_instance.scheduler, "schedule"):
        schedule = scheduler_instance.scheduler.schedule()
        logger.info("Initial schedule with %d assignments", len(schedule))

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()