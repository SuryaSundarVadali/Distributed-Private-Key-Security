"""
Central Scheduler Server
Coordinates task distribution using HEFT algorithm
"""
import time
import threading
import logging
from flask import Flask, request, jsonify
import sys

from task_scheduler import HEFTScheduler, CryptoTask, TaskStatus, TaskPriority, NodeCapabilities
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class SchedulerServer:
    """Central scheduler server managing distributed computational tasks"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.scheduler = HEFTScheduler()
        self.master_seed = secrets.token_bytes(32)
        self.task_counter = 0
        self.is_running = False
        
        # Start monitoring thread
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        
        logger.info(f"Scheduler server initialized on port {port}")
    
    def _monitoring_loop(self):
        """Monitor tasks and trigger rescheduling"""
        while True:
            try:
                if hasattr(self.scheduler, 'monitor_tasks'):
                    self.scheduler.monitor_tasks()
                time.sleep(10)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def create_sample_tasks(self):
        """Create sample computational tasks for testing"""
        logger.info("Creating sample computational tasks")
        
        tasks = [
            # Mathematical computation tasks
            CryptoTask(
                task_id="math_prime_1",
                task_type="prime_generation",
                data={"range_start": 1000000, "range_end": 1010000, "count": 100},
                priority=TaskPriority.HIGH,
                computation_cost=45.0,
                timeout=180.0
            ),
            CryptoTask(
                task_id="math_matrix_1", 
                task_type="matrix_multiplication",
                data={"matrix_size": 1000, "iterations": 5},
                priority=TaskPriority.MEDIUM,
                computation_cost=30.0,
                dependencies=["math_prime_1"]
            ),
            CryptoTask(
                task_id="math_fibonacci_1",
                task_type="fibonacci_calculation",
                data={"n": 50000, "modulo": 1000000007},
                priority=TaskPriority.LOW,
                computation_cost=20.0
            ),
            
            # Data processing tasks
            CryptoTask(
                task_id="data_sort_1",
                task_type="large_array_sort",
                data={"array_size": 1000000, "algorithm": "quicksort"},
                priority=TaskPriority.MEDIUM,
                computation_cost=25.0
            ),
            CryptoTask(
                task_id="data_search_1",
                task_type="pattern_search",
                data={"text_size": 10000000, "pattern": "cryptography", "algorithm": "kmp"},
                priority=TaskPriority.HIGH,
                computation_cost=35.0,
                dependencies=["data_sort_1"]
            ),
            CryptoTask(
                task_id="data_compress_1",
                task_type="data_compression",
                data={"data_size": 5000000, "algorithm": "lz77"},
                priority=TaskPriority.MEDIUM,
                computation_cost=40.0
            ),
            
            # Image processing tasks
            CryptoTask(
                task_id="image_filter_1",
                task_type="image_processing",
                data={"operation": "gaussian_blur", "image_size": "1920x1080", "kernel_size": 15},
                priority=TaskPriority.HIGH,
                computation_cost=50.0,
                timeout=240.0
            ),
            CryptoTask(
                task_id="image_detect_1",
                task_type="object_detection",
                data={"algorithm": "edge_detection", "image_count": 50},
                priority=TaskPriority.CRITICAL,
                computation_cost=60.0,
                dependencies=["image_filter_1"]
            ),
            
            # Network and web tasks
            CryptoTask(
                task_id="web_crawl_1",
                task_type="web_crawling",
                data={"urls": ["https://example.com", "https://httpbin.org"], "depth": 2},
                priority=TaskPriority.MEDIUM,
                computation_cost=30.0,
                timeout=300.0
            ),
            CryptoTask(
                task_id="api_fetch_1",
                task_type="api_data_fetch",
                data={"endpoint": "https://jsonplaceholder.typicode.com/posts", "count": 100},
                priority=TaskPriority.LOW,
                computation_cost=15.0
            ),
            
            # Machine learning tasks
            CryptoTask(
                task_id="ml_linear_regression_1",
                task_type="linear_regression",
                data={"dataset_size": 10000, "features": 20, "iterations": 1000},
                priority=TaskPriority.HIGH,
                computation_cost=55.0,
                dependencies=["data_sort_1", "math_matrix_1"]
            ),
            CryptoTask(
                task_id="ml_clustering_1",
                task_type="kmeans_clustering",
                data={"data_points": 50000, "clusters": 10, "dimensions": 5},
                priority=TaskPriority.MEDIUM,
                computation_cost=45.0,
                dependencies=["ml_linear_regression_1"]
            ),
            
            # File processing tasks
            CryptoTask(
                task_id="file_hash_1",
                task_type="file_hashing",
                data={"file_size": 100000000, "algorithm": "sha256"},
                priority=TaskPriority.LOW,
                computation_cost=25.0
            ),
            CryptoTask(
                task_id="file_encrypt_1",
                task_type="file_encryption",
                data={"file_size": 50000000, "algorithm": "aes256"},
                priority=TaskPriority.HIGH,
                computation_cost=35.0,
                dependencies=["file_hash_1"]
            ),
            
            # Scientific computation tasks
            CryptoTask(
                task_id="science_monte_carlo_1",
                task_type="monte_carlo_simulation",
                data={"iterations": 1000000, "variables": 3},
                priority=TaskPriority.MEDIUM,
                computation_cost=40.0
            ),
            CryptoTask(
                task_id="science_statistics_1",
                task_type="statistical_analysis",
                data={"dataset_size": 100000, "operations": ["mean", "std", "correlation"]},
                priority=TaskPriority.LOW,
                computation_cost=20.0,
                dependencies=["science_monte_carlo_1"]
            )
        ]
        
        # Add tasks to scheduler using correct method
        for task in tasks:
            if hasattr(self.scheduler, 'add_task'):
                self.scheduler.add_task(task)
            elif hasattr(self.scheduler, 'submit_task'):
                self.scheduler.submit_task(task)
        
        logger.info(f"Created {len(tasks)} computational tasks")
        return tasks
    
    def start(self):
        """Start the scheduler server"""
        self.is_running = True
        
        # Create sample tasks
        self.create_sample_tasks()
        
        # Run HEFT scheduling
        if hasattr(self.scheduler, 'heft_schedule'):
            schedule = self.scheduler.heft_schedule()
        elif hasattr(self.scheduler, 'schedule'):
            schedule = self.scheduler.schedule()
        else:
            schedule = {}
            
        logger.info(f"Initial HEFT schedule created with {len(schedule)} task assignments")
        
        # Start Flask server
        app.run(host='0.0.0.0', port=self.port, debug=False)


# Flask routes
scheduler_instance = SchedulerServer()

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """Receive heartbeat from nodes"""
    try:
        data = request.json
        node_id = data['node_id']
        
        # Update or create node capability
        if node_id not in scheduler_instance.scheduler.nodes:
            node_capability = NodeCapabilities(
                node_id=node_id,
                computation_power=1.0,
                max_concurrent_tasks=5,
                available_factors=['biometric', 'password', 'hardware_token'],
                task_completion_rate=0.95,
                communication_latency={}
            )
            
            if hasattr(scheduler_instance.scheduler, 'add_node'):
                scheduler_instance.scheduler.add_node(node_capability)
            elif hasattr(scheduler_instance.scheduler, 'register_node'):
                scheduler_instance.scheduler.register_node(node_id, node_capability)
                
            logger.info(f"New node registered: {node_id}")
        else:
            # Update existing node heartbeat
            node = scheduler_instance.scheduler.nodes[node_id]
            if hasattr(node, 'update_status'):
                node.update_status(
                    current_load=data.get('current_load', 0),
                    last_heartbeat=time.time()
                )
        
        return jsonify({"status": "success", "message": "Heartbeat received"}), 200
        
    except Exception as e:
        logger.error(f"Heartbeat processing error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get_task/<node_id>', methods=['GET'])
def get_task(node_id: str):
    """Get next task assignment for a node"""
    try:
        # Use correct method name
        task = None
        if hasattr(scheduler_instance.scheduler, 'get_task_assignment'):
            task = scheduler_instance.scheduler.get_task_assignment(node_id)
        elif hasattr(scheduler_instance.scheduler, 'get_next_task'):
            task = scheduler_instance.scheduler.get_next_task(node_id)
        
        if task:
            # Convert task to dictionary
            task_dict = {
                'task_id': task.task_id,
                'task_type': task.task_type,
                'data': task.data,
                'priority': task.priority.value,
                'computation_cost': task.computation_cost,
                'communication_cost': task.communication_cost,
                'dependencies': task.dependencies,
                'timeout': task.timeout,
                'max_retries': task.max_retries,
                'created_at': task.created_at,
                'assigned_at': task.assigned_at,
                'assigned_node': task.assigned_node,
                'status': task.status.value,
                'retries': task.retries,
                'required_factors': task.required_factors
            }
            
            logger.info(f"Assigned task {task.task_id} to node {node_id}")
            return jsonify(task_dict), 200
        else:
            return jsonify({}), 200
            
    except Exception as e:
        logger.error(f"Task assignment error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/update_task_status', methods=['POST'])
def update_task_status():
    """Update task execution status"""
    try:
        data = request.json
        task_id = data['task_id']
        status = TaskStatus(data['status'])
        execution_time = data.get('execution_time')
        
        # Use correct method name
        if hasattr(scheduler_instance.scheduler, 'update_task_status'):
            scheduler_instance.scheduler.update_task_status(task_id, status, execution_time)
        elif hasattr(scheduler_instance.scheduler, 'report_task_completion'):
            scheduler_instance.scheduler.report_task_completion(task_id, status, execution_time)
        
        logger.info(f"Task {task_id} status updated to {status.value}")
        return jsonify({"status": "success", "message": "Status updated"}), 200
        
    except Exception as e:
        logger.error(f"Status update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get scheduling statistics"""
    try:
        if hasattr(scheduler_instance.scheduler, 'get_scheduling_statistics'):
            stats = scheduler_instance.scheduler.get_scheduling_statistics()
        elif hasattr(scheduler_instance.scheduler, 'get_stats'):
            stats = scheduler_instance.scheduler.get_stats()
        else:
            stats = {
                'total_tasks': len(scheduler_instance.scheduler.tasks) if hasattr(scheduler_instance.scheduler, 'tasks') else 0,
                'total_nodes': len(scheduler_instance.scheduler.nodes) if hasattr(scheduler_instance.scheduler, 'nodes') else 0
            }
            
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/nodes', methods=['GET'])
def get_nodes():
    """Get information about all nodes"""
    try:
        nodes_info = {}
        if hasattr(scheduler_instance.scheduler, 'nodes'):
            for node_id, node in scheduler_instance.scheduler.nodes.items():
                nodes_info[node_id] = {
                    'computation_power': getattr(node, 'computation_power', 1.0),
                    'max_concurrent_tasks': getattr(node, 'max_concurrent_tasks', 5),
                    'available_factors': getattr(node, 'available_factors', []),
                    'task_completion_rate': getattr(node, 'task_completion_rate', 0.95),
                    'communication_latency': getattr(node, 'communication_latency', {})
                }
        
        return jsonify(nodes_info), 200
        
    except Exception as e:
        logger.error(f"Nodes info error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/tasks', methods=['GET'])
def get_tasks():
    """Get information about all tasks"""
    try:
        tasks_info = {}
        if hasattr(scheduler_instance.scheduler, 'tasks'):
            for task_id, task in scheduler_instance.scheduler.tasks.items():
                tasks_info[task_id] = {
                    'task_type': task.task_type,
                    'status': task.status.value,
                    'assigned_node': task.assigned_node,
                    'priority': task.priority.value,
                    'computation_cost': task.computation_cost,
                    'dependencies': task.dependencies,
                    'retries': task.retries,
                    'created_at': task.created_at,
                    'assigned_at': task.assigned_at,
                    'completed_at': task.completed_at
                }
        
        return jsonify(tasks_info), 200
        
    except Exception as e:
        logger.error(f"Tasks info error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/reschedule', methods=['POST'])
def manual_reschedule():
    """Manually trigger rescheduling"""
    try:
        if hasattr(scheduler_instance.scheduler, 'heft_schedule'):
            schedule = scheduler_instance.scheduler.heft_schedule()
        elif hasattr(scheduler_instance.scheduler, 'reschedule'):
            schedule = scheduler_instance.scheduler.reschedule()
        else:
            schedule = {}
            
        logger.info(f"Manual rescheduling completed: {len(schedule)} assignments")
        return jsonify({
            "status": "success", 
            "message": f"Rescheduled {len(schedule)} tasks"
        }), 200
        
    except Exception as e:
        logger.error(f"Manual reschedule error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    scheduler_instance = SchedulerServer(port)
    scheduler_instance.start()