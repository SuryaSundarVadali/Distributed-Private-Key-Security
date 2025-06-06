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
    """Central scheduler server managing distributed cryptographic tasks"""
    
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
                self.scheduler.monitor_and_reschedule()
                time.sleep(10)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def create_sample_tasks(self):
        """Create sample cryptographic tasks for testing"""
        logger.info("Creating sample cryptographic tasks")
        
        tasks = [
            # Key generation tasks
            CryptoTask(
                task_id="key_gen_1",
                task_type="key_generation",
                data={"key_size": 2048},
                priority=TaskPriority.HIGH,
                computation_cost=30.0,
                required_factors=['biometric', 'password', 'hardware_token']
            ),
            CryptoTask(
                task_id="key_gen_2", 
                task_type="key_generation",
                data={"key_size": 4096},
                priority=TaskPriority.CRITICAL,
                computation_cost=60.0,
                dependencies=["key_gen_1"],
                required_factors=['biometric', 'password', 'hardware_token']
            ),
            
            # Secret sharing tasks
            CryptoTask(
                task_id="secret_share_1",
                task_type="secret_sharing",
                data={"threshold": 3, "num_shares": 5},
                priority=TaskPriority.HIGH,
                computation_cost=15.0,
                dependencies=["key_gen_1"]
            ),
            CryptoTask(
                task_id="secret_share_2",
                task_type="secret_sharing", 
                data={"threshold": 4, "num_shares": 7},
                priority=TaskPriority.HIGH,
                computation_cost=20.0,
                dependencies=["key_gen_2"]
            ),
            
            # HOTP generation tasks
            CryptoTask(
                task_id="hotp_gen_1",
                task_type="hotp_generation",
                data={"num_tokens": 10},
                priority=TaskPriority.MEDIUM,
                computation_cost=5.0
            ),
            CryptoTask(
                task_id="hotp_gen_2",
                task_type="hotp_generation",
                data={"num_tokens": 20},
                priority=TaskPriority.MEDIUM,
                computation_cost=8.0
            ),
            
            # Merkle verification tasks
            CryptoTask(
                task_id="merkle_verify_1",
                task_type="merkle_verification",
                data={"num_blocks": 8},
                priority=TaskPriority.MEDIUM,
                computation_cost=10.0,
                dependencies=["secret_share_1"]
            ),
            CryptoTask(
                task_id="merkle_verify_2",
                task_type="merkle_verification",
                data={"num_blocks": 16},
                priority=TaskPriority.LOW,
                computation_cost=15.0,
                dependencies=["secret_share_2"]
            ),
            
            # MPC computation tasks
            CryptoTask(
                task_id="mpc_compute_1",
                task_type="mpc_computation",
                data={"num_participants": 3, "threshold": 2},
                priority=TaskPriority.HIGH,
                computation_cost=25.0,
                dependencies=["hotp_gen_1", "merkle_verify_1"]
            ),
            CryptoTask(
                task_id="mpc_compute_2",
                task_type="mpc_computation",
                data={"num_participants": 5, "threshold": 3},
                priority=TaskPriority.HIGH,
                computation_cost=40.0,
                dependencies=["hotp_gen_2", "merkle_verify_2"]
            ),
            
            # Reconstruction tasks
            CryptoTask(
                task_id="reconstruct_1",
                task_type="reconstruction",
                data={"threshold": 3},
                priority=TaskPriority.CRITICAL,
                computation_cost=20.0,
                dependencies=["mpc_compute_1"]
            ),
            CryptoTask(
                task_id="reconstruct_2",
                task_type="reconstruction",
                data={"threshold": 4},
                priority=TaskPriority.CRITICAL,
                computation_cost=25.0,
                dependencies=["mpc_compute_2"]
            )
        ]
        
        # Add tasks to scheduler
        for task in tasks:
            self.scheduler.add_task(task)
        
        logger.info(f"Created {len(tasks)} sample tasks")
        return tasks
    
    def start(self):
        """Start the scheduler server"""
        self.is_running = True
        
        # Create sample tasks
        self.create_sample_tasks()
        
        # Run HEFT scheduling
        schedule = self.scheduler.heft_schedule()
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
                processing_speed=1.0,
                max_capacity=5,
                current_load=data.get('current_load', 0),
                specialized_tasks=[],
                available_factors=['biometric', 'password', 'hardware_token']
            )
            scheduler_instance.scheduler.add_node(node_capability)
            logger.info(f"New node registered: {node_id}")
        else:
            # Update existing node
            node = scheduler_instance.scheduler.nodes[node_id]
            node.current_load = data.get('current_load', 0)
            node.last_heartbeat = time.time()
        
        return jsonify({"status": "success", "message": "Heartbeat received"}), 200
        
    except Exception as e:
        logger.error(f"Heartbeat processing error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get_task/<node_id>', methods=['GET'])
def get_task(node_id: str):
    """Get next task assignment for a node"""
    try:
        task = scheduler_instance.scheduler.get_task_assignment(node_id)
        
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
        
        scheduler_instance.scheduler.update_task_status(task_id, status, execution_time)
        
        logger.info(f"Task {task_id} status updated to {status.value}")
        return jsonify({"status": "success", "message": "Status updated"}), 200
        
    except Exception as e:
        logger.error(f"Status update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get scheduling statistics"""
    try:
        stats = scheduler_instance.scheduler.get_scheduling_statistics()
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/nodes', methods=['GET'])
def get_nodes():
    """Get information about all nodes"""
    try:
        nodes_info = {}
        for node_id, node in scheduler_instance.scheduler.nodes.items():
            nodes_info[node_id] = {
                'processing_speed': node.processing_speed,
                'max_capacity': node.max_capacity,
                'current_load': node.current_load,
                'load_factor': node.load_factor,
                'specialized_tasks': node.specialized_tasks,
                'available_factors': node.available_factors,
                'last_heartbeat': node.last_heartbeat,
                'average_performance': node.average_performance
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
        schedule = scheduler_instance.scheduler.heft_schedule()
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