"""
HEFT (Heterogeneous Earliest Finish Time) Algorithm Implementation
Advanced task scheduling with dynamic load balancing and fault tolerance
"""
import time
import threading
import queue
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned" 
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REDISTRIBUTED = "redistributed"


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CryptoTask:
    """Cryptographic task with MFKDF requirements"""
    task_id: str
    task_type: str  # 'key_generation', 'share_distribution', 'hotp_verification', etc.
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: float = 300.0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_node: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    estimated_duration: float = 60.0
    required_factors: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    computation_cost: float = 1.0
    communication_cost: float = 0.1
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return (self.priority.value, self.created_at) > (other.priority.value, other.created_at)


@dataclass
class NodeCapabilities:
    """Node capabilities and current status"""
    node_id: str
    computation_power: float = 1.0  # Relative computation speed
    available_factors: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 5
    current_load: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True
    task_completion_rate: float = 0.95  # Success rate
    average_task_time: float = 60.0
    communication_latency: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_overloaded(self) -> bool:
        return self.current_load >= self.max_concurrent_tasks
    
    @property
    def load_factor(self) -> float:
        return self.current_load / self.max_concurrent_tasks
    
    def can_handle_task(self, task: CryptoTask) -> bool:
        """Check if node can handle the given task"""
        if not self.is_active or self.is_overloaded:
            return False
        
        # Check if node has required factors
        if task.required_factors:
            return all(factor in self.available_factors for factor in task.required_factors)
        
        return True
    
    def estimate_finish_time(self, task: CryptoTask) -> float:
        """Estimate when this node would finish the task"""
        computation_time = task.computation_cost / self.computation_power
        queue_delay = self.current_load * self.average_task_time * 0.1
        return time.time() + computation_time + queue_delay


class HEFTScheduler:
    """HEFT algorithm implementation for cryptographic task scheduling"""
    
    def __init__(self, heartbeat_interval: float = 10.0):
        self.nodes: Dict[str, NodeCapabilities] = {}
        self.tasks: Dict[str, CryptoTask] = {}
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Dict[str, List[str]] = {}  # task_id -> list of failed nodes
        
        self.heartbeat_interval = heartbeat_interval
        self.running = False
        self.scheduler_thread = None
        self.monitor_thread = None
        
        # Task execution callbacks
        self.task_handlers: Dict[str, Callable] = {}
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def register_node(self, node_id: str, capabilities: NodeCapabilities):
        """Register a new compute node"""
        self.nodes[node_id] = capabilities
        self.logger.info(f"Registered node {node_id} with {len(capabilities.available_factors)} factors")
    
    def unregister_node(self, node_id: str):
        """Unregister a compute node"""
        if node_id in self.nodes:
            # Reassign tasks from this node
            self._reassign_node_tasks(node_id)
            del self.nodes[node_id]
            self.logger.info(f"Unregistered node {node_id}")
    
    def submit_task(self, task: CryptoTask):
        """Submit a new task for scheduling"""
        self.tasks[task.task_id] = task
        self.task_queue.put(task)
        self.logger.info(f"Submitted task {task.task_id} of type {task.task_type}")
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type"""
        self.task_handlers[task_type] = handler
    
    def start(self):
        """Start the HEFT scheduler"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        
        self.scheduler_thread.start()
        self.monitor_thread.start()
        
        self.logger.info("HEFT Scheduler started")
    
    def stop(self):
        """Stop the HEFT scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.logger.info("HEFT Scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                if not self.task_queue.empty():
                    task = self.task_queue.get(timeout=1.0)
                    self._schedule_task(task)
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
    
    def _monitor_loop(self):
        """Monitor node health and task timeouts"""
        while self.running:
            try:
                self._check_node_health()
                self._check_task_timeouts()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
    
    def _schedule_task(self, task: CryptoTask):
        """Schedule a task using HEFT algorithm"""
        # Check dependencies
        if not self._dependencies_satisfied(task):
            # Re-queue the task for later
            self.task_queue.put(task)
            return
        
        # Find the best node using HEFT
        best_node = self._find_best_node_heft(task)
        
        if best_node is None:
            # No suitable node available, re-queue
            self.task_queue.put(task)
            self.logger.warning(f"No suitable node for task {task.task_id}, re-queuing")
            return
        
        # Assign task to node
        self._assign_task_to_node(task, best_node)
    
    def _find_best_node_heft(self, task: CryptoTask) -> Optional[str]:
        """Find the best node using HEFT algorithm"""
        if not self.nodes:
            return None
        
        best_node = None
        earliest_finish_time = float('inf')
        
        for node_id, node in self.nodes.items():
            if not node.can_handle_task(task):
                continue
            
            # Calculate finish time for this node
            finish_time = node.estimate_finish_time(task)
            
            # Add communication cost if task has dependencies
            if task.dependencies:
                communication_cost = self._calculate_communication_cost(task, node_id)
                finish_time += communication_cost
            
            if finish_time < earliest_finish_time:
                earliest_finish_time = finish_time
                best_node = node_id
        
        return best_node
    
    def _calculate_communication_cost(self, task: CryptoTask, node_id: str) -> float:
        """Calculate communication cost for task dependencies"""
        node = self.nodes[node_id]
        total_cost = 0.0
        
        for dep_task_id in task.dependencies:
            if dep_task_id in self.tasks:
                dep_task = self.tasks[dep_task_id]
                if dep_task.assigned_node and dep_task.assigned_node != node_id:
                    # Add latency between nodes
                    latency = node.communication_latency.get(dep_task.assigned_node, 0.1)
                    total_cost += latency * task.communication_cost
        
        return total_cost
    
    def _assign_task_to_node(self, task: CryptoTask, node_id: str):
        """Assign task to a specific node"""
        node = self.nodes[node_id]
        
        task.assigned_node = node_id
        task.assigned_at = time.time()
        task.status = TaskStatus.ASSIGNED
        
        node.current_load += 1
        
        # Execute task asynchronously
        threading.Thread(
            target=self._execute_task,
            args=(task, node_id),
            daemon=True
        ).start()
        
        self.logger.info(f"Assigned task {task.task_id} to node {node_id}")
    
    def _execute_task(self, task: CryptoTask, node_id: str):
        """Execute a task on a specific node"""
        try:
            task.status = TaskStatus.IN_PROGRESS
            
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type {task.task_type}")
            
            # Execute the task
            start_time = time.time()
            result = handler(task, node_id)
            execution_time = time.time() - start_time
            
            # Update node statistics
            node = self.nodes[node_id]
            node.average_task_time = (node.average_task_time * 0.9 + execution_time * 0.1)
            node.current_load -= 1
            
            # Mark task as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            self.completed_tasks.add(task.task_id)
            
            self.logger.info(f"Task {task.task_id} completed on node {node_id} in {execution_time:.2f}s")
            
        except Exception as e:
            self._handle_task_failure(task, node_id, str(e))
    
    def _handle_task_failure(self, task: CryptoTask, node_id: str, error: str):
        """Handle task execution failure"""
        self.logger.error(f"Task {task.task_id} failed on node {node_id}: {error}")
        
        # Update node load
        if node_id in self.nodes:
            self.nodes[node_id].current_load -= 1
        
        # Track failed attempts
        if task.task_id not in self.failed_tasks:
            self.failed_tasks[task.task_id] = []
        self.failed_tasks[task.task_id].append(node_id)
        
        # Retry or fail permanently
        task.retries += 1
        if task.retries < task.max_retries:
            task.status = TaskStatus.PENDING
            task.assigned_node = None
            task.assigned_at = None
            self.task_queue.put(task)
            self.logger.info(f"Re-queuing task {task.task_id} (retry {task.retries}/{task.max_retries})")
        else:
            task.status = TaskStatus.FAILED
            self.logger.error(f"Task {task.task_id} permanently failed after {task.retries} retries")
    
    def _dependencies_satisfied(self, task: CryptoTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
        return True
    
    def _check_node_health(self):
        """Check health of all nodes"""
        current_time = time.time()
        dead_nodes = []
        
        for node_id, node in self.nodes.items():
            if current_time - node.last_heartbeat > self.heartbeat_interval * 3:
                # Node is unresponsive
                node.is_active = False
                dead_nodes.append(node_id)
        
        # Reassign tasks from dead nodes
        for node_id in dead_nodes:
            self._reassign_node_tasks(node_id)
    
    def _check_task_timeouts(self):
        """Check for timed out tasks"""
        current_time = time.time()
        
        for task in self.tasks.values():
            if (task.status == TaskStatus.IN_PROGRESS and 
                task.assigned_at and 
                current_time - task.assigned_at > task.timeout):
                
                self.logger.warning(f"Task {task.task_id} timed out on node {task.assigned_node}")
                self._handle_task_failure(task, task.assigned_node, "Timeout")
    
    def _reassign_node_tasks(self, node_id: str):
        """Reassign all tasks from a failed node"""
        for task in self.tasks.values():
            if task.assigned_node == node_id and task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
                task.status = TaskStatus.REDISTRIBUTED
                task.assigned_node = None
                task.assigned_at = None
                self.task_queue.put(task)
                self.logger.info(f"Reassigned task {task.task_id} from failed node {node_id}")
    
    def update_node_heartbeat(self, node_id: str):
        """Update node heartbeat"""
        if node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = time.time()
            self.nodes[node_id].is_active = True
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            'nodes': len(self.nodes),
            'active_nodes': sum(1 for node in self.nodes.values() if node.is_active),
            'pending_tasks': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_tasks': len(self.tasks)
        }
    
    def redistribute_load(self):
        """Redistribute load across nodes"""
        # Find overloaded nodes
        overloaded_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.is_overloaded and node.is_active
        ]
        
        # Find underutilized nodes
        available_nodes = [
            node_id for node_id, node in self.nodes.items()
            if not node.is_overloaded and node.is_active
        ]
        
        if not overloaded_nodes or not available_nodes:
            return
        
        # Move tasks from overloaded to available nodes
        for overloaded_node_id in overloaded_nodes:
            # Find pending tasks assigned to this node
            tasks_to_move = [
                task for task in self.tasks.values()
                if (task.assigned_node == overloaded_node_id and 
                    task.status == TaskStatus.ASSIGNED)
            ]
            
            for task in tasks_to_move:
                # Find best available node
                best_node = self._find_best_node_heft(task)
                if best_node and best_node in available_nodes:
                    # Reassign task
                    self.nodes[overloaded_node_id].current_load -= 1
                    task.assigned_node = best_node
                    self.nodes[best_node].current_load += 1
                    
                    self.logger.info(f"Redistributed task {task.task_id} from {overloaded_node_id} to {best_node}")