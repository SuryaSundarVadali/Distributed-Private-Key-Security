#!/usr/bin/env python3
"""
Local execution script for testing the distributed cryptographic system
"""
import os
import sys
import time
import threading
import subprocess
import signal
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalSystemRunner:
    """Run the entire distributed system locally for development and testing"""
    
    def __init__(self, num_nodes=5, threshold=3, scheduler_port=8000):
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.scheduler_port = scheduler_port
        self.processes = []
        self.running = False
        
        # Ensure logs directory exists
        self.logs_dir = Path(__file__).parent.parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)
    
    def start_scheduler(self):
        """Start the scheduler server"""
        logger.info(f"Starting scheduler on port {self.scheduler_port}")
        
        cmd = [
            sys.executable, "scheduler_server.py", str(self.scheduler_port)
        ]
        
        with open(self.logs_dir / "scheduler.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent.parent
            )
        
        self.processes.append(("scheduler", process))
        return process
    
    def start_nodes(self):
        """Start worker nodes"""
        logger.info(f"Starting {self.num_nodes} worker nodes")
        
        for i in range(self.num_nodes):
            node_id = f"node_{i}"
            logger.info(f"Starting {node_id}")
            
            cmd = [sys.executable, "distributed_node.py", node_id]
            
            with open(self.logs_dir / f"{node_id}.log", "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=Path(__file__).parent.parent,
                    env={
                        **os.environ,
                        "NODE_ID": node_id,
                        "SCHEDULER_HOST": "localhost",
                        "SCHEDULER_PORT": str(self.scheduler_port)
                    }
                )
            
            self.processes.append((node_id, process))
            time.sleep(1)  # Stagger node startup
    
    def start_executor(self):
        """Start the main executor"""
        logger.info("Starting main executor")
        
        cmd = [sys.executable, "enhanced_distributed_executor.py"]
        
        with open(self.logs_dir / "executor.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent.parent,
                env={
                    **os.environ,
                    "NUM_NODES": str(self.num_nodes),
                    "THRESHOLD": str(self.threshold),
                    "SCHEDULER_PORT": str(self.scheduler_port)
                }
            )
        
        self.processes.append(("executor", process))
        return process
    
    def run(self):
        """Run the complete system"""
        try:
            self.running = True
            
            # Start scheduler
            scheduler_process = self.start_scheduler()
            time.sleep(3)  # Wait for scheduler to start
            
            # Start nodes
            self.start_nodes()
            time.sleep(5)  # Wait for nodes to register
            
            # Start executor
            executor_process = self.start_executor()
            
            # Start monitoring in background
            monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
            monitor_thread.start()
            
            logger.info("System started successfully")
            logger.info(f"Scheduler: http://localhost:{self.scheduler_port}")
            logger.info(f"Logs directory: {self.logs_dir}")
            logger.info("Press Ctrl+C to stop")
            
            # Wait for executor to complete or user interrupt
            try:
                executor_process.wait()
                logger.info("Executor completed successfully")
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
            
        except Exception as e:
            logger.error(f"Error running system: {e}")
        finally:
            self.stop()
    
    def _monitor_processes(self):
        """Monitor running processes"""
        while self.running:
            for name, process in self.processes:
                if process.poll() is not None:
                    logger.error(f"Process {name} exited with code {process.returncode}")
            time.sleep(5)
    
    def stop(self):
        """Stop all processes"""
        logger.info("Stopping all processes...")
        self.running = False
        
        for name, process in self.processes:
            try:
                if process.poll() is None:
                    logger.info(f"Terminating {name}")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing {name}")
                        process.kill()
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        self.processes.clear()
        logger.info("All processes stopped")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run distributed crypto system locally')
    parser.add_argument('--nodes', type=int, default=5, help='Number of worker nodes')
    parser.add_argument('--threshold', type=int, default=3, help='Secret sharing threshold')
    parser.add_argument('--port', type=int, default=8000, help='Scheduler port')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.threshold > args.nodes:
        logger.error(f"Threshold ({args.threshold}) cannot be greater than nodes ({args.nodes})")
        sys.exit(1)
    
    # Create and run system
    runner = LocalSystemRunner(
        num_nodes=args.nodes,
        threshold=args.threshold,
        scheduler_port=args.port
    )
    
    # Handle signals
    def signal_handler(signum, frame):
        logger.info("Received signal, shutting down...")
        runner.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the system
    runner.run()


if __name__ == "__main__":
    main()