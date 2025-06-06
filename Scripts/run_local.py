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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalSystemRunner:
    def __init__(self, num_nodes=5, threshold=3, scheduler_port=8000):
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.scheduler_port = scheduler_port
        self.processes = []
        self.running = False
    
    def start_scheduler(self):
        """Start the scheduler server"""
        logger.info("Starting scheduler server...")
        cmd = [sys.executable, "scheduler_server.py", str(self.scheduler_port)]
        process = subprocess.Popen(cmd, env=os.environ.copy())
        self.processes.append(('scheduler', process))
        return process
    
    def start_nodes(self):
        """Start worker nodes"""
        logger.info(f"Starting {self.num_nodes} worker nodes...")
        
        for i in range(self.num_nodes):
            node_env = os.environ.copy()
            node_env['NODE_ID'] = f'node_{i}'
            node_env['SCHEDULER_HOST'] = 'localhost'
            node_env['SCHEDULER_PORT'] = str(self.scheduler_port)
            
            cmd = [sys.executable, "distributed_node.py"]
            process = subprocess.Popen(cmd, env=node_env)
            self.processes.append((f'node_{i}', process))
            
            # Small delay between node starts
            time.sleep(1)
    
    def start_executor(self):
        """Start the main executor"""
        logger.info("Starting distributed executor...")
        time.sleep(5)  # Wait for nodes to register
        
        cmd = [sys.executable, "enhanced_distributed_executor.py"]
        process = subprocess.Popen(cmd)
        self.processes.append(('executor', process))
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
            
            # Wait for executor to complete
            executor_process.wait()
            
            logger.info("System execution completed")
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop all processes"""
        self.running = False
        logger.info("Stopping all processes...")
        
        for name, process in reversed(self.processes):
            try:
                logger.info(f"Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {name}...")
                process.kill()
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        self.processes.clear()

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run distributed crypto system locally')
    parser.add_argument('--nodes', type=int, default=5, help='Number of worker nodes')
    parser.add_argument('--threshold', type=int, default=3, help='Secret sharing threshold')
    parser.add_argument('--port', type=int, default=8000, help='Scheduler port')
    
    args = parser.parse_args()
    
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