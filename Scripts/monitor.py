#!/usr/bin/env python3
"""
System monitoring script for the distributed cryptographic system
"""
import time
import requests
import json
from typing import Dict, Any
import argparse

class SystemMonitor:
    def __init__(self, scheduler_host='localhost', scheduler_port=8000):
        self.base_url = f"http://{scheduler_host}:{scheduler_port}"
        self.running = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            response = requests.get(f"{self.base_url}/statistics", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def get_nodes(self) -> Dict[str, Any]:
        """Get node information"""
        try:
            response = requests.get(f"{self.base_url}/nodes", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def get_tasks(self) -> Dict[str, Any]:
        """Get task information"""
        try:
            response = requests.get(f"{self.base_url}/tasks", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def print_dashboard(self):
        """Print monitoring dashboard"""
        stats = self.get_statistics()
        nodes = self.get_nodes()
        tasks = self.get_tasks()
        
        # Clear screen
        print("\033[2J\033[H")
        
        # Header
        print("=" * 80)
        print("DISTRIBUTED CRYPTOGRAPHIC SYSTEM MONITOR")
        print("=" * 80)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System Statistics
        print("SYSTEM STATISTICS:")
        print(f"  Total Nodes: {stats.get('nodes', 0)}")
        print(f"  Active Nodes: {stats.get('active_nodes', 0)}")
        print(f"  Total Tasks: {stats.get('total_tasks', 0)}")
        print(f"  Completed Tasks: {stats.get('completed_tasks', 0)}")
        print(f"  Failed Tasks: {stats.get('failed_tasks', 0)}")
        print(f"  Pending Tasks: {stats.get('pending_tasks', 0)}")
        print()
        
        # Node Information
        print("NODE STATUS:")
        print(f"{'Node ID':<15} {'Status':<10} {'Load':<8} {'Speed':<8} {'Tasks':<10}")
        print("-" * 60)
        
        for node_id, node_info in nodes.items():
            status = "Active" if node_info.get('last_heartbeat', 0) > time.time() - 30 else "Inactive"
            load = f"{node_info.get('load_factor', 0):.2f}"
            speed = f"{node_info.get('processing_speed', 0):.2f}"
            capacity = f"{node_info.get('current_load', 0)}/{node_info.get('max_capacity', 0)}"
            
            print(f"{node_id:<15} {status:<10} {load:<8} {speed:<8} {capacity:<10}")
        
        print()
        
        # Task Information
        print("TASK STATUS:")
        task_status_counts = {}
        for task_id, task_info in tasks.items():
            status = task_info.get('status', 'unknown')
            task_status_counts[status] = task_status_counts.get(status, 0) + 1
        
        for status, count in task_status_counts.items():
            print(f"  {status.capitalize()}: {count}")
        
        print()
        
        # Performance Metrics
        if stats.get('average_completion_time'):
            print("PERFORMANCE METRICS:")
            print(f"  Average Completion Time: {stats['average_completion_time']:.2f}s")
            print(f"  System Throughput: {stats.get('completed_tasks', 0) / max(1, time.time() - stats.get('start_time', time.time())):.2f} tasks/s")
        
        print("=" * 80)
        print("Press Ctrl+C to stop monitoring")
    
    def monitor(self, interval=5):
        """Start monitoring loop"""
        self.running = True
        try:
            while self.running:
                self.print_dashboard()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.running = False
            print("\nMonitoring stopped.")

def main():
    parser = argparse.ArgumentParser(description='Monitor distributed crypto system')
    parser.add_argument('--host', default='localhost', help='Scheduler hostname')
    parser.add_argument('--port', type=int, default=8000, help='Scheduler port')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(args.host, args.port)
    monitor.monitor(args.interval)

if __name__ == "__main__":
    main()