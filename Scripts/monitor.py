#!/usr/bin/env python3
"""
System monitoring script for the distributed cryptographic system
"""
import time
import requests
import json
import argparse
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

console = Console()


class SystemMonitor:
    """Monitor the distributed cryptographic system"""
    
    def __init__(self, scheduler_host: str = "localhost", scheduler_port: int = 8000):
        self.scheduler_host = scheduler_host
        self.scheduler_port = scheduler_port
        self.base_url = f"http://{scheduler_host}:{scheduler_port}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            response = requests.get(f"{self.base_url}/statistics", timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error getting statistics: {e}[/red]")
        return {}
    
    def get_nodes(self) -> Dict[str, Any]:
        """Get node information"""
        try:
            response = requests.get(f"{self.base_url}/nodes", timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error getting nodes: {e}[/red]")
        return {}
    
    def get_tasks(self) -> Dict[str, Any]:
        """Get task information"""
        try:
            response = requests.get(f"{self.base_url}/tasks", timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error getting tasks: {e}[/red]")
        return {}
    
    def create_dashboard(self) -> Layout:
        """Create the monitoring dashboard layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="stats"),
            Layout(name="nodes")
        )
        
        layout["right"].split_column(
            Layout(name="tasks"),
            Layout(name="performance")
        )
        
        return layout
    
    def update_dashboard(self, layout: Layout):
        """Update dashboard with current data"""
        stats = self.get_statistics()
        nodes = self.get_nodes()
        tasks = self.get_tasks()
        
        # Header
        header_text = f"[bold blue]Distributed Cryptographic System Monitor[/bold blue]\n"
        header_text += f"Scheduler: {self.base_url} | "
        header_text += f"Time: {time.strftime('%H:%M:%S')}"
        layout["header"].update(Panel(header_text, style="blue"))
        
        # System Statistics
        stats_table = Table(title="System Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")
        
        stats_table.add_row("Total Nodes", str(stats.get('nodes', 0)))
        stats_table.add_row("Active Nodes", str(stats.get('active_nodes', 0)))
        stats_table.add_row("Total Tasks", str(stats.get('total_tasks', 0)))
        stats_table.add_row("Completed Tasks", str(stats.get('completed_tasks', 0)))
        stats_table.add_row("Failed Tasks", str(stats.get('failed_tasks', 0)))
        stats_table.add_row("Pending Tasks", str(stats.get('pending_tasks', 0)))
        
        layout["stats"].update(Panel(stats_table, title="Statistics"))
        
        # Node Information
        nodes_table = Table(title="Node Status")
        nodes_table.add_column("Node ID", style="cyan")
        nodes_table.add_column("Status", style="green")
        nodes_table.add_column("Load", style="yellow")
        nodes_table.add_column("Capacity", style="magenta")
        nodes_table.add_column("Specialization", style="blue")
        
        for node_id, node_info in nodes.items():
            status = "[green]Active[/green]" if node_info.get('computation_power', 0) > 0 else "[red]Inactive[/red]"
            current_load = node_info.get('current_load', 0)
            max_tasks = node_info.get('max_concurrent_tasks', 0)
            load_percent = f"{(current_load/max(max_tasks, 1)*100):.1f}%"
            capacity = f"{current_load}/{max_tasks}"
            specializations = ", ".join(node_info.get('specialized_tasks', [])[:2])
            
            nodes_table.add_row(node_id, status, load_percent, capacity, specializations)
        
        layout["nodes"].update(Panel(nodes_table, title="Nodes"))
        
        # Task Information
        tasks_table = Table(title="Task Status")
        tasks_table.add_column("Task ID", style="cyan")
        tasks_table.add_column("Type", style="green")
        tasks_table.add_column("Status", style="yellow")
        tasks_table.add_column("Node", style="magenta")
        tasks_table.add_column("Priority", style="blue")
        
        # Count tasks by status
        task_counts = {}
        for task_id, task_info in tasks.items():
            status = task_info.get('status', 'unknown')
            task_counts[status] = task_counts.get(status, 0) + 1
            
            # Show first few tasks
            if len(tasks_table.rows) < 10:
                task_type = task_info.get('task_type', 'unknown')[:15]
                task_status = task_info.get('status', 'unknown')
                assigned_node = task_info.get('assigned_node', 'none')
                priority = task_info.get('priority', 'medium')
                
                # Color code status
                if task_status == 'completed':
                    status_display = f"[green]{task_status}[/green]"
                elif task_status == 'failed':
                    status_display = f"[red]{task_status}[/red]"
                elif task_status == 'in_progress':
                    status_display = f"[yellow]{task_status}[/yellow]"
                else:
                    status_display = task_status
                
                tasks_table.add_row(
                    task_id[:12], task_type, status_display, 
                    assigned_node or "none", priority
                )
        
        layout["tasks"].update(Panel(tasks_table, title="Tasks"))
        
        # Performance Metrics
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="magenta")
        
        # Calculate performance metrics
        total_tasks = stats.get('total_tasks', 0)
        completed_tasks = stats.get('completed_tasks', 0)
        failed_tasks = stats.get('failed_tasks', 0)
        
        if total_tasks > 0:
            success_rate = (completed_tasks / total_tasks) * 100
            failure_rate = (failed_tasks / total_tasks) * 100
        else:
            success_rate = 0
            failure_rate = 0
        
        perf_table.add_row("Success Rate", f"{success_rate:.1f}%")
        perf_table.add_row("Failure Rate", f"{failure_rate:.1f}%")
        
        # Task status breakdown
        for status, count in task_counts.items():
            perf_table.add_row(f"{status.title()} Tasks", str(count))
        
        layout["performance"].update(Panel(perf_table, title="Performance"))
        
        # Footer
        footer_text = f"[dim]Press Ctrl+C to exit | Refresh interval: 5s | "
        footer_text += f"Scheduler Status: {'🟢 Online' if stats else '🔴 Offline'}[/dim]"
        layout["footer"].update(Panel(footer_text, style="dim"))
    
    def monitor(self, refresh_interval: int = 5):
        """Start monitoring with live updates"""
        layout = self.create_dashboard()
        
        try:
            with Live(layout, refresh_per_second=1, screen=True) as live:
                while True:
                    self.update_dashboard(layout)
                    time.sleep(refresh_interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped by user[/yellow]")
    
    def export_metrics(self, filename: str):
        """Export current metrics to JSON file"""
        try:
            metrics = {
                'timestamp': time.time(),
                'statistics': self.get_statistics(),
                'nodes': self.get_nodes(),
                'tasks': self.get_tasks()
            }
            
            with open(filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            console.print(f"[green]Metrics exported to {filename}[/green]")
        except Exception as e:
            console.print(f"[red]Error exporting metrics: {e}[/red]")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Monitor distributed crypto system')
    parser.add_argument('--host', default='localhost', help='Scheduler hostname')
    parser.add_argument('--port', type=int, default=8000, help='Scheduler port')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds')
    parser.add_argument('--export', help='Export metrics to JSON file')
    parser.add_argument('--simple', action='store_true', help='Simple text output instead of dashboard')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(args.host, args.port)
    
    if args.export:
        monitor.export_metrics(args.export)
    elif args.simple:
        # Simple monitoring output
        while True:
            try:
                stats = monitor.get_statistics()
                nodes = monitor.get_nodes()
                tasks = monitor.get_tasks()
                
                print(f"\n{time.strftime('%H:%M:%S')} - System Status:")
                print(f"  Nodes: {stats.get('active_nodes', 0)}/{stats.get('nodes', 0)}")
                print(f"  Tasks: {stats.get('completed_tasks', 0)}/{stats.get('total_tasks', 0)} completed")
                print(f"  Failed: {stats.get('failed_tasks', 0)}")
                
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break
    else:
        monitor.monitor(args.interval)


if __name__ == "__main__":
    main()