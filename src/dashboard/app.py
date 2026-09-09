"""
Real-time dashboard for the traffic allocation emulator.
(Mock implementation: In a full version, this would be a Streamlit or Dash app
reading from the telemetry CSV or a live socket).
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_dashboard_report(metrics_path: str = "simulation_metrics.csv"):
    """
    Generate static plots from the telemetry CSV to simulate a dashboard.
    """
    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} not found.")
        return

    df = pd.read_csv(metrics_path)

    # 1. Objective Trajectory
    plt.figure(figsize=(10, 5))
    plt.plot(df['iteration'], df['objective'], marker='o', markersize=2, label='Objective')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('System Objective Convergence')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig('dashboard_objective.png')
    plt.close()

    # 2. Relative Change (Log Scale)
    plt.figure(figsize=(10, 5))
    plt.plot(df['iteration'], df['rel_change'], marker='o', markersize=2, color='orange', label='Rel Change')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Change')
    plt.title('Convergence Rate')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig('dashboard_convergence.png')
    plt.close()

    # 3. Max Utilization
    plt.figure(figsize=(10, 5))
    plt.plot(df['iteration'], df['max_util'], marker='o', markersize=2, color='green', label='Max Util')
    plt.xlabel('Iteration')
    plt.ylabel('Utilization')
    plt.title('Max Broker Utilization')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig('dashboard_util.png')
    plt.close()

    print("Dashboard report generated: dashboard_objective.png, dashboard_convergence.png, dashboard_util.png")

if __name__ == "__main__":
    generate_dashboard_report()
