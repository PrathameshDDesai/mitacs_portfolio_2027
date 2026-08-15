import sys
import subprocess
import time

PROJECTS = {
    "1": {
        "name": "01_hospital_los_prediction",
        "description": "Hospital Length of Stay (LOS) Predictor",
        "path": "01_hospital_los_prediction/app.py",
        "port": 5001
    },
    "2": {
        "name": "02_network_anomaly_detection",
        "description": "AgentGuard Network Anomaly Threat Detector",
        "path": "02_network_anomaly_detection/app.py",
        "port": 5002
    }
}

def print_banner():
    print("==================================================================")
    print("🚀 MITACS Multi-Project Machine Learning Portfolio Launcher")
    print("==================================================================")
    for key, proj in PROJECTS.items():
        print(f" [{key}] {proj['name']} - {proj['description']} (Port {proj['port']})")
    print(" [A] Launch ALL Projects Simultaneously")
    print(" [Q] Quit")
    print("==================================================================")

def launch_project(proj_key):
    proj = PROJECTS[proj_key]
    print(f"\n[+] Launching {proj['name']} on http://localhost:{proj['port']}...")
    cmd = [sys.executable, proj["path"]]
    return subprocess.Popen(cmd)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "all":
        choice = "A"
    else:
        print_banner()
        choice = input("Select option [1/2/A/Q]: ").strip().upper()

    processes = []
    if choice == "1":
        processes.append(launch_project("1"))
    elif choice == "2":
        processes.append(launch_project("2"))
    elif choice == "A":
        processes.append(launch_project("1"))
        processes.append(launch_project("2"))
        print("\n✅ All Projects launched in parallel!")
        print("  👉 Project 1 (Hospital LOS):        http://localhost:5001")
        print("  👉 Project 2 (Network Anomaly):     http://localhost:5002")
    else:
        print("Exiting launcher.")
        sys.exit(0)

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all project servers...")
        for p in processes:
            p.terminate()
        sys.exit(0)
