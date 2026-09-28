import os
import sys
import subprocess
import time

def main():
    print("=" * 60)
    print(" PREDICTIVE FLIGHT DELAY ANALYSIS SYSTEM ")
    print("=" * 60)

    # Step 1: Check dataset & model artifacts
    if not os.path.exists("models/flight_delay_nn.pt"):
        print("\n[1/2] Training Deep Neural Network Model...")
        subprocess.run([sys.executable, "src/train.py"], check=True)
    else:
        print("\n[1/2] Deep Learning Model & preprocessor artifacts found.")

    # Step 2: Start FastAPI backend server
    print("\n[2/2] Launching Flighora System Server on http://127.0.0.1:8001 ...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8001"
        ], check=True)
    except KeyboardInterrupt:
        print("\nShutting down Flighora System...")

if __name__ == "__main__":
    main()
