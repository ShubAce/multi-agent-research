# run_dev.py
# Helper script to run FastAPI and Celery worker concurrently in a single terminal.

import os
import subprocess
import sys
import threading
import time

# Children write UTF-8; the console may not be able to show every character,
# so replace those instead of letting a log-forwarding thread crash (a dead
# reader thread lets the pipe fill up and the child process hang).
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

def log_stream(stream, prefix):
    for line in iter(stream.readline, b''):
        try:
            sys.stdout.write(f"[{prefix}] {line.decode('utf-8', errors='replace')}")
            sys.stdout.flush()
        except Exception:
            pass  # keep draining the pipe no matter what

def main():
    print("🚀 Starting FastAPI and Celery worker concurrently...")
    
    # Start uvicorn
    # sys.executable = the Python running this script, so the backend uses the
    # same virtualenv (e.g. `backend\.multiAgent\Scripts\python run_dev.py`)
    uvicorn_cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"]
    uvicorn_proc = subprocess.Popen(
        uvicorn_cmd,
        cwd="backend",
        env=CHILD_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Start celery
    celery_cmd = [sys.executable, "-m", "celery", "-A", "app.workers.celery_app", "worker", "--loglevel=info", "--pool=solo"]
    celery_proc = subprocess.Popen(
        celery_cmd,
        cwd="backend",
        env=CHILD_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Start threads to read stdout and stderr
    t1 = threading.Thread(target=log_stream, args=(uvicorn_proc.stdout, "API"), daemon=True)
    t2 = threading.Thread(target=log_stream, args=(uvicorn_proc.stderr, "API"), daemon=True)
    t3 = threading.Thread(target=log_stream, args=(celery_proc.stdout, "WORKER"), daemon=True)
    t4 = threading.Thread(target=log_stream, args=(celery_proc.stderr, "WORKER"), daemon=True)

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    try:
        while True:
            # Check if any process terminated
            if uvicorn_proc.poll() is not None:
                print("API process terminated.")
                break
            if celery_proc.poll() is not None:
                print("Worker process terminated.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down services...")
    finally:
        uvicorn_proc.terminate()
        celery_proc.terminate()
        uvicorn_proc.wait()
        celery_proc.wait()
        print("Done.")

if __name__ == "__main__":
    main()
