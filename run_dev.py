# run_dev.py
# Helper script to run FastAPI and Celery worker concurrently in a single terminal.

import subprocess
import sys
import threading
import time

def log_stream(stream, prefix):
    for line in iter(stream.readline, b''):
        sys.stdout.write(f"[{prefix}] {line.decode('utf-8', errors='ignore')}")
        sys.stdout.flush()

def main():
    print("🚀 Starting FastAPI and Celery worker concurrently...")
    
    # Start uvicorn
    uvicorn_cmd = ["poetry", "run", "uvicorn", "app.main:app", "--reload", "--port", "8000"]
    uvicorn_proc = subprocess.Popen(
        uvicorn_cmd,
        cwd="backend",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Start celery
    celery_cmd = ["poetry", "run", "celery", "-A", "app.workers.celery_app", "worker", "--loglevel=info", "--pool=solo"]
    celery_proc = subprocess.Popen(
        celery_cmd,
        cwd="backend",
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
