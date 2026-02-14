import subprocess
import sys
import time
from pathlib import Path
import os


def get_project_root() -> Path:
    """
    requirements.txt를 기준으로 LGBM 루트 탐색
    """
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    raise RuntimeError("프로젝트 루트를 찾을 수 없습니다.")


def main():
    ROOT = get_project_root()
    env = dict(**dict(os.environ))
    env["PYTHONPATH"] = str(ROOT)

    uvicorn_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]

    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "frontend" / "ui" / "streamlit_app.py"),
    ]

    print(f"[INFO] Project root: {ROOT}")

    uvicorn_proc = subprocess.Popen(
        uvicorn_cmd,
        cwd=ROOT,        
        env=env
    )

    time.sleep(1.0)  

    streamlit_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=ROOT,         
        env=env
    )

    try:
        streamlit_proc.wait()
    finally:
        uvicorn_proc.terminate()
        uvicorn_proc.wait()


if __name__ == "__main__":
    main()
