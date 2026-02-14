import json
import uuid  # 아이디 랜덤 생성
import time
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import os, traceback
from backend.config import JOBS_DIR, UPLOAD_DIR, ANALYSIS_MAIN


@dataclass
class Job:
    job_id: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    input_path: Optional[str] = None
    log_path: Optional[str] = None
    report_path: Optional[str] = None
    error_message: Optional[str] = None


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def _job_dir(self, job_id: str) -> Path:
        d = JOBS_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_job(self, uploaded_filename: str, file_bytes: bytes) -> Job:
        
        job_id = str(uuid.uuid4())
        job_dir = self._job_dir(job_id)

        # (선택) 업로드 원본 보관: 추적용
        upload_path = UPLOAD_DIR / f"{job_id}_{uploaded_filename}"
        upload_path.write_bytes(file_bytes)

        job_input = job_dir / "input.json"
        job_input.write_bytes(file_bytes)

        log_path = job_dir / "job.log"
        log_path.write_text("", encoding="utf-8")

        job = Job(
            job_id=job_id,
            status="queued",
            created_at=time.time(),
            input_path=str(job_input),  
            log_path=str(log_path),
        )
        self.jobs[job_id] = job
        self._persist_job(job_id)
        return job

    def _persist_job(self, job_id: str):
        job = self.jobs[job_id]
        job_dir = self._job_dir(job_id)
        (job_dir / "job.json").write_text(
            json.dumps(asdict(job), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def get_job(self, job_id: str) -> Job:
        if job_id in self.jobs:
            return self.jobs[job_id]
        job_dir = JOBS_DIR / job_id
        job_json = job_dir / "job.json"
        if job_json.exists():
            data = json.loads(job_json.read_text(encoding="utf-8"))
            job = Job(**data)
            self.jobs[job_id] = job
            return job
        raise KeyError(f"Job not found: {job_id}")

    def read_log(self, job_id: str, from_bytes: int = 0):
        job = self.get_job(job_id)
        log_path = Path(job.log_path)
        if not log_path.exists():
            return {"text": "", "next_from_bytes": from_bytes}
        raw = log_path.read_bytes()
        chunk = raw[from_bytes:]
        return {
            "text": chunk.decode("utf-8", errors="replace"),
            "next_from_bytes": len(raw),
        }

    def _append_log(self, log_path: Path, line: str):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def run_job(self, job_id: str):
        job = self.get_job(job_id)
        job.status = "running"
        job.started_at = time.time()
        self._persist_job(job_id)

        job_dir = self._job_dir(job_id)
        log_path = Path(job.log_path)

        # 실행:  <ANALYSIS_MAIN>
        cmd = [ sys.executable,"-u",
               str(ANALYSIS_MAIN),]

        try:
            with subprocess.Popen(
                cmd,
                cwd=str(job_dir),              
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            ) as proc:
                for line in proc.stdout:
                    self._append_log(log_path, line)

                code = proc.wait()

            if code != 0:
                job.status = "error"
                job.error_message = f"Process exited with code {code}"
                job.finished_at = time.time()
                self._persist_job(job_id)
                return

            # ✅ job_dir에서 report 탐색
            report_path = None
            for name in ["report.html","report.json"]:
                p = job_dir / name
                if p.exists():
                    report_path = str(p)
                    break

            if report_path is None:
                job.status = "error"
                job.error_message = "Process finished but report file not found in job_dir"
                job.finished_at = time.time()
                self._persist_job(job_id)
                return

            job.status = "done"
            job.report_path = report_path
            job.finished_at = time.time()
            self._persist_job(job_id)

        except Exception as e:
            job.status = "error"
            job.error_message = repr(e)
            job.finished_at = time.time()
            self._persist_job(job_id)


job_manager = JobManager()
