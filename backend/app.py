from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from backend.job_manager import job_manager
from backend.config import JOBS_DIR

app = FastAPI(title="CARE-CRITIC Backend", version="0.1.0")


@app.post("/jobs")
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are allowed")

    content = await file.read()
    job = job_manager.create_job(file.filename, content)

    background_tasks.add_task(job_manager.run_job, job.job_id)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = job_manager.get_job(job_id)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "report_path": job.report_path,
            "error_message": job.error_message,
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/jobs/{job_id}/log")
def get_log(job_id: str, from_bytes: int = 0):
    try:
        return job_manager.read_log(job_id, from_bytes=from_bytes)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/jobs/{job_id}/download")
def download_report(job_id: str):
    try:
        job = job_manager.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "done" or not job.report_path:
        raise HTTPException(status_code=400, detail="Report not ready")

    report_path = Path(job.report_path)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file missing")

    return FileResponse(
        path=str(report_path),
        filename=report_path.name,
        media_type="application/octet-stream",
    )


@app.get("/outputs")
def list_outputs():
    # "이전 결과물" 페이지에서 보여주기 위한 outputs 브라우징
    items = []
    for p in sorted(JOBS_DIR.glob("*")):
        if p.is_dir():
            job_json = p / "job.json"
            report_html = p / "report.html"
            report_md = p / "report.md"
            items.append({
                "job_id": p.name,
                "has_job_json": job_json.exists(),
                "has_report_html": report_html.exists(),
                "has_report_md": report_md.exists(),
            })
    return {"jobs": items}
