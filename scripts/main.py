import json
from pathlib import Path
from datetime import datetime
from scripts.run_agent_critique import run_agent_critique_pipeline


def run_pipeline(input_json: dict) -> dict:

    # 1) patient_json_path가 있으면 파일 로드
    if "patient_json_path" in input_json and input_json["patient_json_path"]:
        patient_path = Path(input_json["patient_json_path"])
        patient_data = json.loads(patient_path.read_text(encoding="utf-8"))
    else:
        patient_data = input_json

    # 2) 옵션 파라미터(없으면 기본값)
    db_path = input_json.get("db_path", "vector_db")
    top_k = int(input_json.get("top_k", 3))
    similarity_threshold = float(input_json.get("similarity_threshold", 0.7))
    max_iterations = int(input_json.get("max_iterations", 3))

    # 3) 파이프라인 실행
    result = run_agent_critique_pipeline(
        patient_data=patient_data,
        db_path=db_path,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        max_iterations=max_iterations,
    )
    return result




def main():
    # job_dir는 "현재 실행 작업 폴더(cwd)"로 고정
    job_dir = Path.cwd()

    in_path = job_dir / "input.json"
    if not in_path.exists():
        raise FileNotFoundError(f"[ERROR] input.json not found: {in_path}")

    outdir = job_dir  # 결과도 job_dir에 저장

    print("[INFO] Loading input:", in_path)
    input_json = json.loads(in_path.read_text(encoding="utf-8-sig"))  # BOM 안전

    print("[INFO] Running pipeline...")
    result = run_pipeline(input_json)

    # 리포트 저장 
    report_json = outdir / "report.json"
    report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[INFO] Saved report:", report_json)

    # HTML 생성 
    report_html = outdir / "report.html"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def esc(x):
        import html
        return html.escape("" if x is None else str(x))

    def sev_badge(sev: str) -> str:
        sev = (sev or "").lower()
        cls = {
            "critical": "sev sev-critical",
            "high": "sev sev-high",
            "medium": "sev sev-medium",
            "low": "sev sev-low",
        }.get(sev, "sev sev-unknown")
        return f'<span class="{cls}">{esc(sev.upper() if sev else "N/A")}</span>'

    patient_id = result.get("patient_id", "N/A")
    chart = result.get("structured_chart", {}) or {}
    demo = (chart.get("demographics", {}) or {})
    vitals = (chart.get("vitals", {}) or {})
    symptoms = (chart.get("symptoms", {}) or {})
    course = (chart.get("clinical_course", {}) or {})
    outcome = (chart.get("outcome", {}) or {})
    critique = result.get("critique", []) or []
    solutions = result.get("solutions", []) or []
    diag = result.get("diagnosis_analysis", {}) or {}
    treat = result.get("treatment_analysis", {}) or {}
    evidence_spans = chart.get("evidence_spans", []) or []

    # Timeline: structured_chart.clinical_course.events를 우선 사용
    timeline_events = course.get("events", []) or []
    # outcome의 critical events도 보강
    timeline_events2 = (outcome.get("critical_events_leading_to_outcome", []) or [])
    # 중복 제거(순서 유지)
    seen = set()
    timeline = []
    for ev in list(timeline_events) + list(timeline_events2):
        if ev and ev not in seen:
            seen.add(ev)
            timeline.append(ev)

    # Key failure modes: critique + diagnosis_analysis.issues 결합해 상위(critical/high) 우선
    diag_issues = diag.get("issues", []) or []
    combined_issues = []
    for c in critique:
        combined_issues.append({
            "title": c.get("point") or c.get("issue"),
            "severity": c.get("severity"),
            "category": c.get("category"),
            "evidence": c.get("span_id"),
            "cohort": c.get("cohort_comparison"),
        })
    for d in diag_issues:
        combined_issues.append({
            "title": d.get("issue"),
            "severity": d.get("severity"),
            "category": d.get("category"),
            "evidence": d.get("evidence_in_text"),
            "cohort": None,
        })

    sev_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    combined_issues.sort(key=lambda x: sev_rank.get((x.get("severity") or "").lower(), 9))

    def render_list(items):
        if not items:
            return "<div class='muted'>None documented</div>"
        return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"

    # Evidence span map: "E6 | E9" 같은 표기를 실제 span 텍스트로 연결(가능하면)
    # 현재 evidence_spans는 field/text_span만 있고 E번호는 없어 매핑 한계가 있음.
    # 대신 evidence_spans 자체를 부록으로 표기.
    evidence_html = ""
    if evidence_spans:
        rows = []
        for i, sp in enumerate(evidence_spans, start=1):
            rows.append(
                f"<tr><td class='mono'>E{i}</td><td>{esc(sp.get('field'))}</td><td class='mono'>{esc(sp.get('text_span'))}</td></tr>"
            )
        evidence_html = (
            "<table class='table'>"
            "<thead><tr><th>ID</th><th>Field</th><th>Text span</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>"
        )
    else:
        evidence_html = "<div class='muted'>No evidence spans available.</div>"

    # Critique table
    crit_rows = ""
    for i, c in enumerate(critique, start=1):
        crit_rows += f"""
        <tr>
            <td class="mono">{i}</td>
            <td>{sev_badge(c.get("severity"))}</td>
            <td>{esc(c.get("category"))}</td>
            <td>{esc(c.get("point") or c.get("issue"))}</td>
            <td class="mono">{esc(c.get("span_id"))}</td>
            <td>{esc(c.get("cohort_comparison"))}</td>
        </tr>
        """
    crit_table = (
        "<div class='muted'>No critique points generated.</div>"
        if not critique else
        f"""
        <table class="table">
        <thead>
            <tr>
            <th>#</th><th>Severity</th><th>Category</th><th>Finding</th><th>Evidence</th><th>Cohort comparison</th>
            </tr>
        </thead>
        <tbody>{crit_rows}</tbody>
        </table>
        """
    )

    # Solutions
    sol_rows = ""
    for i, s in enumerate(solutions, start=1):
        sol_rows += f"""
        <tr>
            <td class="mono">{i}</td>
            <td><span class="pill pill-priority">{esc((s.get("priority") or "").upper() or "N/A")}</span></td>
            <td>{esc(s.get("target_issue"))}</td>
            <td>{esc(s.get("action"))}</td>
            <td class="mono">{esc(s.get("citation"))}</td>
        </tr>
        """
    sol_table = (
        "<div class='muted'>No solutions generated.</div>"
        if not solutions else
        f"""
        <table class="table">
        <thead>
            <tr>
            <th>#</th><th>Priority</th><th>Targets</th><th>Recommended action</th><th>Reference</th>
            </tr>
        </thead>
        <tbody>{sol_rows}</tbody>
        </table>
        """
    )

    # Uncertainty / alternatives
    alt = result.get("alternative_explanations", {}) or {}
    alt_list = alt.get("alternative_explanations", []) or []
    uncertainty_notes = alt.get("uncertainty_notes", []) or []
    caveats = alt.get("caveats", []) or []

    # Header summary strings
    cc = demo.get("chief_complaint")
    sex = demo.get("sex")
    age = demo.get("age")
    cause_of_death = outcome.get("cause_of_death")
    status = outcome.get("status")

    # Safety disclaimer: 연구/QA 목적임을 명시(실제 진료용 오해 방지)
    disclaimer = (
        "This report is automatically generated for quality improvement / research use. "
        "It is not medical advice and must not be used for direct clinical decision-making."
    )

    report_html.write_text(
    f"""
    <html>
    <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>CARE-CRITIC Report | {esc(patient_id)}</title>
    <style>
        :root {{
        --bg: #0b1020;
        --fg: #d6deeb;
        --muted: #8aa1c1;
        --card: rgba(255,255,255,0.06);
        --border: rgba(255,255,255,0.12);
        --accent: #7aa2ff;
        --good: #37d67a;
        --warn: #ffcc66;
        --bad: #ff6b6b;
        --crit: #ff4d4d;
        }}
        body {{
        margin: 0; padding: 28px;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto;
        background: radial-gradient(1200px 600px at 20% -10%, rgba(122,162,255,0.25), transparent 60%),
                    radial-gradient(1000px 500px at 90% 0%, rgba(55,214,122,0.12), transparent 55%),
                    var(--bg);
        color: var(--fg);
        }}
        a {{ color: var(--accent); text-decoration: none; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .topbar {{ display:flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
        .title {{ font-size: 22px; font-weight: 800; letter-spacing: 0.2px; }}
        .subtitle {{ color: var(--muted); font-size: 13px; }}
        .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; margin-top: 16px; }}
        .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        backdrop-filter: blur(8px);
        }}
        .card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #e6eeff; }}
        .kv {{ display:grid; grid-template-columns: 170px 1fr; gap: 8px 12px; font-size: 13px; }}
        .k {{ color: var(--muted); }}
        .v {{ color: var(--fg); }}
        .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; font-size: 12px; }}
        .muted {{ color: var(--muted); font-size: 13px; }}
        .section {{ margin-top: 18px; }}
        .section h2 {{ font-size: 16px; margin: 0 0 10px 0; }}
        .sev {{
        display:inline-block; padding: 3px 9px; border-radius: 999px;
        font-size: 11px; font-weight: 800; letter-spacing: 0.3px;
        border: 1px solid var(--border);
        }}
        .sev-critical {{ background: rgba(255,77,77,0.18); color: #ffd1d1; border-color: rgba(255,77,77,0.35); }}
        .sev-high     {{ background: rgba(255,107,107,0.14); color: #ffdada; border-color: rgba(255,107,107,0.28); }}
        .sev-medium   {{ background: rgba(255,204,102,0.14); color: #ffe7b3; border-color: rgba(255,204,102,0.28); }}
        .sev-low      {{ background: rgba(55,214,122,0.10); color: #c7ffe0; border-color: rgba(55,214,122,0.22); }}
        .sev-unknown  {{ background: rgba(138,161,193,0.12); color: #d6deeb; border-color: rgba(138,161,193,0.25); }}
        .pill {{
        display:inline-block; padding: 3px 8px; border-radius: 10px;
        border: 1px solid var(--border);
        font-size: 11px; font-weight: 700;
        }}
        .pill-priority {{ background: rgba(122,162,255,0.14); color: #dbe6ff; border-color: rgba(122,162,255,0.28); }}
        .table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(0,0,0,0.18);
        }}
        .table th, .table td {{
        padding: 10px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.10);
        vertical-align: top;
        font-size: 12.5px;
        line-height: 1.35;
        }}
        .table th {{
        text-align: left;
        color: #e6eeff;
        font-size: 12px;
        background: rgba(255,255,255,0.04);
        }}
        .table tr:last-child td {{ border-bottom: none; }}
        .cols-6 {{ grid-column: span 6; }}
        .cols-4 {{ grid-column: span 4; }}
        .cols-8 {{ grid-column: span 8; }}
        .cols-12 {{ grid-column: span 12; }}
        ul {{ margin: 8px 0 0 18px; color: var(--fg); }}
        li {{ margin: 4px 0; }}
        .hr {{ height: 1px; background: rgba(255,255,255,0.12); margin: 14px 0; }}
        pre {{
        background: rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 12px;
        border-radius: 14px;
        overflow: auto;
        color: var(--fg);
        }}
        .footer {{ margin-top: 18px; color: var(--muted); font-size: 12px; }}
    </style>
    </head>

    <body>
    <div class="container">
    <div class="topbar">
        <div>
        <div class="title">CARE-CRITIC Clinical Critique Report</div>
        <div class="subtitle">Generated: {esc(now)} · Patient ID: <span class="mono">{esc(patient_id)}</span></div>
        </div>
        <div class="muted mono">{esc(disclaimer)}</div>
    </div>

    <div class="grid">
        <div class="card cols-6">
        <h3>Case Summary</h3>
        <div class="kv">
            <div class="k">Chief complaint</div><div class="v">{esc(cc)}</div>
            <div class="k">Demographics</div><div class="v">{esc(sex)}{(" / " + esc(age) + "y") if age is not None else ""}</div>
            <div class="k">Vitals</div>
            <div class="v">
            T {esc(vitals.get("temperature"))} · BP {esc(vitals.get("blood_pressure"))} · HR {esc(vitals.get("heart_rate"))} · RR {esc(vitals.get("respiratory_rate"))} · SpO2 {esc(vitals.get("oxygen_saturation"))}% ({esc(vitals.get("oxygen_requirement"))})
            </div>
            <div class="k">Key symptoms</div>
            <div class="v">
            Resp: {esc(", ".join(symptoms.get("respiratory", []) or []))} /
            Systemic: {esc(", ".join(symptoms.get("systemic", []) or []))}
            </div>
        </div>
        </div>

        <div class="card cols-6">
        <h3>Outcome</h3>
        <div class="kv">
            <div class="k">Status</div><div class="v">{esc(status)}</div>
            <div class="k">Cause of death</div><div class="v">{esc(cause_of_death)}</div>
            <div class="k">Safety assessment</div><div class="v">{esc((diag.get("procedural_safety_assessment") or {}).get("overall"))}</div>
            <div class="k">Confidence</div><div class="v">{esc(result.get("confidence"))}</div>
        </div>
        <div class="hr"></div>
        <div class="muted">
            Death-cause mismatch:
            <span class="mono">{esc((diag.get("death_cause_alignment") or {}).get("mismatch"))}</span>
            · Admission reason: <span class="mono">{esc((diag.get("death_cause_alignment") or {}).get("admission_reason"))}</span>
        </div>
        </div>

        <div class="card cols-12">
        <h3>Timeline (Key Events)</h3>
        {render_list(timeline)}
        </div>
    </div>

    <div class="section">
        <h2>Key Critique Points</h2>
        <div class="card">
        {crit_table}
        </div>
    </div>

    <div class="section">
        <h2>Top Failure Modes (Ranked)</h2>
        <div class="card">
        <table class="table">
            <thead>
            <tr>
                <th>Severity</th><th>Category</th><th>Issue</th><th>Evidence</th><th>Cohort / Notes</th>
            </tr>
            </thead>
            <tbody>
            {"".join(
                f"<tr>"
                f"<td>{sev_badge(x.get('severity'))}</td>"
                f"<td>{esc(x.get('category'))}</td>"
                f"<td>{esc(x.get('title'))}</td>"
                f"<td class='mono'>{esc(x.get('evidence'))}</td>"
                f"<td>{esc(x.get('cohort'))}</td>"
                f"</tr>"
                for x in combined_issues[:8]
            ) if combined_issues else "<tr><td colspan='5' class='muted'>No issues detected.</td></tr>"}
            </tbody>
        </table>
        </div>
    </div>

    <div class="section">
        <h2>Corrective Actions (Solutions)</h2>
        <div class="card">
        {sol_table}
        </div>
    </div>

    <div class="section">
        <h2>Clinical Reasoning Checks</h2>
        <div class="grid">
        <div class="card cols-6">
            <h3>Diagnosis Analysis</h3>
            <div class="kv">
            <div class="k">Evaluation</div><div class="v">{esc(diag.get("diagnosis_evaluation"))}</div>
            <div class="k">Timing assessment</div><div class="v">{esc(diag.get("timing_assessment"))}</div>
            <div class="k">Outcome analysis</div><div class="v">{esc(diag.get("actual_outcome_analysis"))}</div>
            </div>
        </div>

        <div class="card cols-6">
            <h3>Treatment Analysis</h3>
            <div class="kv">
            <div class="k">Evaluation</div><div class="v">{esc(treat.get("treatment_evaluation"))}</div>
            <div class="k">Guideline adherence</div><div class="v">{esc(treat.get("guideline_adherence"))}</div>
            </div>
            <div class="hr"></div>
            <div class="muted">Medication issues</div>
            {render_list(treat.get("medication_issues", []) or [])}
        </div>

        <div class="card cols-12">
            <h3>Recommended Clinical Next Steps (from analysis)</h3>
            {render_list(treat.get("recommendations", []) or [])}
        </div>
        </div>
    </div>

    <div class="section">
        <h2>Uncertainty & Alternative Explanations</h2>
        <div class="grid">
        <div class="card cols-6">
            <h3>Alternative explanations</h3>
            {render_list(alt_list)}
        </div>
        <div class="card cols-6">
            <h3>Uncertainty notes</h3>
            {render_list(uncertainty_notes)}
        </div>
        <div class="card cols-12">
            <h3>Caveats</h3>
            {render_list(caveats)}
        </div>
        </div>
    </div>

    <div class="section">
        <h2>Evidence Appendix</h2>
        <div class="card">
        <div class="muted">Evidence spans extracted from source note (indexed as E1..En)</div>
        <div style="margin-top:10px;">{evidence_html}</div>
        </div>
    </div>

    <div class="footer">
        Generated by CARE-CRITIC pipeline · For QI / research only.
    </div>
    </div>
    </body>
    </html>
    """,
    encoding="utf-8"
    )

    print("[INFO] Saved HTML report:", report_html)


if __name__ == "__main__":
    main()
