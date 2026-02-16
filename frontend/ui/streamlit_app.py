import time
import requests
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import re
from datetime import datetime

st.set_page_config(page_title="CARE-CRITIC", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000"  # FastAPI 주소

# ============================== SessionState 초기값 ==============================
DEFAULT_PAGE = "intro"
VALID_PAGES = ["intro", "dashboard", "patient_analysis", "previous_outputs"]

def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

ss_init("page", DEFAULT_PAGE)

# 분석(파이프라인) 상태는 네임스페이스처럼 키를 분리
ss_init("analysis_job_id", None)
ss_init("analysis_job_status", None)
ss_init("analysis_log_from_bytes", 0)
ss_init("analysis_log_text", "")
ss_init("analysis_auto_refresh", True)



# ============================== query param 유틸 ==============================
def get_qp(name: str):
    try:
        return st.query_params.get(name, None)
    except Exception:
        return st.experimental_get_query_params().get(name, [None])[0]

def set_qp(params: dict):
    try:
        st.query_params.update(params)
    except Exception:
        st.experimental_set_query_params(**params)

if "qp_initialized" not in st.session_state:
    qp_page = get_qp("page")
    if qp_page in VALID_PAGES:
        st.session_state.page = qp_page
    st.session_state.qp_initialized = True


def goto(page: str):
    st.session_state.page = page
    set_qp({"page": page})


# ============================== CSS 로드 ==============================
css_path = Path(__file__).parent / "ui" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ============================== 페이지 상태 ==============================
qp_page = get_qp("page")
if qp_page in ["dashboard", "patient_analysis", "previous_outputs", "intro"]:
    st.session_state.page = qp_page

if "page" not in st.session_state:
    st.session_state.page = "intro"

# ============================== 공통 Footer ==============================
def render_footer():
    st.markdown(
        """
        <div class="footer">
          © CARE-CRITIC / BITAmin 15TH
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================== INTRO ==============================
if st.session_state.page == "intro":
    st.markdown(
        """
        <style>
        .intro-wrap{
          height:40vh;
          display:flex;
          flex-direction:column;
          justify-content:flex-start;
          align-items:center;
          text-align:center;
          padding-top:160px;
        }
        .intro-title{
          font-size:42px;
          font-weight:800;
          margin-bottom:16px;
        }
        .intro-desc{
          font-size:16px;
          color:#666;
          margin-bottom:22px;
          line-height:1.6;
        }
        .start-btn-wrap{
          width: 220px;
          margin-top: -40px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="intro-wrap">
          <div class="intro-title">CARE-CRITIC</div>
          <div class="intro-desc">
            환자 데이터 기반 분석 파이프라인을 실행하고,<br/>
            실시간 로그와 리포트를 한 화면에서 관리합니다.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='start-btn-wrap'>", unsafe_allow_html=True)
    if st.button("대시보드 시작하기 ▶", use_container_width=True, key="start_btn"):
        st.session_state.page = "dashboard"
        set_qp({"page": "dashboard"})
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    render_footer()
    st.stop()

# ============================== DASHBOARD(공통) ==============================
if st.session_state.page == "dashboard":
    st.title("CARE-CRITIC Dashboard")
    st.info("왼쪽 사이드바에서 기능을 선택하세요.")

# ============================== 사이드바 ==============================
st.sidebar.title("옵션 선택")

st.sidebar.button(
    "1. 환자 데이터 기반 분석",
    use_container_width=True,
    on_click=goto,
    kwargs={"page": "patient_analysis"},
)

st.sidebar.button(
    "2. 이전 결과물",
    use_container_width=True,
    on_click=goto,
    kwargs={"page": "previous_outputs"},
)

st.sidebar.divider()

st.sidebar.button(
    "3. 처음 화면으로",
    use_container_width=True,
    on_click=goto,
    kwargs={"page": "intro"},
)


# ============================== 유틸: 상태 pill ==============================
def pill(status: str) -> str:
    if status == "running":
        return "<span class='pill pill-running'>RUNNING</span>"
    if status == "done":
        return "<span class='pill pill-done'>DONE</span>"
    if status == "error":
        return "<span class='pill pill-error'>ERROR</span>"
    return "<span class='pill'>QUEUED</span>"

def start_analysis():
    uploaded = st.session_state.get("patient_json_uploader")
    if not uploaded:
        st.session_state["analysis_error"] = "JSON 파일을 업로드."
        return

    files = {"file": (uploaded.name, uploaded.getvalue(), "application/json")}
    try:
        resp = requests.post(f"{BACKEND_URL}/jobs", files=files, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        st.session_state.analysis_job_id = data["job_id"]
        st.session_state.analysis_job_status = data["status"]
        st.session_state.analysis_log_from_bytes = 0
        st.session_state.analysis_log_text = ""
        st.session_state["analysis_error"] = None
        st.session_state["analysis_started"] = True
    except Exception as e:
        st.session_state["analysis_error"] = f"백엔드 호출 실패: {repr(e)}"


# ============================== 페이지 1) 환자 데이터 기반 분석 ==============================
if st.session_state.page == "patient_analysis":
    st.title("환자 데이터 기반 분석")
    st.caption("JSON 파일을 드랍하고, 분석을 실행하면 로그가 실시간으로 누적 출력됩니다.")

    c1, c2 = st.columns([1.1, 0.9])

    with c1:
        uploaded = st.file_uploader("환자 데이터 JSON 업로드", type=["json"], accept_multiple_files=False,key="patient_json_uploader")

        start_clicked = st.button("분석 시작", type="primary", use_container_width=True,on_click=start_analysis)

    with c2:
        st.markdown("#### 실행 상태")
        if "job_status" not in st.session_state:
            st.session_state.job_status = None
        if "job_id" not in st.session_state:
            st.session_state.job_id = None

        if st.session_state.job_status:
            st.markdown(pill(st.session_state.job_status), unsafe_allow_html=True)
            if st.session_state.job_id:
                st.code(st.session_state.job_id, language="text")
        else:
            st.markdown("<span class='pill'>IDLE</span>", unsafe_allow_html=True)

    st.divider()

    # ---- 분석 시작 ----
    if start_clicked:
        if not uploaded:
            st.warning("먼저 JSON 파일을 업로드해줘.")
        else:
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/json")}
            try:
                resp = requests.post(f"{BACKEND_URL}/jobs", files=files, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                st.session_state.job_id = data["job_id"]
                st.session_state.job_status = data["status"]
                st.session_state.log_from_bytes = 0
                st.session_state.log_text = ""
                st.success("분석을 시작했어. 아래 로그를 확인해줘.")
                st.rerun()
            except Exception as e:
                st.error(f"백엔드 호출 실패: {repr(e)}")

    # ---- 로그/리포트 영역 ----
    if st.session_state.get("job_id"):
        job_id = st.session_state.job_id

        # ---------- UI 스타일 ----------
        st.markdown(
            """
            <style>
            .job-card{
            border:1px solid rgba(49,51,63,.2);
            border-radius:16px;
            padding:16px 16px 12px 16px;
            background: rgba(255,255,255,0.65);
            backdrop-filter: blur(8px);
            box-shadow: 0 6px 18px rgba(0,0,0,.06);
            }
            .row{
            display:flex;
            gap:10px;
            align-items:center;
            flex-wrap:wrap;
            }
            .badge{
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:6px 10px;
            border-radius:999px;
            font-size:13px;
            font-weight:600;
            border:1px solid rgba(49,51,63,.16);
            background:#fff;
            }
            .dot{ width:10px; height:10px; border-radius:50%; display:inline-block; }
            .subtle{ color: rgba(49,51,63,.65); font-size:12px; }

            .log-shell{
            margin-top:12px;
            border-radius:16px;
            border:1px solid rgba(49,51,63,.18);
            overflow:hidden;
            background: #0b1020;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.04);
            }
            .log-topbar{
            display:flex;
            align-items:center;
            justify-content:space-between;
            padding:10px 12px;
            background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
            border-bottom:1px solid rgba(255,255,255,.08);
            }
            .traffic{ display:flex; gap:6px; align-items:center; }
            .t-dot{ width:10px; height:10px; border-radius:50%; opacity:.9; }
            .t-red{ background:#ff5f57; }
            .t-yellow{ background:#febc2e; }
            .t-green{ background:#28c840; }
            .log-title{
            color: rgba(255,255,255,.75);
            font-size:12px;
            font-weight:700;
            letter-spacing:.2px;
            }
            .log-meta{ color: rgba(255,255,255,.55); font-size:11px; }

            .log-body{
            padding:12px;
            max-height: 420px;
            overflow:auto;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size:12.5px;
            line-height:1.55;
            color:#d6deeb;
            white-space:pre-wrap;
            word-break:break-word;
            }
            .line{ display:flex; gap:10px; }
            .ln{
            width:44px;
            flex: 0 0 44px;
            text-align:right;
            color: rgba(214,222,235,.35);
            user-select:none;
            }
            .msg{ flex:1; }
            .hl-err{ color:#ff6b6b; font-weight:700; }
            .hl-warn{ color:#ffd166; font-weight:700; }
            .hl-ok{ color:#7ae582; font-weight:700; }
            .hl-sec{ color:#80d8ff; font-weight:700; }
            </style>
            """,
            unsafe_allow_html=True
        )

        # ---------- helpers ----------
        def status_ui(status: str):
            status = (status or "unknown").lower()
            mapping = {
                "queued":   ("#8a8f98", "대기중"),
                "running":  ("#3b82f6", "분석중"),
                "done":     ("#22c55e", "완료"),
                "error":    ("#ef4444", "오류"),
            }
            color, label = mapping.get(status, ("#8a8f98", status))
            return color, label

        def pretty_log_html(log_text: str, start_line_no: int = 1) -> str:
            t = (log_text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            lines = t.split("\n")

            out = []
            line_no = start_line_no
            for line in lines:
                cls = ""
                if re.search(r"\b(error|exception|traceback|failed)\b", line, re.IGNORECASE):
                    cls = "hl-err"
                elif re.search(r"\b(warn|warning)\b", line, re.IGNORECASE):
                    cls = "hl-warn"
                elif re.search(r"\b(\[ok\]|\bok\b|success|saved)\b", line, re.IGNORECASE):
                    cls = "hl-ok"
                elif re.search(r"\[([1-5]/[1-5])\]", line):
                    cls = "hl-sec"

                msg = line if line.strip() != "" else " "
                out.append(
                    f"<div class='line'><div class='ln'>{line_no:>4}</div><div class='msg {cls}'>{msg}</div></div>"
                )
                line_no += 1
            return "\n".join(out)

        # ---------- session_state init ----------
        if "log_from_bytes" not in st.session_state:
            st.session_state.log_from_bytes = 0
        if "log_text" not in st.session_state:
            st.session_state.log_text = ""
        if "last_line_no" not in st.session_state:
            st.session_state.last_line_no = 1
        if "report_shown" not in st.session_state:
            st.session_state.report_shown = False

        header_placeholder = st.empty()
        controls_placeholder = st.empty()
        log_placeholder = st.empty()
        report_placeholder = st.empty()

        # ---------- controls ----------
        with controls_placeholder.container():
            refresh_col1, refresh_col2, refresh_col3 = st.columns([0.22, 0.48, 0.30])
            with refresh_col1:
                manual_refresh = st.button("🔄 로그 새로고침", use_container_width=True)
            with refresh_col2:
                auto = st.checkbox("자동 갱신(2초)", value=True)
            with refresh_col3:
                if st.button("🧹 로그 리셋", use_container_width=True):
                    st.session_state.log_from_bytes = 0
                    st.session_state.log_text = ""
                    st.session_state.last_line_no = 1

        # ============================
        # ✅ 핵심: 깜빡임 제거용 fragment
        # ============================
        @st.fragment(run_every="2s")
        def live_panel(job_id: str, auto: bool, manual_refresh: bool):
            # 1) 상태 조회
            sdata = None
            try:
                sresp = requests.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=15)
                sresp.raise_for_status()
                sdata = sresp.json()
                st.session_state.job_status = sdata.get("status")
            except Exception as e:
                st.error(f"상태 조회 실패: {repr(e)}")

            # Header
            if sdata:
                color, label = status_ui(sdata.get("status"))
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header_placeholder.markdown(
                    f"""
                    <div class="job-card">
                    <div class="row">
                        <span class="badge"><span class="dot" style="background:{color}"></span>
                        Job <span style="opacity:.75;">{job_id}</span>
                        </span>
                        <span class="badge">상태: <span style="color:{color};">{label}</span></span>
                        <span class="badge"><span class="subtle">마지막 갱신</span>&nbsp;{now}</span>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 2) 로그 조회 (증분) + 누적
            should_poll = (auto and st.session_state.job_status in ["queued", "running"]) or manual_refresh
            if should_poll:
                try:
                    from_bytes = st.session_state.log_from_bytes
                    lresp = requests.get(
                        f"{BACKEND_URL}/jobs/{job_id}/log",
                        params={"from_bytes": from_bytes},
                        timeout=15
                    )
                    lresp.raise_for_status()
                    ldata = lresp.json()

                    new_text = ldata.get("text", "")
                    st.session_state.log_from_bytes = ldata.get("next_from_bytes", from_bytes)

                    if new_text:
                        st.session_state.log_text += new_text
                except Exception as e:
                    st.error(f"로그 조회 실패: {repr(e)}")

            # 3) 로그 HTML 생성
            full = st.session_state.log_text
            if not full.strip():
                log_html = "<div class='subtle' style='color:rgba(214,222,235,.55)'>아직 로그가 없습니다.</div>"
                meta = f"from_bytes={st.session_state.log_from_bytes}"
            else:
                log_html = pretty_log_html(full, start_line_no=1)
                meta = f"bytes={len(full)} · from_bytes={st.session_state.log_from_bytes}"

            # ✅ 핵심: 디자인 유지 + 스크롤 = iframe 안에 CSS + JS 같이 넣기
            components.html(
                f"""
                <style>
                .log-shell{{
                margin-top:12px;
                border-radius:16px;
                border:1px solid rgba(49,51,63,.18);
                overflow:hidden;
                background:#0b1020;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,.04);
                }}
                .log-topbar{{
                display:flex;
                align-items:center;
                justify-content:space-between;
                padding:10px 12px;
                background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
                border-bottom:1px solid rgba(255,255,255,.08);
                }}
                .traffic{{ display:flex; gap:6px; align-items:center; }}
                .t-dot{{ width:10px; height:10px; border-radius:50%; opacity:.9; display:inline-block; }}
                .t-red{{ background:#ff5f57; }}
                .t-yellow{{ background:#febc2e; }}
                .t-green{{ background:#28c840; }}
                .log-title{{
                color: rgba(255,255,255,.75);
                font-size:12px;
                font-weight:700;
                letter-spacing:.2px;
                }}
                .log-meta{{ color: rgba(255,255,255,.55); font-size:11px; }}
                .log-body{{
                padding:12px;
                max-height:420px;
                overflow:auto;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                font-size:12.5px;
                line-height:1.55;
                color:#d6deeb;
                white-space:pre-wrap;
                word-break:break-word;
                }}
                .line{{ display:flex; gap:10px; }}
                .ln{{
                width:44px;
                flex:0 0 44px;
                text-align:right;
                color: rgba(214,222,235,.35);
                user-select:none;
                }}
                .msg{{ flex:1; }}
                .hl-err{{ color:#ff6b6b; font-weight:700; }}
                .hl-warn{{ color:#ffd166; font-weight:700; }}
                .hl-ok{{ color:#7ae582; font-weight:700; }}
                .hl-sec{{ color:#80d8ff; font-weight:700; }}
                </style>

                <div class="log-shell">
                <div class="log-topbar">
                    <div class="traffic">
                    <span class="t-dot t-red"></span>
                    <span class="t-dot t-yellow"></span>
                    <span class="t-dot t-green"></span>
                    </div>
                    <div class="log-title">Live Logs</div>
                    <div class="log-meta">{meta}</div>
                </div>

                <div class="log-body" id="logBody">
                    {log_html}
                </div>
                </div>

                <script>
                // iframe 내부에서만 접근 
                const el = document.getElementById("logBody");
                if (el) {{
                    el.scrollTop = el.scrollHeight;
                }}
                </script>
                """,
                height=520,
                scrolling=False
            )

            # 4) 완료 시 리포트 표시 + 다운로드 (기존 그대로)
            if st.session_state.job_status == "done" and sdata:
                if not st.session_state.report_shown:
                    report_placeholder.success("분석 완료! 아래에서 리포트를 확인하고 다운로드할 수 있어.")
                    st.session_state.report_shown = True

                report_path = sdata.get("report_path")
                with report_placeholder.container():
                    if report_path and report_path.endswith(".html"):
                        try:
                            report_file = Path(report_path)
                            report_html = report_file.read_text(encoding="utf-8", errors="replace")
                            components.html(report_html, height=720, scrolling=True)
                        except Exception as e:
                            st.warning(f"리포트 HTML 로드 실패: {repr(e)}")
                    else:
                        st.warning("리포트 파일 존재하지 않음.")

                    st.link_button(
                        "다운로드",
                        url=f"{BACKEND_URL}/jobs/{job_id}/download",
                        use_container_width=True
                    )

            elif st.session_state.job_status == "error" and sdata:
                with report_placeholder.container():
                    st.error("분석 실패")
                    if sdata.get("error_message"):
                        st.code(sdata["error_message"], language="text")


        # fragment 실행
        live_panel(job_id, auto, manual_refresh)

        render_footer()


# ============================== 페이지 2) 이전 결과물 ==============================
elif st.session_state.page == "previous_outputs":
    st.title("이전 결과물")
    st.caption("outputs/jobs 아래에 누적된 결과를 리스트로 보여줍니다.")

    try:
        resp = requests.get(f"{BACKEND_URL}/outputs", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        jobs = data.get("jobs", [])

        if not jobs:
            st.info("아직 저장된 결과물이 없습니다.")
        else:
            for item in jobs:
                job_id = item["job_id"]
                with st.expander(f"Job: {job_id}", expanded=False):
                    # 다운로드는 done 여부를 모르니, 직접 job 상태 조회해서 done이면 버튼 제공
                    sresp = requests.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=10)
                    if sresp.ok:
                        sdata = sresp.json()
                        st.markdown(f"상태: {pill(sdata['status'])}", unsafe_allow_html=True)
                        if sdata["status"] == "done":
                            st.link_button(
                                "리포트 다운로드",
                                url=f"{BACKEND_URL}/jobs/{job_id}/download",
                                use_container_width=True
                            )
    except Exception as e:
        st.error(f"이전 결과물 로드 실패: {repr(e)}")

    render_footer()
