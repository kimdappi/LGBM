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

        log_placeholder = st.empty()
        status_placeholder = st.empty()
        report_placeholder = st.empty()

        # “실시간 느낌”을 위해 폴링(새 로그만)
        refresh_col1, refresh_col2 = st.columns([0.2, 0.8])
        with refresh_col1:
            if st.button("로그 새로고침", use_container_width=True):
                pass
        with refresh_col2:
            auto = st.checkbox("자동 갱신(2초)", value=True)

        # 1) 상태 조회
        try:
            sresp = requests.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=15)
            sresp.raise_for_status()
            sdata = sresp.json()
            st.session_state.job_status = sdata["status"]

            status_placeholder.markdown(
                f"**현재 상태:** {pill(sdata['status'])}",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"상태 조회 실패: {repr(e)}")

        # 2) 로그 조회(증분)
        try:
            from_bytes = st.session_state.get("log_from_bytes", 0)
            lresp = requests.get(f"{BACKEND_URL}/jobs/{job_id}/log", params={"from_bytes": from_bytes}, timeout=15)
            lresp.raise_for_status()
            ldata = lresp.json()
            new_text = ldata["text"]
            st.session_state.log_from_bytes = ldata["next_from_bytes"]

            if "log_text" not in st.session_state:
                st.session_state.log_text = ""
            st.session_state.log_text += new_text

            log_placeholder.markdown(
                f"<div class='log-box'>{st.session_state.log_text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('\\n','<br/>')}</div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"로그 조회 실패: {repr(e)}")

        st.divider()

        # 3) 완료 시 리포트 표시 + 다운로드
        if st.session_state.job_status == "done":
            try:
                sresp = requests.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=15)
                sresp.raise_for_status()
                sdata = sresp.json()
                report_path = sdata.get("report_path")

                st.success("분석 완료! 아래에서 리포트를 확인하고 다운로드할 수 있어.")

                if report_path and report_path.endswith(".html"):
                    report_file = Path(report_path)
                    html = report_file.read_text(encoding="utf-8", errors="replace")
                    components.html(html, height=720, scrolling=True)

                else:
                    st.warning("리포트 파일 존재하지 않음.")

                # 다운로드 버튼
                st.link_button(
                    "다운로드",
                    url=f"{BACKEND_URL}/jobs/{job_id}/download",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"리포트 표시 실패: {repr(e)}")

        elif st.session_state.job_status == "error":
            st.error("분석 실패")
            try:
                sresp = requests.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=15)
                sresp.raise_for_status()
                sdata = sresp.json()
                if sdata.get("error_message"):
                    st.code(sdata["error_message"], language="text")
            except Exception:
                pass

        # 자동 갱신
        if auto and st.session_state.job_status in ["queued", "running"]:
            time.sleep(2)
            st.rerun()

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
