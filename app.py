"""
SilverstoneLab — Electrochemical Analysis Suite
Streamlit web application  v3.1
Features: multi-user auth · project folders · AI expert commentary · PDF + Word reports
"""

import streamlit as st
import tempfile, os, sys, io, zipfile, re, json, datetime, base64
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# ── Persistent user-data directory ──────────────────────────
_DATA_DIR = Path(__file__).parent / "user_data"
_DATA_DIR.mkdir(exist_ok=True)

def _user_data_path(username: str) -> Path:
    return _DATA_DIR / f"{username}_projects.json"

def _projects_to_json(projects: dict) -> str:
    """Serialize projects to JSON — bytes → base64."""
    def _walk(v):
        if isinstance(v, bytes):
            return {"_b64": base64.b64encode(v).decode()}
        elif isinstance(v, dict):
            return {k: _walk(vv) for k, vv in v.items()}
        elif isinstance(v, list):
            return [_walk(i) for i in v]
        return v
    return json.dumps(_walk(projects), ensure_ascii=False)

def _projects_from_json(s: str) -> dict:
    """Deserialize projects from JSON — base64 → bytes."""
    def _walk(v):
        if isinstance(v, dict):
            if list(v.keys()) == ["_b64"]:
                return base64.b64decode(v["_b64"])
            return {k: _walk(vv) for k, vv in v.items()}
        elif isinstance(v, list):
            return [_walk(i) for i in v]
        return v
    return _walk(json.loads(s))

def _load_user_data(username: str) -> dict | None:
    """Load saved projects from disk. Returns None if no saved data."""
    p = _user_data_path(username)
    if not p.exists():
        return None
    try:
        return _projects_from_json(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _persist_user_data(username: str, projects: dict, current_project: str):
    """Save current projects to disk (called after every change)."""
    try:
        payload = {"projects": projects, "current_project": current_project}
        _user_data_path(username).write_text(
            _projects_to_json(payload), encoding="utf-8"
        )
    except Exception:
        pass  # persistence is best-effort; session_state still works

sys.path.insert(0, str(Path(__file__).parent))

from cv_module          import run_cv_analysis
from eis_module         import run_eis_analysis
from ph_module          import run_ph_analysis
from analytical_module  import run_analytical_performance, standard_addition, ich_q2_precision_validation
from ai_interpreter     import (interpret_cv, interpret_eis, interpret_ph,
                                  interpret_calibration, interpret_standard_addition,
                                  interpret_selectivity, interpret_stability, interpret_surface,
                                  interpret_full_project,
                                  interpret_dpv_calibration, interpret_real_sample,
                                  interpret_literature_comparison,
                                  interpret_dpv_optimization)
from report_generator   import generate_reports

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "SilverstoneLab | Echem Suite",
    page_icon   = "⚗️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
/* ── Google Font (Inter) via @import — link tags engelleniyor Streamlit sandbox'ta */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global font ───────────────────────────────────────── */
/* [data-testid] kasıtlı çıkarıldı: Streamlit'in Lucide ikonlarını bozuyor */
html, body, [class*="css"],
.stApp, .stMarkdown, .stText,
button, input, select, textarea,
p, li, td, th, label, caption, figcaption,
h1, h2, h3, h4, h5, h6
{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; }

/* Streamlit'in SVG/Lucide ikon elementlerini fonttan koru */
svg, svg *, [class*="icon"], [class*="Icon"],
span[data-baseweb], [role="img"] { font-family: inherit; }

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #132040 45%, #0e1a30 100%);
    border-right: 1px solid rgba(100,160,220,0.08);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #c2d4e8 !important; }
[data-testid="stSidebar"] h2 { color: #7ec8e3 !important; font-weight: 600; letter-spacing: 0.01em; }
[data-testid="stSidebar"] h3 { color: #90b8d8 !important; font-weight: 500; }

/* Sidebar radio / selectbox hover */
[data-testid="stSidebar"] .stRadio label:hover span,
[data-testid="stSidebar"] .stSelectbox label:hover { color: #aee4f7 !important; }

/* ── Main layout ─────────────────────────────────────────── */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* ── Section header ──────────────────────────────────────── */
.sec {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1a3a5c;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-bottom: 2px solid;
    border-image: linear-gradient(90deg, #1a3a5c 0%, #2b7cb8 60%, transparent 100%) 1;
    padding-bottom: 5px;
    margin-top: 1.4rem;
    margin-bottom: 0.6rem;
}

/* ── Native Streamlit metric widget — aynı tasarım dili ─── */
/* .metric-card CSS'i st.metric() native bileşenine uygulandı */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f4f8ff 0%, #eaf1fb 100%);
    border: 1px solid #c8daf0;
    border-radius: 12px;
    padding: 12px 16px !important;
    transition: box-shadow 0.2s;
}
[data-testid="stMetric"]:hover { box-shadow: 0 4px 16px rgba(27,74,145,0.12); }
[data-testid="stMetricLabel"] > div {
    font-size: 0.72rem !important;
    color: #5a7a9a !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: #1a3a5c !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
}

/* ── AI interpretation box ───────────────────────────────── */
.interp-box {
    background: linear-gradient(135deg, #f5f8ff 0%, #eef3fc 100%);
    border-left: 4px solid #2b7cb8;
    padding: 16px 20px;
    border-radius: 0 10px 10px 0;
    font-style: italic;
    line-height: 1.7;
    color: #2a3f5a;
    box-shadow: 2px 2px 10px rgba(43,124,184,0.07);
}

/* ── Login box ───────────────────────────────────────────── */
.login-box {
    max-width: 420px;
    margin: 4rem auto;
    padding: 2.4rem 2.8rem;
    border: 1px solid #c8daf0;
    border-radius: 16px;
    background: linear-gradient(160deg, #f8faff 0%, #f0f5fc 100%);
    text-align: center;
    box-shadow: 0 8px 32px rgba(26,58,92,0.10);
}

/* ── Status badge ────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-ok   { background: #d4edda; color: #1a6630; }
.badge-warn { background: #fff3cd; color: #856404; }
.badge-err  { background: #f8d7da; color: #721c24; }

/* ── Watermark ───────────────────────────────────────────── */
.wm {
    position: fixed;
    bottom: 12px;
    right: 16px;
    font-size: 10.5px;
    color: #1a3a5c;
    opacity: 0.22;
    font-style: italic;
    letter-spacing: 0.03em;
    pointer-events: none;
}

/* ── Streamlit button polish ─────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
</style>
<div class="wm">© SilverstoneLab</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════
def _load_config():
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def _save_config(cfg):
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

try:
    import streamlit_authenticator as stauth
    _AUTH_AVAILABLE = True
except ImportError:
    _AUTH_AVAILABLE = False

if _AUTH_AVAILABLE:
    _cfg = _load_config()
    _authenticator = stauth.Authenticate(
        _cfg["credentials"],
        _cfg["cookie"]["name"],
        _cfg["cookie"]["key"],
        _cfg["cookie"]["expiry_days"],
    )
    _auth_result = _authenticator.login(location="main")
    # Support both old (tuple) and new (None) return styles
    if isinstance(_auth_result, tuple):
        _auth_name, _auth_status, _auth_username = _auth_result
    else:
        _auth_name     = st.session_state.get("name")
        _auth_status   = st.session_state.get("authentication_status")
        _auth_username = st.session_state.get("username")

    if _auth_status is False:
        st.error("❌ Username or password is incorrect.")
        st.stop()
    elif _auth_status is None:
        st.markdown("""
        <div style='text-align:center;margin-top:3rem'>
        <h2>⚗️ SilverstoneLab</h2>
        <p style='color:#555'>Electrochemical Analysis Suite</p>
        <p style='color:#888;font-size:0.9rem'>Enter your credentials above to continue.</p>
        </div>""", unsafe_allow_html=True)
        st.stop()
    # ── logged in ───────────────────────────────────────────
    _current_user     = _auth_username or "guest"
    _current_name     = _auth_name or _current_user
    _current_role     = (_cfg["credentials"]["usernames"]
                         .get(_current_user, {}).get("role", "user"))
else:
    # Auth package not installed — allow access as guest (dev mode)
    _current_user = "guest"
    _current_name = "Guest"
    _current_role = "user"


# ═══════════════════════════════════════════════════════════
# SESSION STATE — PROJECTS  (scoped per logged-in user)
# ═══════════════════════════════════════════════════════════
def _user_key(suffix): return f"{_current_user}__{suffix}"

def _init_state():
    pk = _user_key("projects")
    cp = _user_key("current_project")
    if pk not in st.session_state:
        # First load in this browser session — try to restore from disk
        saved = _load_user_data(_current_user)
        if saved:
            st.session_state[pk] = saved.get("projects", {})
            saved_cp = saved.get("current_project", "")
            st.session_state[cp] = (saved_cp
                if saved_cp in st.session_state[pk]
                else next(iter(st.session_state[pk]), ""))
        else:
            st.session_state[pk] = {}
    if cp not in st.session_state or st.session_state[cp] not in st.session_state[pk]:
        if st.session_state.get(pk):
            st.session_state[cp] = list(st.session_state[pk].keys())[-1]
        else:
            _new_project(f"{_current_name}'s First Study")
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

def _new_project(name: str):
    pid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state[_user_key("projects")][pid] = {
        "name": name, "created": str(datetime.date.today()),
        "notes": "", "analyses": {}, "executive_summary": "",
        "owner": _current_user,
    }
    st.session_state[_user_key("current_project")] = pid
    _persist_user_data(_current_user, _projects(),
                       st.session_state[_user_key("current_project")])
    return pid

def _projects() -> dict:
    return st.session_state[_user_key("projects")]

def _cur() -> dict:
    return _projects().get(
        st.session_state.get(_user_key("current_project"), ""), {})

def _save_analysis(key: str, metrics: dict,
                   plot_dir: str | None = None,
                   param_df=None,
                   interpretation: str = "",
                   notes: str = ""):
    proj = _cur()
    if not proj: return
    plot_bytes = []
    if plot_dir:
        for f in sorted(Path(plot_dir).glob("*.png")):
            plot_bytes.append(f.read_bytes())
    proj["analyses"][key] = {
        "metrics":        metrics,
        "plot_bytes":     plot_bytes,
        "param_df_str":   param_df.to_string() if param_df is not None else "",
        "interpretation": interpretation,
        "notes":          notes,
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    # Auto-persist to disk after every analysis save
    _persist_user_data(_current_user, _projects(),
                       st.session_state.get(_user_key("current_project"), ""))


def _analysis_notes_widget(key: str) -> str:
    """Render a compact notes textarea for the given module key; return current value."""
    _existing = _cur().get("analyses", {}).get(key, {}).get("notes", "")
    _note_val = st.text_area(
        "📝 Analysis notes (optional)",
        value=st.session_state.get(f"anote_{key}", _existing),
        key=f"anote_{key}_widget",
        height=68,
        placeholder="Lab conditions, observations, deviations from protocol…",
        help="Stored with this analysis. Visible in project summary and exported ZIP.",
    )
    st.session_state[f"anote_{key}"] = _note_val
    return _note_val

_init_state()


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚗️ SilverstoneLab")
    st.markdown("**Electrochemical Analysis Suite**")
    st.caption("v3.1 — Projects · AI · Reports")
    st.divider()

    # ── User info + logout ───────────────────────────────────
    role_badge = "👑 Admin" if _current_role == "admin" else "👤"
    st.markdown(f"**{role_badge} {_current_name}**")
    if _AUTH_AVAILABLE:
        _authenticator.logout("🚪 Log out", location="sidebar")
    _keep_alive = st.checkbox(
        "🔁 Keep session alive",
        value=st.session_state.get("keep_alive", False),
        key="keep_alive_cb",
        help="When checked, the page refreshes silently every 4 minutes; "
             "the session stays open and no data is lost."
    )
    st.session_state["keep_alive"] = _keep_alive
    if _keep_alive:
        st.components.v1.html(
            """<script>
            /* Reload yerine hafif health ping — session sıfırlanmıyor */
            if (!window._silKA) {
                window._silKA = true;
                setInterval(function() {
                    fetch('/_stcore/health', {method: 'GET', cache: 'no-store'})
                        .catch(function() {});
                }, 180000); /* 3 dakikada bir */
            }
            </script>""",
            height=0,
        )
    st.divider()

    # ── Project management ───────────────────────────────────
    st.markdown("### 📁 My Projects")
    proj_names = {pid: p["name"] for pid,p in _projects().items()}
    if proj_names:
        cp_key = _user_key("current_project")
        cur_pid = st.selectbox("Active project", list(proj_names.keys()),
                               format_func=lambda x: proj_names[x],
                               index=list(proj_names.keys()).index(
                                   st.session_state.get(cp_key,""))
                               if st.session_state.get(cp_key,"") in proj_names else 0,
                               key="proj_select")
        if st.session_state.get(cp_key) != cur_pid:
            st.session_state[cp_key] = cur_pid
            _persist_user_data(_current_user, _projects(), cur_pid)
        st.session_state[cp_key] = cur_pid

    new_name = st.text_input("New project name", key="new_proj_name",
                              placeholder="e.g. Mn₃O₄-SPCE Study")
    if st.button("➕ Create project", use_container_width=True):
        if new_name.strip():
            _new_project(new_name.strip())
            st.rerun()

    with st.expander("✏️ Rename / add notes"):
        cp = _cur()
        new_proj_name = st.text_input("Rename", value=cp.get("name",""), key="rename_proj")
        new_notes     = st.text_area("Notes / electrode description",
                                      value=cp.get("notes",""), key="proj_notes", height=80)
        if st.button("Save", key="save_proj_meta"):
            if cp:
                cp["name"]  = new_proj_name or cp["name"]
                cp["notes"] = new_notes
                _persist_user_data(_current_user, _projects(),
                                   st.session_state.get(_user_key("current_project"), ""))
                st.success("Saved")

    with st.expander("💾 Export / Import project"):
        cp = _cur()
        if cp:
            export_data = {k: v for k,v in cp.items() if k != "analyses"}
            export_data["analyses_keys"] = list(cp.get("analyses",{}).keys())
            st.download_button("⬇ Export metadata (JSON)",
                               json.dumps(export_data, indent=2, default=str),
                               f"{cp.get('name','project')}.json",
                               "application/json")
        st.caption("Plots stay in session memory only. Export is metadata + interpretations.")

    with st.expander("🗑️ Delete project"):
        cp = _cur()
        cp_key = _user_key("current_project")
        cur_pid = st.session_state.get(cp_key, "")
        if cp:
            st.warning(f"This will permanently delete **{cp.get('name', cur_pid)}** and all its analyses.")
            confirm_del = st.checkbox("Yes, I want to delete this project", key="confirm_del_proj")
            if st.button("🗑️ Delete project", disabled=not confirm_del, key="btn_del_proj",
                         type="primary" if confirm_del else "secondary"):
                projs = _projects()
                projs.pop(cur_pid, None)
                # Switch to another project if available, else create a blank default
                remaining = list(projs.keys())
                if remaining:
                    st.session_state[cp_key] = remaining[0]
                else:
                    import uuid as _uuid
                    new_pid = str(_uuid.uuid4())
                    projs[new_pid] = {"name": "Default Project", "notes": "", "analyses": {}}
                    st.session_state[cp_key] = new_pid
                _persist_user_data(_current_user, _projects(), st.session_state[cp_key])
                st.rerun()
        else:
            st.info("No active project selected.")

    # ── Admin panel ─────────────────────────────────────────
    if _current_role == "admin":
        with st.expander("👑 Admin — Manage users"):
            st.caption("Add a new user to config.yaml")
            new_uname = st.text_input("Username (no spaces)", key="adm_uname")
            new_dname = st.text_input("Display name", key="adm_dname")
            new_email = st.text_input("Email (optional)", key="adm_email")
            new_pw    = st.text_input("Temporary password", key="adm_pw",
                                       type="password")
            if st.button("➕ Add user", key="adm_add"):
                if new_uname and new_pw:
                    try:
                        import bcrypt as _bcrypt
                        hashed = _bcrypt.hashpw(
                            new_pw.encode(), _bcrypt.gensalt(12)).decode()
                        cfg2 = _load_config()
                        cfg2["credentials"]["usernames"][new_uname] = {
                            "name": new_dname or new_uname,
                            "email": new_email or "",
                            "password": hashed,
                            "role": "user",
                        }
                        _save_config(cfg2)
                        st.success(f"✅ User '{new_uname}' added. "
                                   "They can log in after the next app restart.")
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.warning("Username and password are required.")

            st.divider()
            st.caption("Current users:")
            try:
                cfg3 = _load_config()
                _all_users = list(cfg3["credentials"]["usernames"].items())
                for uname, udata in _all_users:
                    role = udata.get("role","user")
                    badge = "👑" if role=="admin" else "👤"
                    _uc1, _uc2, _uc3 = st.columns([3, 2, 2])
                    _uc1.markdown(f"{badge} **{uname}** ({udata.get('name','')})")
                    # Change role button (don't allow demoting yourself)
                    if uname != _current_user:
                        _new_role = "user" if role == "admin" else "admin"
                        _role_lbl = "→ User" if role == "admin" else "→ Admin"
                        if _uc2.button(_role_lbl, key=f"adm_role_{uname}", use_container_width=True):
                            try:
                                _cfg_r = _load_config()
                                _cfg_r["credentials"]["usernames"][uname]["role"] = _new_role
                                _save_config(_cfg_r)
                                st.success(f"✅ {uname} is now '{_new_role}'. Effective after re-login.")
                                st.rerun()
                            except Exception as _re:
                                st.error(str(_re))
                        if _uc3.button("🗑 Remove", key=f"adm_del_{uname}", use_container_width=True):
                            if st.session_state.get(f"adm_del_confirm_{uname}"):
                                try:
                                    _cfg_d = _load_config()
                                    del _cfg_d["credentials"]["usernames"][uname]
                                    _save_config(_cfg_d)
                                    st.success(f"✅ User '{uname}' removed.")
                                    st.rerun()
                                except Exception as _de:
                                    st.error(str(_de))
                            else:
                                st.session_state[f"adm_del_confirm_{uname}"] = True
                                st.warning(f"⚠️ Click 🗑 Remove again to confirm deleting '{uname}'.")
                    else:
                        _uc2.caption("(you)")
            except Exception:
                pass

    st.divider()

    # ── AI Configuration ─────────────────────────────────────
    st.markdown("### 🤖 AI Configuration")
    # Session state defaults
    for _k, _d in [("api_key",""), ("openai_api_key",""), ("gemini_api_key",""),
                   ("ai_provider","Claude")]:
        if _k not in st.session_state:
            st.session_state[_k] = _d

    _ai_provider = st.selectbox("Provider", ["Claude (Anthropic)", "ChatGPT (OpenAI)", "Gemini (Google)"],
                                 key="ai_provider_select")
    st.session_state.ai_provider = _ai_provider

    if _ai_provider == "Claude (Anthropic)":
        _ak = st.text_input("Claude API key", value=st.session_state.api_key,
                             type="password", key="api_key_input", placeholder="sk-ant-...")
        st.session_state.api_key = _ak
        if _ak:
            st.caption("✅ Claude key active")
            _MODEL_OPTIONS = ["claude-opus-5-5","claude-sonnet-4-6",
                              "claude-sonnet-4-5","claude-3-5-sonnet-20241022"]
            if "ai_model" not in st.session_state:
                st.session_state.ai_model = _MODEL_OPTIONS[0]
            st.session_state.ai_model = st.selectbox(
                "Model", _MODEL_OPTIONS,
                index=_MODEL_OPTIONS.index(st.session_state.ai_model)
                      if st.session_state.ai_model in _MODEL_OPTIONS else 0,
                key="ai_model_select")
        else:
            st.caption("console.anthropic.com → API Keys")
            if "ai_model" not in st.session_state:
                st.session_state.ai_model = "claude-opus-5-5"

    elif _ai_provider == "ChatGPT (OpenAI)":
        _gk = st.text_input("OpenAI API key", value=st.session_state.openai_api_key,
                             type="password", key="gpt_key_input", placeholder="sk-...")
        st.session_state.openai_api_key = _gk
        if _gk:
            st.caption("✅ OpenAI key active")
            _GPT_FALLBACK = ["gpt-4o","gpt-4o-mini","gpt-4-turbo","gpt-3.5-turbo"]
            @st.cache_data(show_spinner=False, ttl=3600)
            def _fetch_openai_models(key: str):
                try:
                    import openai as _oai
                    _client = _oai.OpenAI(api_key=key)
                    _all = [m.id for m in _client.models.list()
                            if m.id.startswith("gpt-") and "instruct" not in m.id]
                    _all.sort(reverse=True)
                    return _all if _all else None
                except Exception:
                    return None
            _fetched_gpt = _fetch_openai_models(_gk)
            _GPT_MODELS = _fetched_gpt if _fetched_gpt else _GPT_FALLBACK
            if "gpt_model" not in st.session_state or st.session_state.gpt_model not in _GPT_MODELS:
                st.session_state.gpt_model = _GPT_MODELS[0]
            st.session_state.gpt_model = st.selectbox(
                "Model", _GPT_MODELS,
                index=_GPT_MODELS.index(st.session_state.gpt_model)
                      if st.session_state.gpt_model in _GPT_MODELS else 0,
                key="gpt_model_select")
            if _fetched_gpt:
                st.caption(f"📡 {len(_fetched_gpt)} model API'dan alındı")
        else:
            st.caption("platform.openai.com → API Keys")

    else:  # Gemini
        _gmk = st.text_input("Gemini API key", value=st.session_state.gemini_api_key,
                              type="password", key="gemini_key_input", placeholder="AIza...")
        st.session_state.gemini_api_key = _gmk
        if _gmk:
            st.caption("✅ Gemini key active")
            # Try to fetch available models from API; fall back to known list
            _GEM_FALLBACK = ["gemini-3.8-flash","gemini-2.5-flash","gemini-2.5-pro",
                             "gemini-2.0-flash","gemini-1.5-pro","gemini-1.5-flash"]
            @st.cache_data(show_spinner=False, ttl=3600)
            def _fetch_gemini_models(key: str):
                try:
                    import google.generativeai as _g
                    _g.configure(api_key=key)
                    _names = sorted(
                        [m.name.replace("models/", "") for m in _g.list_models()
                         if "generateContent" in getattr(m, "supported_generation_methods", [])
                         and "flash" in m.name or "pro" in m.name],
                        reverse=True)
                    return _names if _names else None
                except Exception:
                    return None
            _fetched = _fetch_gemini_models(_gmk)
            _GEM_MODELS = _fetched if _fetched else _GEM_FALLBACK
            if "gemini_model" not in st.session_state or st.session_state.gemini_model not in _GEM_MODELS:
                st.session_state.gemini_model = _GEM_MODELS[0]
            st.session_state.gemini_model = st.selectbox(
                "Model", _GEM_MODELS,
                index=_GEM_MODELS.index(st.session_state.gemini_model)
                      if st.session_state.gemini_model in _GEM_MODELS else 0,
                key="gemini_model_select")
            if _fetched:
                st.caption(f"📡 {len(_fetched)} model API'dan alındı")
        else:
            st.caption("aistudio.google.com → Get API key")

    # ── Plot Style ───────────────────────────────────────────
    with st.expander("🎨 Plot Style"):
        _PALETTES = {
            "Navy Classic":  ['#1a3a5c','#c55a11','#375623','#7030a0','#843c0c',
                              '#2b7cb8','#ff7f0e','#2ca02c','#d62728','#9467bd'],
            "Matplotlib":    ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                              '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'],
            "Vibrant":       ['#E63946','#457B9D','#2D6A4F','#F4A261','#7B2D8B',
                              '#E9C46A','#264653','#E76F51','#06D6A0','#118AB2'],
            "Earthy":        ['#8B4513','#228B22','#4169E1','#DC143C','#FF8C00',
                              '#6A0DAD','#20B2AA','#B8860B','#708090','#C71585'],
            "Grayscale":     ['#111111','#333333','#555555','#777777','#999999',
                              '#444444','#666666','#888888','#AAAAAA','#BBBBBB'],
        }
        if "plot_palette" not in st.session_state:
            st.session_state.plot_palette = "Navy Classic"
        _pal = st.selectbox("Color palette", list(_PALETTES.keys()),
                            index=list(_PALETTES.keys()).index(st.session_state.plot_palette),
                            key="plot_palette_sel")
        st.session_state.plot_palette = _pal

        # Renk önizlemesi
        _prev_colors = _PALETTES[_pal][:5]
        _swatch = "".join(
            f'<span style="display:inline-block;width:22px;height:14px;'
            f'background:{c};border-radius:3px;margin:1px;"></span>'
            for c in _prev_colors
        )
        st.markdown(_swatch, unsafe_allow_html=True)

        _lw_def = st.session_state.get("plot_linewidth", 2.0)
        _lw = st.slider("Line width", 0.5, 5.0, _lw_def, 0.5, key="plot_lw_sl")
        st.session_state.plot_linewidth = _lw

        _LS_OPTIONS = ["─── Solid", "--- Dashed", "··· Dotted", "-·- Dash-dot"]
        _LS_MAP     = {"─── Solid": "-", "--- Dashed": "--", "··· Dotted": ":", "-·- Dash-dot": "-."}
        _ls_def = st.session_state.get("plot_linestyle_label", "--- Dashed")
        _ls_lbl = st.radio("Fit line style", _LS_OPTIONS,
                           index=_LS_OPTIONS.index(_ls_def),
                           key="plot_ls_radio", horizontal=True)
        st.session_state.plot_linestyle_label = _ls_lbl

        _ms_def = int(st.session_state.get("plot_markersize", 60))
        _ms = st.slider("Marker size", 10, 150, _ms_def, 10, key="plot_ms_sl")
        st.session_state.plot_markersize = _ms

        # Build the dict every time sidebar renders
        st.session_state.plot_style = {
            "colors":        _PALETTES[_pal],
            "linewidth":     _lw,
            "fit_linestyle": _LS_MAP[_ls_lbl],
            "marker_size":   _ms,
        }

    st.divider()

    # ── Navigation ───────────────────────────────────────────
    _NAV_OPTIONS = [
        "🏠  Home",
        "📈  CV — Scan Rate Study",
        "🔵  EIS — Impedance",
        "🧪  pH — Working pH",
        "📊  Calibration & LOD/LOQ",
        "➕  Standard Addition",
        "🧬  Selectivity",
        "📅  Stability",
        "🔬  Surface Characterization",
        "⚡  DPV — Sensor Calibration",
        "🔧  DPV — Optimization",
        "🧫  Real Sample Analysis",
        "📚  Literature Comparison",
        "📄  Report",
    ]
    _NAV_KEY     = f"nav_radio_{_current_user}"
    _NAV_PENDING = f"nav_pending_{_current_user}"
    # Buton tıklamasından gelen bekleyen navigasyonu radio'ya ÖNCE uygula
    if _NAV_PENDING in st.session_state:
        st.session_state[_NAV_KEY] = st.session_state.pop(_NAV_PENDING)
    page = st.radio(f"navigate_{_current_user}", label_visibility="collapsed",
                    options=_NAV_OPTIONS, key=_NAV_KEY)

    with st.expander("📋 CSV format guide"):
        st.markdown("""
| Module | Required columns | Units |
|---|---|---|
| **CV** | `Potential_V`, `Current_A` | V · A — one file per scan rate |
| **EIS** | `Freq_Hz`, `Zreal_Ohm`, `Zimag_Ohm` | Hz · Ω |
| **pH** | `pH`, `Ep_mV` | — · mV |
| **Calibration** | `concentration_uM`, `ip_run1_uA`, `ip_run2_uA`, … | µM · µA |
| **Std Addition** | `added_conc_uM`, `ip_uA` | µM · µA |
| **Stability** | `Day`, `Signal_uA`, `SD_uA` | — · µA |
| **Surface EIS** | `Zreal_Ohm`, `Zimag_Ohm` | Ω |
| **Surface CV** | `Potential_V`, `Current_uA` | V · µA |
| **DPV Cal.** | `Potential_V`, `Current_uA` | V · µA — one file per concentration |

💡 Use the **⬇️ Download CSV Template** buttons in each module for pre-filled examples.
        """)



# ═══════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════
def _zip_dir(d):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(Path(d).iterdir()):
            if f.is_file(): zf.write(f, f.name)
    buf.seek(0); return buf.read()

def _show_imgs(d):
    imgs = sorted([f for f in Path(d).iterdir() if f.suffix=='.png'])
    if not imgs: st.info("No plots were generated."); return
    cols = st.columns(min(len(imgs),2))
    for i,img in enumerate(imgs):
        cols[i%2].image(str(img), use_container_width=True)

def _dl_btn(d, name):
    st.download_button("⬇ Download results (ZIP)", _zip_dir(d),
                       name, "application/zip", use_container_width=True)

def _rate_from_name(n):
    m = re.search(r'(\d+)\s*m[Vv]', n)
    return int(m.group(1)) if m else None

def _input_mode(key):
    return st.radio("Data input", ["📁 Upload CSV", "✏️ Enter manually"],
                    horizontal=True, key=key)

def _csv_template_btn(csv_text: str, file_name: str, label: str = "⬇️ Download CSV Template"):
    """Render a small download button for a CSV template."""
    st.download_button(
        label=label,
        data=csv_text.encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=False,
    )

def _metrics_csv_btn(metrics: dict, file_name: str):
    """Render a download button that exports the metrics dict as a CSV."""
    if not metrics:
        return
    import io as _io
    _buf = _io.StringIO()
    _buf.write("Metric,Value\n")
    for k, v in metrics.items():
        _buf.write(f"{str(k).replace(',','')},{str(v).replace(',','')}\n")
    st.download_button(
        label="📥 Download metrics as CSV",
        data=_buf.getvalue().encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=False,
    )

def _csv_preview(uploaded_file, n_rows: int = 5, key_prefix: str = "prev"):
    """Show first n rows of an uploaded CSV inside an expander."""
    if uploaded_file is None:
        return
    try:
        uploaded_file.seek(0)
        _df_prev = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        with st.expander(f"🔍 Data preview — {uploaded_file.name} ({len(_df_prev)} rows × {len(_df_prev.columns)} columns)", expanded=False):
            st.caption(f"**Detected columns:** {', '.join(f'`{c}`' for c in _df_prev.columns)}")
            st.dataframe(_df_prev.head(n_rows), use_container_width=True, hide_index=True)
            if len(_df_prev) > n_rows:
                st.caption(f"Showing first {n_rows} of {len(_df_prev)} rows.")
    except Exception as _e:
        st.warning(f"⚠️ Could not preview `{uploaded_file.name}`: {_e}")

def _ai_interpret_btn(module_key: str, interp_fn, fn_kwargs: dict,
                      result_key: str = "interpretation",
                      btn_key: str = None,
                      images: list = None):
    """Renders the 🤖 Interpret button and shows/saves commentary.
    images: optional list of Streamlit UploadedFile objects (for vision-capable models).
    """
    _provider = st.session_state.get("ai_provider", "Claude (Anthropic)")

    # Resolve active key and model
    if _provider == "ChatGPT (OpenAI)":
        _active_key   = st.session_state.get("openai_api_key", "")
        _active_model = st.session_state.get("gpt_model", "gpt-4o")
        _provider_label = "ChatGPT"
    elif _provider == "Gemini (Google)":
        _active_key   = st.session_state.get("gemini_api_key", "")
        _active_model = st.session_state.get("gemini_model", "gemini-3.8-flash")
        _provider_label = "Gemini"
    else:
        _active_key   = st.session_state.get("api_key", "")
        _active_model = st.session_state.get("ai_model", "claude-opus-5-5")
        _provider_label = "Claude"

    if not _active_key:
        st.info(f"🤖 Add your {_provider_label} API key in the sidebar to enable AI interpretation.")
        return

    _btn_key = btn_key if btn_key else f"ai_{module_key}"
    if st.button(f"🤖 Generate AI Interpretation ({_provider_label})",
                 use_container_width=True, key=_btn_key):
        with st.spinner(f"Consulting the electrochemistry expert via {_provider_label}…"):
            try:
                _imgs = [img for img in (images or []) if img is not None]
                if _provider == "ChatGPT (OpenAI)":
                    text = _call_openai(_active_key, _active_model, interp_fn, fn_kwargs,
                                        images=_imgs)
                elif _provider == "Gemini (Google)":
                    text = _call_gemini(_active_key, _active_model, interp_fn, fn_kwargs,
                                        images=_imgs)
                else:
                    text = _call_claude_vision(_active_key, _active_model, interp_fn, fn_kwargs,
                                               images=_imgs)
                proj = _cur()
                if module_key in proj.get("analyses", {}):
                    proj["analyses"][module_key]["interpretation"] = text
                st.session_state[f"interp_{module_key}"] = text
            except Exception as e:
                import traceback
                st.error(f"AI error: {e}")
                with st.expander("🔍 Full error details"):
                    st.code(traceback.format_exc())

    saved = (_cur().get("analyses", {}).get(module_key, {}).get("interpretation", "")
             or st.session_state.get(f"interp_{module_key}", ""))
    if saved:
        st.markdown('<div class="sec">Expert Interpretation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="interp-box">{saved.replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# VISION HELPER
# ═══════════════════════════════════════════════════════════
def _images_to_b64(images: list) -> list:
    """Convert UploadedFile list → list of (b64_str, mime_type) tuples."""
    import base64, imghdr
    result = []
    for img in images:
        img.seek(0)
        raw = img.read()
        img.seek(0)
        hdr = imghdr.what(None, h=raw)
        mime = {"jpeg":"image/jpeg","jpg":"image/jpeg","png":"image/png",
                "tiff":"image/tiff","gif":"image/gif"}.get(hdr or "", "image/png")
        result.append((base64.b64encode(raw).decode(), mime))
    return result


# ═══════════════════════════════════════════════════════════
# AI PROVIDER BRIDGES  (Claude · OpenAI · Gemini)
# ═══════════════════════════════════════════════════════════
def _call_openai(api_key: str, model: str, interp_fn, fn_kwargs: dict,
                 images: list = None) -> str:
    """Bridge: build the same user prompt via interp_fn dummy, then call OpenAI."""
    try:
        import openai as _oai
    except ImportError:
        return ("⚠️ 'openai' package not installed.\n"
                "Add `openai>=1.0` to requirements.txt and redeploy.")
    from ai_interpreter import _SYSTEM
    import ai_interpreter as _ai
    _orig_call = _ai._call
    _captured_prompt = {}
    def _capture(ak, msg, mdl=""):
        _captured_prompt["msg"] = msg
        return "__captured__"
    _ai._call = _capture
    try:
        result = interp_fn(api_key="__dummy__", model="__dummy__", **fn_kwargs)
    finally:
        _ai._call = _orig_call
    if "__captured__" not in str(result) and result != "__captured__":
        return result
    user_msg = _captured_prompt.get("msg", "Interpret the electrochemical data provided.")
    client = _oai.OpenAI(api_key=api_key)
    # Build user content — add images if provided (gpt-4o and later support vision)
    if images:
        b64_imgs = _images_to_b64(images)
        user_content = [{"type": "text", "text": user_msg}]
        for b64, mime in b64_imgs:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}
            })
    else:
        user_content = user_msg
    resp = client.chat.completions.create(
        model=model, max_tokens=1500,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user",   "content": user_content}]
    )
    return resp.choices[0].message.content.strip()


def _call_gemini(api_key: str, model: str, interp_fn, fn_kwargs: dict,
                 images: list = None) -> str:
    """Bridge: build prompt via interp_fn, then call Google Gemini (with optional vision)."""
    try:
        import google.generativeai as _genai
    except ImportError:
        return ("⚠️ 'google-generativeai' package not installed.\n"
                "Add `google-generativeai>=0.5` to requirements.txt and redeploy.")
    from ai_interpreter import _SYSTEM
    import ai_interpreter as _ai
    _orig_call = _ai._call
    _captured_prompt = {}
    def _capture(ak, msg, mdl=""):
        _captured_prompt["msg"] = msg
        return "__captured__"
    _ai._call = _capture
    try:
        result = interp_fn(api_key="__dummy__", model="__dummy__", **fn_kwargs)
    finally:
        _ai._call = _orig_call
    if "__captured__" not in str(result) and result != "__captured__":
        return result
    user_msg = _captured_prompt.get("msg", "Interpret the electrochemical data provided.")
    _genai.configure(api_key=api_key)
    _gmodel = _genai.GenerativeModel(model_name=model, system_instruction=_SYSTEM)
    if images:
        import io
        from PIL import Image as _PILImage
        content_parts = [user_msg]
        for img_file in images:
            img_file.seek(0)
            pil_img = _PILImage.open(io.BytesIO(img_file.read()))
            img_file.seek(0)
            content_parts.append(pil_img)
        resp = _gmodel.generate_content(content_parts)
    else:
        resp = _gmodel.generate_content(user_msg)
    return resp.text.strip()


def _call_claude_vision(api_key: str, model: str, interp_fn, fn_kwargs: dict,
                        images: list = None) -> str:
    """Call Claude directly — with optional SEM image attachments (vision)."""
    if not images:
        # No images: use the standard interp_fn path
        return interp_fn(api_key=api_key, model=model, **fn_kwargs)
    # With images: build a messages-style request using the Anthropic SDK directly
    try:
        import anthropic as _ant
    except ImportError:
        return ("⚠️ 'anthropic' package not installed.")
    from ai_interpreter import _SYSTEM
    import ai_interpreter as _ai
    # Capture the text prompt
    _orig_call = _ai._call
    _captured_prompt = {}
    def _capture(ak, msg, mdl=""):
        _captured_prompt["msg"] = msg
        return "__captured__"
    _ai._call = _capture
    try:
        result = interp_fn(api_key="__dummy__", model="__dummy__", **fn_kwargs)
    finally:
        _ai._call = _orig_call
    if "__captured__" not in str(result) and result != "__captured__":
        return result
    user_msg = _captured_prompt.get("msg", "Interpret the electrochemical data provided.")
    # Build multimodal content
    b64_imgs = _images_to_b64(images)
    user_content = []
    for b64, mime in b64_imgs:
        user_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64}
        })
    user_content.append({"type": "text", "text": user_msg})
    client = _ant.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_content}]
    )
    return resp.content[0].text.strip()


# ═══════════════════════════════════════════════════════════
# SYNTHESIS helpers (manual → synthetic raw data)
# ═══════════════════════════════════════════════════════════
def _synth_cv(rate_mVs, ip_a_uA, ip_c_uA, Epa_V, Epc_V, n=200):
    E = np.linspace(min(Epa_V,Epc_V)-0.4, max(Epa_V,Epc_V)+0.4, n)
    s = 0.045
    I = (ip_a_uA*1e-6 * np.exp(-0.5*((E-Epa_V)/s)**2)
       - abs(ip_c_uA)*1e-6 * np.exp(-0.5*((E-Epc_V)/s)**2))
    return pd.DataFrame({'Potential': E, 'Current': I})

def _synth_eis(Rs, Rct, Q, n_cpe, f_min=0.01, f_max=1e5, n_pts=50):
    freqs   = np.logspace(np.log10(f_min), np.log10(f_max), n_pts)
    omega   = 2 * np.pi * freqs
    Z_cpe2  = Q * (1j*omega)**n_cpe
    sigma   = 20.0
    Z_total = Rs + 1.0/(Z_cpe2 + 1.0/(Rct + sigma*(1-1j)/np.sqrt(omega)))
    return pd.DataFrame({'Freq': freqs, 'Zreal': Z_total.real,
                         'Zimag': -Z_total.imag})


# ═══════════════════════════════════════════════════════════
# 🏠  HOME
# ═══════════════════════════════════════════════════════════
if page == "🏠  Home":
    proj = _cur()
    _NAV_KEY     = f"nav_radio_{_current_user}"
    _NAV_PENDING = f"nav_pending_{_current_user}"

    st.title(f"⚗️ {proj.get('name','My Study')}")
    st.caption(f"Created {proj.get('created','')}  ·  SilverstoneLab Echem Suite  ·  Welcome, {_current_name}")
    if proj.get("notes"):
        st.info(proj["notes"])

    # ── Completed analyses status bar ────────────────────────
    done = proj.get("analyses", {})
    _labels = {"cv":"📈 CV","eis":"🔵 EIS","ph":"🧪 pH",
               "cal":"📊 Cal","sa":"➕ SA"}
    if done:
        st.success(f"✅ **{len(done)} analysis completed** in this project  ·  "
                   + "  ".join(f"{_labels.get(k,k)} ✓" for k in done)
                   + "  ·  Go to **📄 Report** to export")

    st.divider()
    st.markdown("#### Choose a module to begin")

    # ── CSS for tile cards ───────────────────────────────────
    st.markdown("""
<style>
.tile-card {
    border: 2px solid #dce8f4;
    border-radius: 14px;
    padding: 20px 14px 14px 14px;
    text-align: center;
    background: linear-gradient(145deg, #f4f8ff, #e8f1fb);
    margin-bottom: 6px;
    min-height: 118px;
}
.tile-icon  { font-size: 2.4rem; line-height: 1; }
.tile-title { font-size: 1.05rem; font-weight: 700; color: #1a3f6b; margin: 7px 0 3px; }
.tile-desc  { font-size: 0.78rem; color: #5a7a9a; line-height: 1.35; }
.tile-done  { border-color: #28a745; background: linear-gradient(145deg, #f0fff4, #e0f7e9); }
</style>
""", unsafe_allow_html=True)

    _MODULES = [
        ("📈", "CV Analysis",         "ECSA · b-value · k° · E½",          "📈  CV — Scan Rate Study",    "cv"),
        ("🔵", "EIS Impedance",       "Rct · CPE · roughness factor",       "🔵  EIS — Impedance",         "eis"),
        ("🧪", "pH Study",            "Ep/pH slope · Nernst · optimal pH",  "🧪  pH — Working pH",         "ph"),
        ("📊", "Calibration",         "LOD · LOQ · linear range · r",       "📊  Calibration & LOD/LOQ",   "cal"),
        ("➕", "Standard Addition",   "C_sample · recovery · 95 % PI",      "➕  Standard Addition",        "sa"),
        ("🧬", "Selectivity",         "Interferents · % signal change",      "🧬  Selectivity",              "sel"),
        ("📅", "Stability",           "Daily % recovery · RSD trend",        "📅  Stability",                "stab"),
        ("🔬", "Surface Charact.",    "SEM · EIS · bare vs modified CV",     "🔬  Surface Characterization", "surf"),
        ("⚡", "DPV Calibration",    "LOD · LOQ · sensitivity · linear range", "⚡  DPV — Sensor Calibration", "dpv"),
        ("🔧", "DPV Optimization",  "OFAT · pulse amp · step · tacc · modifier", "🔧  DPV — Optimization", "dpvopt"),
        ("🧫", "Real Sample",        "Spike-recovery · % RSD · matrices",   "🧫  Real Sample Analysis",     "realsample"),
        ("📚", "Literature Table",   "Compare LOD vs published sensors",    "📚  Literature Comparison",    "litcomp"),
        ("📄", "Report",              "PDF + Word  ·  all plots embedded",   "📄  Report",                  None),
    ]

    # Row 1 — 3 modules
    _cols1 = st.columns(3, gap="medium")
    for _i, (_icon, _title, _desc, _nav, _key) in enumerate(_MODULES[:3]):
        _done_cls = " tile-done" if _key and _key in done else ""
        _done_lbl = " ✓" if _key and _key in done else ""
        with _cols1[_i]:
            st.markdown(f"""
<div class="tile-card{_done_cls}">
  <div class="tile-icon">{_icon}</div>
  <div class="tile-title">{_title}{_done_lbl}</div>
  <div class="tile-desc">{_desc}</div>
</div>""", unsafe_allow_html=True)
            if st.button(f"Open {_title}", key=f"home_nav_{_i}", use_container_width=True):
                st.session_state[_NAV_PENDING] = _nav
                st.rerun()

    st.write("")  # spacer

    # Row 2 — modules 3-5
    _cols2 = st.columns(3, gap="medium")
    for _i, (_icon, _title, _desc, _nav, _key) in enumerate(_MODULES[3:6]):
        _done_cls = " tile-done" if _key and _key in done else ""
        _done_lbl = " ✓" if _key and _key in done else ""
        with _cols2[_i]:
            st.markdown(f"""
<div class="tile-card{_done_cls}">
  <div class="tile-icon">{_icon}</div>
  <div class="tile-title">{_title}{_done_lbl}</div>
  <div class="tile-desc">{_desc}</div>
</div>""", unsafe_allow_html=True)
            if st.button(f"Open {_title}", key=f"home_nav_{_i+3}", use_container_width=True):
                st.session_state[_NAV_PENDING] = _nav
                st.rerun()

    st.write("")  # spacer

    # Row 3 — modules 6+
    _remaining = _MODULES[6:]
    if _remaining:
        _cols3 = st.columns(3, gap="medium")
        for _i, (_icon, _title, _desc, _nav, _key) in enumerate(_remaining):
            _done_cls = " tile-done" if _key and _key in done else ""
            _done_lbl = " ✓" if _key and _key in done else ""
            with _cols3[_i % 3]:
                st.markdown(f"""
<div class="tile-card{_done_cls}">
  <div class="tile-icon">{_icon}</div>
  <div class="tile-title">{_title}{_done_lbl}</div>
  <div class="tile-desc">{_desc}</div>
</div>""", unsafe_allow_html=True)
                if st.button(f"Open {_title}", key=f"home_nav_{_i+6}", use_container_width=True):
                    st.session_state[_NAV_PENDING] = _nav
                    st.rerun()

    # ── Project analysis summary ──────────────────────────────
    _home_proj = _cur()
    _home_analyses = _home_proj.get("analyses", {})
    if _home_analyses:
        st.write("")
        st.markdown('<div class="sec">📋 Project Analysis Summary</div>', unsafe_allow_html=True)
        _MOD_LABELS = {
            "cv":"📈 CV","eis":"🔵 EIS","ph":"🧪 pH",
            "cal":"📊 Calibration","sa":"➕ Std Addition",
            "sel":"🧬 Selectivity","stab":"📅 Stability","surf":"🔬 Surface Charact."
        }
        # Key metrics to surface per module
        _HIGHLIGHT_KEYS = {
            "cv":   ["ECSA (cm²)", "k0 (cm/s)"],
            "eis":  ["Rct (Ω)", "Rs (Ω)"],
            "ph":   ["Slope (mV/pH)", "R²"],
            "cal":  ["LOD (µM)", "LOQ (µM)", "R²"],
            "sa":   ["Determined concentration (µM)", "Recovery (%)"],
            "sel":  [],
            "stab": ["Last % Recovery", "Min. threshold"],
            "surf": [],
        }
        _sum_cols = st.columns(min(len(_home_analyses), 4), gap="small")
        for _si, (_mk, _ma) in enumerate(_home_analyses.items()):
            with _sum_cols[_si % 4]:
                _has_interp = bool(_ma.get("interpretation",""))
                _lbl = _MOD_LABELS.get(_mk, _mk)
                _ts  = _ma.get("timestamp","")[:10] if _ma.get("timestamp") else ""
                _note_flag = "📝 " if _ma.get("notes","") else ""
                _ai_flag   = "🤖 " if _has_interp else ""
                st.markdown(f"""<div style='border:1px solid #c8d8e8;border-radius:8px;padding:10px 12px;background:#f7fafd;margin-bottom:8px'>
<b>{_lbl}</b><br><small style='color:#666'>{_ts} {_ai_flag}{_note_flag}</small>""",
                            unsafe_allow_html=True)
                _mets = _ma.get("metrics", {})
                _shown = 0
                for _hk in _HIGHLIGHT_KEYS.get(_mk, []):
                    _v = _mets.get(_hk)
                    if _v is not None:
                        st.caption(f"**{_hk}:** {_v}")
                        _shown += 1
                        if _shown >= 2:
                            break
                if _shown == 0 and _mets:
                    # Fallback: first two scalar metrics
                    for _fk, _fv in _mets.items():
                        if isinstance(_fv, (int, float, str)) and _shown < 2:
                            st.caption(f"**{_fk}:** {_fv}")
                            _shown += 1
                st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 📈  CV
# ═══════════════════════════════════════════════════════════
elif page == "📈  CV — Scan Rate Study":
    st.title("📈 CV — Scan Rate Study")
    st.caption(f"Project: **{_cur().get('name','')}**")

    left,right = st.columns([1,1])
    with left:
        st.markdown('<div class="sec">Parameters</div>', unsafe_allow_html=True)
        cv_analyte = st.text_input("Electrode / analyte name",
                                    value=st.session_state.get("cv_analyte", _cur().get("name","Modified Electrode")),
                                    key="cv_analyte_inp",
                                    help="Name used in AI interpretation and reports (e.g. 'Mn₃O₄/SPCE for dopamine').")
        st.session_state["cv_analyte"] = cv_analyte
        conc_mM    = st.number_input("Probe concentration (mM)", value=1.0,
                                      help="Concentration of the redox probe in solution (e.g. 1 mM [Fe(CN)₆]³⁻). Used for ECSA calculation via Randles–Ševčík.")
        diff_coeff = st.number_input("D (cm²/s)", value=7.6e-6, format="%.2e",
                                      help="Diffusion coefficient of the redox probe in cm²/s. Literature value for [Fe(CN)₆]³⁻ in 0.1 M KCl ≈ 7.6×10⁻⁶ cm²/s.")
        n_el       = st.number_input("n (electrons)", value=1, min_value=1, max_value=6,
                                      help="Number of electrons transferred in the redox reaction (e.g. n=1 for Fe(CN)₆³⁻/⁴⁻, n=2 for hydroquinone). Used in Randles–Ševčík and Butler–Volmer equations.")
    with right:
        st.markdown('<div class="sec">Input mode</div>', unsafe_allow_html=True)
        cv_mode = _input_mode("cv_mode")

    disk_map = {}
    tmpdir   = tempfile.mkdtemp()

    if cv_mode == "📁 Upload CSV":
        _csv_template_btn(
            "Potential_V,Current_A\n"
            "-0.6,1.23e-6\n-0.5,8.45e-7\n-0.4,3.21e-7\n-0.3,-1.05e-7\n-0.2,-5.67e-7\n"
            "-0.1,-9.12e-7\n0.0,-1.24e-6\n0.1,-1.45e-6\n0.2,-1.52e-6\n0.3,-1.41e-6\n"
            "0.4,-1.18e-6\n0.5,-7.34e-7\n0.6,-2.11e-7\n",
            "cv_template_50mVs.csv",
            "⬇️ Download CV Template (name file e.g. cv_50mVs.csv)"
        )
        cv_ups = st.file_uploader("CV files (one per scan rate)",
                                   type="csv", accept_multiple_files=True)
        rate_map = {}
        if cv_ups:
            for _f_prev in cv_ups:
                _csv_preview(_f_prev, key_prefix=f"cv_{_f_prev.name}")
            st.markdown('<div class="sec">Assign scan rates (mV/s)</div>',
                        unsafe_allow_html=True)
            for f in cv_ups:
                auto = _rate_from_name(f.name)
                r = st.number_input(f"`{f.name}`",
                                     value=float(auto) if auto else 100.0,
                                     min_value=1.0, key=f"cv_r_{f.name}")
                rate_map[r] = f
            for rate,fobj in rate_map.items():
                p = os.path.join(tmpdir, f"cv_{int(rate)}mVs.csv")
                Path(p).write_bytes(fobj.getvalue())
                disk_map[rate] = p
            # ── Renk seçiciler ───────────────────────────────────────────
            if rate_map:
                _cv_base = st.session_state.get("plot_style",{}).get(
                    "colors", ['#1a3a5c','#c55a11','#375623','#7030a0',
                               '#843c0c','#1f77b4','#ff7f0e','#2ca02c'])
                st.markdown('<div class="sec">🎨 Dataset Colors</div>',
                            unsafe_allow_html=True)
                _sorted_rates = sorted(rate_map.keys())
                _ncols        = min(len(_sorted_rates), 4)
                _cv_color_cols= st.columns(_ncols)
                _cv_colors    = []
                for _ci, _rate in enumerate(_sorted_rates):
                    _def = _cv_base[_ci % len(_cv_base)]
                    _pk  = _cv_color_cols[_ci % _ncols].color_picker(
                        f"{int(_rate)} mV/s", _def,
                        key=f"cv_col_{int(_rate)}")
                    _cv_colors.append(_pk)
                st.session_state["cv_custom_colors"] = _cv_colors
    else:
        st.markdown('<div class="sec">Enter peak data</div>', unsafe_allow_html=True)
        default_cv = pd.DataFrame({
            'scan_rate_mVs': [10.0,25.0,50.0,100.0,200.0],
            'ip_anodic_uA':  [None]*5,'ip_cathodic_uA':[None]*5,
            'Epa_V':[None]*5,'Epc_V':[None]*5,
        })
        cv_tbl = st.data_editor(default_cv, num_rows="dynamic",
                                 use_container_width=True, key="cv_tbl")
        for _, row in cv_tbl.dropna(subset=['ip_anodic_uA','Epa_V','Epc_V']).iterrows():
            rate = float(row['scan_rate_mVs'])
            df   = _synth_cv(rate,float(row['ip_anodic_uA']),
                              float(row.get('ip_cathodic_uA') or row['ip_anodic_uA']),
                              float(row['Epa_V']),float(row['Epc_V']))
            p = os.path.join(tmpdir,f"cv_{int(rate)}mVs.csv")
            df.to_csv(p,index=False); disk_map[rate] = p

    if disk_map:
        if "cv" in _cur().get("analyses", {}):
            st.warning("⚠️ A CV analysis is already saved for this project. Running again will overwrite it.")
    if disk_map and st.button("▶ Run CV Analysis",type="primary",use_container_width=True):
        with st.spinner("⚙️ Analysing CV data — ECSA, mechanism, k°…"):
            out = os.path.join(tmpdir,"cv_out"); os.makedirs(out,exist_ok=True)
            try:
                _cv_ps = dict(st.session_state.get("plot_style") or {})
                if st.session_state.get("cv_custom_colors"):
                    _cv_ps["colors"] = st.session_state["cv_custom_colors"]
                res = run_cv_analysis(disk_map,conc_mM,diff_coeff,int(n_el),out,
                                      plot_style=_cv_ps)
                st.success("✅ Done — results saved to project")
                ecsa = res.get('ecsa',{}); mech = res.get('mechanism',{})
                k0   = res.get('k0',{})
                b    = mech.get('slope_b',0.75)
                lbl  = "Diffusion" if b<0.6 else ("Adsorption" if b>0.9 else "Mixed")
                k0v  = k0.get('mean')
                metrics = {
                    "ECSA (cm²)": f"{ecsa.get('ECSA_cm2',0):.4f}",
                    "ECSA R²":    f"{ecsa.get('R2',0):.5f}",
                    "b value":    f"{b:.4f}",
                    "Mechanism":  lbl,
                    "k° (cm/s)":  f"{k0v:.4e}" if k0v else "N/A",
                }
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("ECSA (cm²)", metrics["ECSA (cm²)"])
                m2.metric("R²",         metrics["ECSA R²"])
                m3.metric("b value",    f"{b:.4f}  ({lbl})")
                m4.metric("k° (cm/s)",  metrics["k° (cm/s)"])
                _cv_notes = _analysis_notes_widget("cv")
                _save_analysis("cv", metrics, out, notes=_cv_notes)
                _metrics_csv_btn(metrics, "cv_metrics.csv")
                st.markdown('<div class="sec">Plots</div>', unsafe_allow_html=True)
                _show_imgs(out); _dl_btn(out,"cv_analysis.zip")
                _ai_interpret_btn("cv", interpret_cv,
                                   {"results": res,
                                    "electrode_name": st.session_state.get("cv_analyte", _cur().get("name",""))})
            except ValueError as e:
                st.error(str(e))
                st.markdown("""
**CSV Format Checklist:**
- First column: `Potential` (V)
- Second column: `Current` (A)
- Scan rate in filename (e.g. `cv_50mVs.csv`, `50.csv`)
- Anodic peak should be clearly visible in the scan direction
""")
            except Exception as e:
                st.error(str(e))
                import traceback; st.code(traceback.format_exc())
    elif _cur().get("analyses",{}).get("cv"):
        a = _cur()["analyses"]["cv"]
        st.info(f"Previous result from {a.get('timestamp','')} stored in this project.")
        _ai_interpret_btn("cv", interpret_cv,
                           {"results": {"ecsa":{},"mechanism":{},"k0":{}},
                            "electrode_name": st.session_state.get("cv_analyte", _cur().get("name",""))})


# ═══════════════════════════════════════════════════════════
# 🔵  EIS
# ═══════════════════════════════════════════════════════════
elif page == "🔵  EIS — Impedance":
    st.title("🔵 EIS — Impedance Spectroscopy")
    st.caption(f"Project: **{_cur().get('name','')}**")

    left,right = st.columns([1,1])
    with left:
        st.markdown('<div class="sec">Parameters</div>', unsafe_allow_html=True)
        eis_analyte = st.text_input("Electrode / analyte name",
                                     value=st.session_state.get("eis_analyte", _cur().get("name","Modified Electrode")),
                                     key="eis_analyte_inp",
                                     help="Name used in AI interpretation and reports.")
        st.session_state["eis_analyte"] = eis_analyte
        sign_z = st.selectbox("Z_imag sign",[-1,1],
                               format_func=lambda x:"−1 (most potentiostats)" if x==-1 else "+1",
                               help="Convention used by your instrument for the imaginary impedance. Most potentiostats (Autolab, BioLogic, Gamry) export Z'' as negative values in the inductive quadrant. Select +1 only if your software already inverts the sign.")
    with right:
        st.markdown('<div class="sec">Input mode</div>', unsafe_allow_html=True)
        eis_mode = _input_mode("eis_mode")

    tmpdir=tempfile.mkdtemp(); file_dict={}

    if eis_mode == "📁 Upload CSV":
        _csv_template_btn(
            "Freq_Hz,Zreal_Ohm,Zimag_Ohm\n"
            "100000,10.2,-0.8\n50000,10.5,-1.4\n10000,11.8,-3.2\n5000,14.1,-7.1\n"
            "1000,28.5,-22.4\n500,55.3,-38.7\n100,210.4,-85.3\n50,340.1,-72.1\n"
            "10,480.2,-38.4\n5,510.3,-22.1\n1,530.5,-10.2\n",
            "eis_template.csv",
            "⬇️ Download EIS Template (Zreal / Zimag columns)"
        )
        eis_ups = st.file_uploader("EIS files",type="csv",accept_multiple_files=True)
        if eis_ups:
            for _f_prev in eis_ups:
                _csv_preview(_f_prev, key_prefix=f"eis_{_f_prev.name}")
            st.markdown('<div class="sec">Electrode labels</div>',unsafe_allow_html=True)
            _eis_labels = []
            for i,f in enumerate(eis_ups):
                lbl = st.text_input(f"`{f.name}`",
                                     value=f.name.replace('.csv','').replace('_',' '),
                                     key=f"eis_l{i}")
                p = os.path.join(tmpdir,f"eis_{i}.csv")
                Path(p).write_bytes(f.getvalue()); file_dict[lbl]=p
                _eis_labels.append(lbl)
            # ── Renk seçiciler ───────────────────────────────────────────
            _eis_base = st.session_state.get("plot_style",{}).get(
                "colors", ['#1a3a5c','#c55a11','#375623','#7030a0',
                           '#843c0c','#1f77b4','#ff7f0e','#2ca02c'])
            st.markdown('<div class="sec">🎨 Dataset Colors</div>',
                        unsafe_allow_html=True)
            _eis_ncols      = min(len(_eis_labels), 4)
            _eis_color_cols = st.columns(_eis_ncols)
            _eis_colors     = []
            for _ci, _lbl in enumerate(_eis_labels):
                _def = _eis_base[_ci % len(_eis_base)]
                _pk  = _eis_color_cols[_ci % _eis_ncols].color_picker(
                    _lbl[:18], _def, key=f"eis_col_{_ci}")
                _eis_colors.append(_pk)
            st.session_state["eis_custom_colors"] = _eis_colors
    else:
        st.markdown('<div class="sec">Enter Randles-CPE parameters per electrode</div>',
                    unsafe_allow_html=True)
        st.info(
            "**Rs** = solution resistance (Ω) | "
            "**Rct** = charge-transfer resistance (Ω, lower = faster kinetics) | "
            "**Q** = CPE pre-factor (F·s^(n−1)) | "
            "**n_cpe** = CPE exponent (1 = ideal capacitor, <1 = rough/porous surface)"
        )
        default_eis = pd.DataFrame({
            'Label':  ['Bare','Modified 1','Modified 2','Final'],
            'Rs_ohm': [35.0,33.0,31.0,29.0],
            'Rct_ohm':[4000.0,2500.0,800.0,200.0],
            'Q':      [3.5e-7,5.0e-7,1.2e-6,3.0e-6],
            'n_cpe':  [0.88,0.86,0.84,0.82],
        })
        eis_tbl = st.data_editor(default_eis,num_rows="dynamic",
                                  use_container_width=True,key="eis_tbl")
        for _,row in eis_tbl.dropna().iterrows():
            lbl=str(row['Label'])
            df=_synth_eis(float(row['Rs_ohm']),float(row['Rct_ohm']),
                           float(row['Q']),float(row['n_cpe']))
            p=os.path.join(tmpdir,f"eis_{lbl}.csv")
            df.to_csv(p,index=False); file_dict[lbl]=p
        sign_z=1

    if file_dict:
        if "eis" in _cur().get("analyses", {}):
            st.warning("⚠️ An EIS analysis is already saved for this project. Running again will overwrite it.")
    if file_dict and st.button("▶ Run EIS Analysis",type="primary",use_container_width=True):
        with st.spinner("⚙️ Fitting Randles-CPE equivalent circuit — may take a moment…"):
            out=os.path.join(tmpdir,"eis_out"); os.makedirs(out,exist_ok=True)
            try:
                _eis_ps = dict(st.session_state.get("plot_style") or {})
                if st.session_state.get("eis_custom_colors"):
                    _eis_ps["colors"] = st.session_state["eis_custom_colors"]
                _eis_result=run_eis_analysis(file_dict,out,sign_zimag=sign_z,
                                             plot_style=_eis_ps)
                param_df = _eis_result[0] if isinstance(_eis_result, tuple) else _eis_result
                fit_results = _eis_result[1] if isinstance(_eis_result, tuple) and len(_eis_result)>1 else None
                st.success("✅ Done — results saved to project")
                if param_df is not None and not param_df.empty:
                    st.markdown('<div class="sec">Fit parameters</div>',
                                unsafe_allow_html=True)
                    st.dataframe(param_df,use_container_width=True,hide_index=True)
                # Extract key EIS metrics into the metrics dict so AI can use them
                _eis_metrics = {}
                if param_df is not None and not param_df.empty:
                    for col in param_df.columns:
                        try:
                            _eis_metrics[col] = param_df[col].tolist()
                        except Exception:
                            pass
                    _eis_metrics["electrodes"] = list(file_dict.keys())
                _eis_notes = _analysis_notes_widget("eis")
                _save_analysis("eis", _eis_metrics, out, param_df=param_df, notes=_eis_notes)
                _metrics_csv_btn({k:v for k,v in _eis_metrics.items() if k != "electrodes"}, "eis_metrics.csv")
                st.markdown('<div class="sec">Plots</div>',unsafe_allow_html=True)
                _show_imgs(out); _dl_btn(out,"eis_analysis.zip")
                _ai_interpret_btn("eis",interpret_eis,
                                   {"param_df":param_df,
                                    "electrode_labels":list(file_dict.keys())})
            except Exception as e:
                st.error(str(e))
                import traceback; st.code(traceback.format_exc())
    elif _cur().get("analyses",{}).get("eis"):
        a=_cur()["analyses"]["eis"]
        st.info(f"Previous EIS result from {a.get('timestamp','')} stored in this project.")
        _ai_interpret_btn("eis",interpret_eis,{"param_df":None,"electrode_labels":[]})


# ═══════════════════════════════════════════════════════════
# 🧪  pH
# ═══════════════════════════════════════════════════════════
elif page == "🧪  pH — Working pH":
    st.title("🧪 pH — Working pH Optimisation")
    st.caption(f"Project: **{_cur().get('name','')}**")

    left,right = st.columns([1,1])
    with left:
        st.markdown('<div class="sec">Parameters</div>',unsafe_allow_html=True)
        ph_n_el = st.number_input("n (electrons)",value=2,min_value=1,max_value=6,
                                   help="Number of electrons involved in the electrode reaction. Used to calculate the theoretical Nernst slope: −59.2 × m_H⁺ / n mV/pH at 25 °C. For most organic oxidations n=2.")
        pk_type = st.selectbox("Peak type",["anodic","cathodic"],
                                help="Direction of the peak you are tracking. Anodic (oxidation) peaks shift negatively with increasing pH; cathodic (reduction) peaks shift positively.")
        analyte_ph = st.text_input("Analyte (for report)",value="Analyte")
    with right:
        st.markdown('<div class="sec">Input mode</div>',unsafe_allow_html=True)
        ph_mode = _input_mode("ph_mode")

    tmpdir=tempfile.mkdtemp(); ph_path=None

    if ph_mode == "📁 Upload CSV":
        _csv_template_btn(
            "pH,Ep_mV\n"
            "4.0,320.5\n4.5,308.1\n5.0,295.4\n5.5,282.7\n6.0,270.3\n"
            "6.5,257.9\n7.0,245.2\n7.5,232.8\n8.0,220.1\n8.5,207.4\n9.0,195.0\n",
            "ph_template.csv",
            "⬇️ Download pH Template (pH vs Peak Potential)"
        )
        ph_up=st.file_uploader("pH table CSV",type="csv")
        if ph_up:
            _csv_preview(ph_up, key_prefix="ph")
            ph_path=os.path.join(tmpdir,"ph.csv")
            Path(ph_path).write_bytes(ph_up.getvalue())
    else:
        st.markdown('<div class="sec">Enter pH data</div>',unsafe_allow_html=True)
        default_ph = pd.DataFrame({'pH':[5.0,5.5,6.0,6.5,7.0,7.4,8.0,8.5,9.0],
                                    'Ep_V':[None]*9,'ip_A':[None]*9})
        ph_tbl=st.data_editor(default_ph,num_rows="dynamic",
                               use_container_width=True,key="ph_tbl")
        clean=ph_tbl.dropna(subset=['Ep_V'])
        if len(clean)>=3:
            ph_path=os.path.join(tmpdir,"ph.csv"); clean.to_csv(ph_path,index=False)

    # ── pH Renk seçiciler ─────────────────────────────────────────────────
    _ph_base = st.session_state.get("plot_style",{}).get(
        "colors", ['#1a3a5c','#c55a11'])
    st.markdown('<div class="sec">🎨 Plot Colors</div>', unsafe_allow_html=True)
    _phc1, _phc2 = st.columns(2)
    _ph_data_col = _phc1.color_picker(
        "Data points", _ph_base[0], key="ph_col_data")
    _ph_fit_col  = _phc2.color_picker(
        "Fit line",    _ph_base[1] if len(_ph_base)>1 else '#c55a11',
        key="ph_col_fit")
    st.session_state["ph_custom_colors"] = [_ph_data_col, _ph_fit_col]

    if ph_path:
        if "ph" in _cur().get("analyses", {}):
            st.warning("⚠️ A pH analysis is already saved for this project. Running again will overwrite it.")
    if ph_path and st.button("▶ Run pH Analysis",type="primary",use_container_width=True):
        with st.spinner("Analysing…"):
            out=os.path.join(tmpdir,"ph_out"); os.makedirs(out,exist_ok=True)
            try:
                _ph_ps = dict(st.session_state.get("plot_style") or {})
                _ph_ps["colors"] = st.session_state.get(
                    "ph_custom_colors", _ph_ps.get("colors", ['#1a3a5c','#c55a11']))
                res=run_ph_analysis(ph_path,int(ph_n_el),pk_type,out,
                                    plot_style=_ph_ps)
                st.success("✅ Done — results saved to project")
                theory=-59.2*res.get('m_proton',ph_n_el)/int(ph_n_el)
                metrics={
                    "Slope (mV/pH)": f"{res.get('slope_mV_per_pH',0):.2f}",
                    "Theory (mV/pH)":f"{theory:.1f}",
                    "R²":            f"{res.get('R2',0):.5f}",
                    "m/n ratio":     f"{res.get('ratio_m_n',0):.3f}",
                    "Optimal pH":    f"{res.get('optimal_pH',0):.1f}",
                }
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Slope (mV/pH)",metrics["Slope (mV/pH)"],
                           delta=f"Theory: {theory:.1f}")
                m2.metric("R²",metrics["R²"])
                m3.metric("m/n ratio",metrics["m/n ratio"])
                m4.metric("Optimal pH",metrics["Optimal pH"])

                # ── Mechanism verdict from m/n ratio ───────────
                _mn = res.get("ratio_m_n", 0)
                _slope_obs = res.get("slope_mV_per_pH", 0)
                _slope_pct_err = abs(_slope_obs - theory) / abs(theory) * 100 if theory else 0
                _mn_round = round(_mn * 2) / 2  # round to nearest 0.5
                if abs(_mn_round - 1.0) < 0.15:
                    st.success(f"✅ m/n ≈ 1 — consistent with an equal-proton–equal-electron transfer mechanism "
                               f"(e.g., 2H⁺/2e⁻ for n={int(ph_n_el)}). "
                               f"Slope deviation from Nernst theory: {_slope_pct_err:.1f}%.")
                elif abs(_mn_round - 0.5) < 0.15:
                    st.info(f"ℹ️ m/n ≈ 0.5 — suggests {int(ph_n_el)} electrons transferred with "
                            f"{int(ph_n_el)//2} protons (or vice versa). "
                            "Verify mechanism against CV redox data.")
                elif abs(_mn_round - 2.0) < 0.2:
                    st.info(f"ℹ️ m/n ≈ 2 — twice as many protons as electrons transferred. "
                            "Uncommon; verify experimental pH range excludes buffer interference.")
                else:
                    st.warning(f"⚠️ m/n = {_mn:.3f} — non-integer ratio. "
                               "This may indicate a complex multi-step mechanism, pH range outside "
                               "linearity, or buffer-electrode interaction. Review raw Ep vs pH plot.")
                _ph_notes = _analysis_notes_widget("ph")
                _save_analysis("ph",metrics,out, notes=_ph_notes)
                _metrics_csv_btn(metrics, "ph_metrics.csv")
                st.markdown('<div class="sec">Plots</div>',unsafe_allow_html=True)
                _show_imgs(out); _dl_btn(out,"ph_analysis.zip")
                _ai_interpret_btn("ph",interpret_ph,
                                   {"results":res,"analyte":analyte_ph})
            except Exception as e:
                st.error(str(e))
                import traceback; st.code(traceback.format_exc())
    elif _cur().get("analyses",{}).get("ph"):
        a=_cur()["analyses"]["ph"]
        st.info(f"Previous pH result from {a.get('timestamp','')} stored.")
        _ai_interpret_btn("ph",interpret_ph,{"results":{},"analyte":analyte_ph})


# ═══════════════════════════════════════════════════════════
# 📊  CALIBRATION
# ═══════════════════════════════════════════════════════════
elif page == "📊  Calibration & LOD/LOQ":
    st.title("📊 Calibration Curve, LOD & LOQ")
    st.caption(f"Project: **{_cur().get('name','')}**")

    # ── Toplu Validasyon Şablonu İndir & Yükle ─────────────────────────────
    with st.expander("📦 Bulk Validation Template (.xlsx) — all sections in one file",
                     expanded=False):
        col_dl, col_up2 = st.columns([1, 1])
        with col_dl:
            st.markdown("**1 · Download Template**")

            @st.cache_data(show_spinner=False)
            def _build_template_bytes():
                """Build the template in memory and return bytes (Streamlit Cloud compatible)."""
                import io, openpyxl
                from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
                from openpyxl.utils import get_column_letter

                BLUE_H = "1F4E79"; BLUE_L = "BDD7EE"
                GREEN  = "E2EFDA"; YELLOW = "FFEB9C"

                def _tb():
                    s = Side(style="thin", color="BFBFBF")
                    return Border(left=s, right=s, top=s, bottom=s)

                def _hdr(ws, row, vals):
                    for col, v in enumerate(vals, 1):
                        c = ws.cell(row=row, column=col, value=v)
                        c.font      = Font(bold=True, color="FFFFFF", size=10)
                        c.fill      = PatternFill("solid", fgColor=BLUE_H)
                        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                        c.border    = _tb()
                    ws.row_dimensions[row].height = 24

                def _dat(ws, row, vals, fill=GREEN):
                    for col, v in enumerate(vals, 1):
                        c = ws.cell(row=row, column=col, value=v)
                        c.font      = Font(size=10)
                        c.alignment = Alignment(horizontal="center", vertical="center")
                        c.fill      = PatternFill("solid", fgColor=fill)
                        c.border    = _tb()

                def _cw(ws, widths):
                    for i, w in enumerate(widths, 1):
                        ws.column_dimensions[get_column_letter(i)].width = w

                wb = openpyxl.Workbook()

                # README
                ws = wb.active; ws.title = "README"
                _cw(ws, [28, 72])
                readme = [
                    ("SilverstoneLab Echem Suite",
                     "ICH Q2(R2) Bulk Validation Template  –  v1.0"),
                    ("", ""),
                    ("HOW TO USE", ""),
                    ("1. Fill in", "Use the green rows as a guide; enter your own data in the yellow rows."),
                    ("2. Save", "Save as .xlsx; do not rename the sheet tabs."),
                    ("3. Upload", "Drag the file into the app's 'Bulk Validation → Upload filled template' section."),
                    ("4. Run", "Click 'Run All Validation' to compute everything at once."),
                    ("", ""),
                    ("SHEET DESCRIPTIONS", ""),
                    ("Calibration",     "Concentration (column A) + replicate ip measurements (B, C, … columns). Min. 5 levels."),
                    ("Repeatability",   "Level | Concentration_uM | ip_uA — ≥ 6 rows per level."),
                    ("Inter_Precision", "Day | Level | Concentration_uM | ip_uA — at least 3 different days."),
                    ("Accuracy",        "QC_Level | Nominal_uM | Replicate | ip_uA — LLOQ/LQC/MQC/HQC."),
                    ("Std_Addition",    "Added_uM | ip_uA — added standard concentration and measured current."),
                    ("", ""),
                    ("COLOUR KEY", ""),
                    ("🟢 Green", "Sample / reference data — you may delete these rows."),
                    ("🟡 Yellow","Enter your own data here."),
                    ("🔵 Blue",  "Column headers — do not change."),
                    ("", ""),
                    ("IMPORTANT", "All concentrations in µM, all currents in µA. Leave blank cells empty (do not write 0)."),
                ]
                for r, (k, v) in enumerate(readme, 1):
                    ck = ws.cell(row=r, column=1, value=k)
                    cv = ws.cell(row=r, column=2, value=v)
                    for c in (ck, cv):
                        c.alignment = Alignment(vertical="center", wrap_text=True)
                        ws.row_dimensions[r].height = 18
                    if k == "SilverstoneLab Echem Suite":
                        for c in (ck, cv):
                            c.font = Font(bold=True, color="FFFFFF", size=13)
                            c.fill = PatternFill("solid", fgColor=BLUE_H)
                    elif k in ("HOW TO USE", "SHEET DESCRIPTIONS", "COLOUR KEY", "IMPORTANT"):
                        for c in (ck, cv):
                            c.font = Font(bold=True, color="1F4E79", size=11)
                            c.fill = PatternFill("solid", fgColor=BLUE_L)
                    elif k.startswith(("1.", "2.", "3.", "4.")):
                        for c in (ck, cv):
                            c.fill = PatternFill("solid", fgColor=GREEN)

                # Calibration
                ws2 = wb.create_sheet("Calibration"); _cw(ws2, [22,18,18,18])
                _hdr(ws2, 1, ["Konsantrasyon_uM","ip_run1_uA","ip_run2_uA","ip_run3_uA"])
                ws2.freeze_panes = "A2"
                for r, row in enumerate([
                    [0.1,0.281,0.277,0.285],[0.5,1.112,1.105,1.119],
                    [1.0,2.157,2.149,2.165],[5.0,10.497,10.481,10.513],
                    [10.0,20.922,20.904,20.941],[20.0,41.763,41.741,41.785],
                    [40.0,83.448,83.421,83.476],[60.0,125.14,125.11,125.17],
                    [80.0,166.83,166.80,166.86],[100.0,208.52,208.49,208.56],
                ], 2):
                    _dat(ws2, r, row)
                for r in range(12, 16):
                    _dat(ws2, r, [""]*4, fill=YELLOW)

                # Repeatability
                ws3 = wb.create_sheet("Repeatability"); _cw(ws3,[18,22,16])
                _hdr(ws3, 1, ["Level","Concentration_uM","ip_uA"]); ws3.freeze_panes="A2"
                _rep_ex = (
                    [["Low",1.0,v] for v in [2.158,2.151,2.163,2.156,2.160,2.154]] +
                    [["Orta",20.0,v] for v in [41.764,41.748,41.779,41.753,41.771,41.758]] +
                    [["High",80.0,v] for v in [166.831,166.808,166.853,166.819,166.844,166.826]]
                )
                for r, row in enumerate(_rep_ex, 2):
                    _dat(ws3, r, row)
                for r in range(len(_rep_ex)+2, len(_rep_ex)+8):
                    _dat(ws3, r, [""]*3, fill=YELLOW)

                # Inter_Precision
                ws4 = wb.create_sheet("Inter_Precision"); _cw(ws4,[14,18,22,16])
                _hdr(ws4, 1, ["Day","Level","Concentration_uM","ip_uA"]); ws4.freeze_panes="A2"
                _ipr_ex = []
                for gn in range(1,4):
                    for sev, conc, base in [("Low",1.0,2.157),("Mid",20.0,41.763),("High",80.0,166.831)]:
                        for i in range(3):
                            _ipr_ex.append([f"Day {gn}", sev, conc, round(base + (i-1)*0.012,3)])
                for r, row in enumerate(_ipr_ex, 2):
                    _dat(ws4, r, row)
                for r in range(len(_ipr_ex)+2, len(_ipr_ex)+8):
                    _dat(ws4, r, [""]*4, fill=YELLOW)

                # Accuracy
                ws5 = wb.create_sheet("Accuracy"); _cw(ws5,[16,16,12,16])
                _hdr(ws5, 1, ["QC_Level","Nominal_uM","Replicate","ip_uA"]); ws5.freeze_panes="A2"
                _acc_ex = (
                    [["LLOQ",0.14,i+1,round(0.373+(i-1)*0.004,4)] for i in range(3)] +
                    [["LQC", 0.42,i+1,round(0.950+(i-1)*0.008,4)] for i in range(3)] +
                    [["MQC", 1.75,i+1,round(3.721+(i-1)*0.015,4)] for i in range(3)] +
                    [["HQC",80.0, i+1,round(166.83+(i-1)*0.120,4)] for i in range(3)]
                )
                for r, row in enumerate(_acc_ex, 2):
                    _dat(ws5, r, row)
                for r in range(len(_acc_ex)+2, len(_acc_ex)+6):
                    _dat(ws5, r, [""]*4, fill=YELLOW)

                # Std_Addition
                ws6 = wb.create_sheet("Std_Addition"); _cw(ws6,[18,16])
                _hdr(ws6, 1, ["Added_uM","ip_uA"]); ws6.freeze_panes="A2"
                for r, row in enumerate([
                    [0.0,22.845],[2.0,27.019],[5.0,33.280],[10.0,43.714],[20.0,64.583],[40.0,106.321]
                ], 2):
                    _dat(ws6, r, row)
                for r in range(8, 12):
                    _dat(ws6, r, [""]*2, fill=YELLOW)

                buf = io.BytesIO()
                wb.save(buf)
                buf.seek(0)
                return buf.getvalue()

            _tpl_bytes = _build_template_bytes()
            st.download_button(
                "⬇️ Download Template (.xlsx)",
                data=_tpl_bytes,
                file_name="SilverstoneLab_Validation_Template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption("Read the README sheet · Green = example · Yellow = your data · Do not rename sheets.")

        with col_up2:
            st.markdown("**2 · Upload Filled Template**")
            bulk_up = st.file_uploader("Upload filled template (.xlsx or .csv)",
                                       type=["xlsx", "csv"], key="bulk_xlsx_up")
            if bulk_up:
                try:
                    _is_csv = bulk_up.name.lower().endswith(".csv")
                    if _is_csv:
                        # CSV → sadece Calibration sayfası olarak işle
                        import io as _io
                        bulk_up.seek(0)
                        _df_csv = pd.read_csv(_io.BytesIO(bulk_up.read()), comment="#")
                        bulk_up.seek(0)
                        st.success(f"✓ CSV yüklendi — {len(_df_csv)} satır, sütunlar: {', '.join(_df_csv.columns.tolist())}")
                        # Calibration formatına dönüştür: ilk sütun konsantrasyon, kalanlar replikat
                        _conc_col = _df_csv.columns[0]
                        _rep_cols = _df_csv.columns[1:]
                        _bulk_cal_csv = {}
                        for _, _r in _df_csv.iterrows():
                            try:
                                _c = float(_r[_conc_col])
                            except (TypeError, ValueError):
                                continue
                            _vals = []
                            for _rc in _rep_cols:
                                try:
                                    if pd.notna(_r[_rc]):
                                        _vals.append(float(_r[_rc]))
                                except (TypeError, ValueError):
                                    pass
                            if _vals:
                                _bulk_cal_csv[_c] = _vals
                        if _bulk_cal_csv:
                            st.session_state["bulk_cal"] = _bulk_cal_csv
                            st.caption(f"  Calibration: {len(_bulk_cal_csv)} konsantrasyon seviyesi aktarıldı.")
                        else:
                            st.warning("CSV'den kalibrasyon verisi çıkarılamadı. İlk sütun konsantrasyon, diğerleri replikat olmalı.")
                    else:
                        import openpyxl as _opxl
                        _wb = _opxl.load_workbook(bulk_up, data_only=True)
                        _sheets = [s.title for s in _wb.worksheets]
                        st.success(f"✓ Loaded — sheets: {', '.join(_sheets)}")

                    # ── Xlsx-only parsing (CSV path zaten yukarıda bitti) ────
                    if not _is_csv:
                      def _xl_sheet_to_df(wb, name):
                        """Return a sheet as a list-of-lists (row 0 = headers)."""
                        if name not in [s.title for s in wb.worksheets]:
                            return None
                        ws = wb[name]
                        rows = list(ws.iter_rows(values_only=True))
                        return rows  # rows[0] = header

                      # ── Calibration ───────────────────────────────────────
                      _cal_rows = _xl_sheet_to_df(_wb, "Calibration")
                      if _cal_rows and len(_cal_rows) > 1:
                        _cal_hdr = _cal_rows[0]
                        _bulk_cal = {}
                        for _row in _cal_rows[1:]:
                            if _row[0] is None:
                                continue
                            try:
                                _conc = float(_row[0])
                            except (TypeError, ValueError):
                                continue
                            _vals = []
                            for _v in _row[1:]:
                                try:
                                    if _v is not None:
                                        _vals.append(float(_v))
                                except (TypeError, ValueError):
                                    pass
                            if _vals:
                                _bulk_cal[_conc] = _vals
                        if _bulk_cal:
                            st.session_state["bulk_cal"] = _bulk_cal
                            st.caption(f"  Calibration: {len(_bulk_cal)} concentration levels imported.")

                      # ── Repeatability ──────────────────────────────────────
                      _rep_rows = _xl_sheet_to_df(_wb, "Repeatability")
                      if _rep_rows and len(_rep_rows) > 1:
                        _bulk_rep = {}
                        for _row in _rep_rows[1:]:
                            if _row[0] is None:
                                continue
                            _lbl = str(_row[0]).strip()
                            try:
                                _ip = float(_row[2]) if len(_row) > 2 else float(_row[1])
                            except (TypeError, ValueError):
                                continue
                            _bulk_rep.setdefault(_lbl, []).append(_ip)
                        if _bulk_rep:
                            st.session_state["bulk_rep"] = _bulk_rep
                            st.caption(f"  Repeatability: {len(_bulk_rep)} levels imported.")

                      # ── Inter_Precision ────────────────────────────────────
                      _ipr_rows = _xl_sheet_to_df(_wb, "Inter_Precision")
                      if _ipr_rows and len(_ipr_rows) > 1:
                        _bulk_ipr = {}
                        for _row in _ipr_rows[1:]:
                            if _row[0] is None:
                                continue
                            _gun = str(_row[0]).strip()
                            _sev = str(_row[1]).strip() if len(_row) > 1 else "?"
                            try:
                                _ip  = float(_row[3]) if len(_row) > 3 else float(_row[2])
                            except (TypeError, ValueError):
                                continue
                            _bulk_ipr.setdefault(_gun, {}).setdefault(_sev, []).append(_ip)
                        if _bulk_ipr:
                            st.session_state["bulk_ipr"] = _bulk_ipr
                            st.caption(f"  Intermediate Precision: {len(_bulk_ipr)} days imported.")

                      # ── Accuracy ──────────────────────────────────────────
                      _acc_rows = _xl_sheet_to_df(_wb, "Accuracy")
                      if _acc_rows and len(_acc_rows) > 1:
                        _bulk_acc = {}
                        for _row in _acc_rows[1:]:
                            if _row[0] is None:
                                continue
                            _lbl = str(_row[0]).strip()
                            try:
                                _nom = float(_row[1])
                                _ip  = float(_row[3]) if len(_row) > 3 else float(_row[2])
                            except (TypeError, ValueError):
                                continue
                            if _lbl not in _bulk_acc:
                                _bulk_acc[_lbl] = (_nom, [])
                            _bulk_acc[_lbl][1].append(_ip)
                        if _bulk_acc:
                            st.session_state["bulk_acc"] = _bulk_acc
                            st.caption(f"  Accuracy: {len(_bulk_acc)} QC levels imported.")

                      # ── Std_Addition ───────────────────────────────────────
                      _sa_rows = _xl_sheet_to_df(_wb, "Std_Addition")
                      if _sa_rows and len(_sa_rows) > 1:
                        _bulk_sa = []
                        for _row in _sa_rows[1:]:
                            try:
                                if _row[0] is not None and _row[1] is not None:
                                    _bulk_sa.append((float(_row[0]), float(_row[1])))
                            except (TypeError, ValueError):
                                pass
                        if _bulk_sa:
                            st.session_state["bulk_sa"] = _bulk_sa
                            st.caption(f"  Standard Addition: {len(_bulk_sa)} points imported.")

                    # ── Her iki format için yükleme onayı ─────────────────────
                    if "bulk_cal" in st.session_state:
                        st.info("✅ Veri yüklendi. Aşağıdan analizi başlatın."
                                " ICH Q2 bölümleri açıldığında veriler otomatik uygulanır.")

                except Exception as _e:
                    st.error(f"Could not read template: {_e}")

    # ── Toplu Validasyon: Tek Tıkla Çalıştır ──────────────────────────────────
    _has_bulk = ("bulk_cal" in st.session_state and st.session_state.get("bulk_cal"))
    if _has_bulk:
        st.success("📦 Template loaded — all sections ready. Run everything below in one click.")
        _bulk_col1, _bulk_col2, _bulk_col3 = st.columns([2, 1, 1])
        with _bulk_col1:
            _bulk_analyte  = st.text_input("Analyte name (for report)",
                                            value=st.session_state.get("bulk_analyte","Analyte"),
                                            key="bulk_analyte_inp")
            st.session_state["bulk_analyte"] = _bulk_analyte
        with _bulk_col2:
            _BUNITS = ["μM","mM","nM","mg/mL","μg/mL","ng/mL","mg/L","μg/L","ng/L","mol/L","mmol/L"]
            _bulk_unit = st.selectbox("Birim", _BUNITS, key="bulk_unit_sel",
                                       index=_BUNITS.index(st.session_state.get("cal_unit","μM")))
            st.session_state["cal_unit"] = _bulk_unit
        with _bulk_col3:
            _bulk_area = st.number_input("Electrode area (cm²)",
                                          value=0.0, min_value=0.0, format="%.4f",
                                          key="bulk_area_inp")

        if st.button("▶ Run All Validation", type="primary",
                     use_container_width=True, key="btn_bulk_run"):
            import tempfile, os as _os
            _banalyte = st.session_state.get("bulk_analyte", "Analyte")
            _bunit    = st.session_state.get("cal_unit", "μM")
            _barea    = _bulk_area if _bulk_area > 0 else None
            _bout     = tempfile.mkdtemp(prefix="bulk_validation_")
            _bps      = st.session_state.get("plot_style", {}) or {}

            with st.spinner("⚙️ Computing calibration + LOD/LOQ…"):
                try:
                    _bulk_res = run_analytical_performance(
                        calibration_source=st.session_state["bulk_cal"],
                        analyte_name=_banalyte,
                        electrode_area_cm2=_barea,
                        unit=_bunit,
                        output_dir=_bout,
                        plot_style=_bps,
                    )
                    st.session_state["bulk_full_res"] = _bulk_res
                    st.session_state["bulk_out_dir"]  = _bout
                    _cr = _bulk_res.get("calibration", _bulk_res)
                    _save_analysis("cal", {
                        "LOD":    f"{_cr.get('LOD',0):.4g} {_bunit}",
                        "LOQ":    f"{_cr.get('LOQ',0):.4g} {_bunit}",
                        "r":      f"{_cr.get('r',_cr.get('R2',0)):.5f}",
                        "Slope":  f"{_cr.get('slope',0):.4f} µA/{_bunit}",
                        "Intercept": f"{_cr.get('intercept',0):.4f} µA",
                    }, _bout)
                    st.success("✅ Calibration complete.")
                except Exception as _e:
                    st.error(f"Calibration error: {_e}")
                    import traceback; st.code(traceback.format_exc())

            # ICH Q2 Precision ───────────────────────────────────────────────
            if "bulk_rep" in st.session_state and st.session_state["bulk_rep"]:
                with st.spinner("⚙️ Computing ICH Q2(R2) precision…"):
                    try:
                        _intra  = st.session_state["bulk_rep"]
                        _inter_flat = {}
                        if "bulk_ipr" in st.session_state:
                            for _gn, _sevs in st.session_state["bulk_ipr"].items():
                                for _sv, _vs in _sevs.items():
                                    _inter_flat[f"{_gn} – {_sv}"] = _vs
                        _prec_res = ich_q2_precision_validation(
                            intraday_data=_intra,
                            interday_data=_inter_flat or None,
                            analyte_name=_banalyte,
                            unit=_bunit,
                            output_dir=_bout,
                            plot_style=_bps,
                        )
                        st.session_state["bulk_prec_res"] = _prec_res
                        st.success("✅ Precision validation complete.")
                    except Exception as _e:
                        st.error(f"Precision error: {_e}")

            # Accuracy ────────────────────────────────────────────────────────
            if "bulk_acc" in st.session_state and "bulk_full_res" in st.session_state:
                with st.spinner("⚙️ Computing accuracy / recovery…"):
                    try:
                        _fcr = st.session_state["bulk_full_res"].get(
                            "calibration", st.session_state["bulk_full_res"])
                        _sl2  = _fcr.get("slope", 1)
                        _ic2  = _fcr.get("intercept", 0)
                        _acc_rows_b = []
                        for _qlbl, (_qnom, _qvals) in st.session_state["bulk_acc"].items():
                            if not _qvals:
                                continue
                            _ipm  = float(np.mean(_qvals))
                            _ipsd = float(np.std(_qvals, ddof=1)) if len(_qvals) > 1 else 0.0
                            _iprsd= (_ipsd / _ipm * 100) if _ipm != 0 else 0.0
                            _cc   = (_ipm - _ic2) / _sl2 if _sl2 != 0 else float('nan')
                            _rp   = (_cc / _qnom * 100) if _qnom != 0 else float('nan')
                            _acc_rows_b.append({
                                "QC Level": _qlbl,
                                f"Nominal C ({_bunit})": _qnom,
                                "n": len(_qvals),
                                "ip ort. (µA)": round(_ipm, 4),
                                "ip SD (µA)": round(_ipsd, 4),
                                "%RSD": round(_iprsd, 2),
                                f"Hesaplanan C ({_bunit})": round(_cc, 4),
                                "Recovery (%)": round(_rp, 2),
                                "ICH Kabul (98–102%)": "✅" if 98 <= _rp <= 102 else "⚠️",
                            })
                        st.session_state["bulk_acc_rows"] = _acc_rows_b
                        st.success("✅ Accuracy computed.")
                    except Exception as _e:
                        st.error(f"Accuracy error: {_e}")

            # Standard Addition ───────────────────────────────────────────────
            if "bulk_sa" in st.session_state and st.session_state["bulk_sa"]:
                with st.spinner("⚙️ Computing standard addition…"):
                    try:
                        _sa_adds = [x[0] for x in st.session_state["bulk_sa"]]
                        _sa_ips  = [x[1] for x in st.session_state["bulk_sa"]]
                        _sa_res  = standard_addition(
                            additions_uM=_sa_adds, ip_uA=_sa_ips,
                            analyte_name=_banalyte, unit=_bunit,
                            output_dir=_bout, plot_style=_bps,
                        )
                        st.session_state["bulk_sa_res"] = _sa_res
                        st.success("✅ Standard addition computed.")
                    except Exception as _e:
                        st.error(f"Standard addition error: {_e}")

            st.session_state["bulk_ran"] = True
            st.rerun()

        # ── Sonuçlar & Rapor (run sonrası görünür) ────────────────────────────
        if st.session_state.get("bulk_ran"):
            st.divider()
            st.markdown("## 📊 Validation Results")

            # Kalibrasyon özeti
            if "bulk_full_res" in st.session_state:
                _fcr2 = st.session_state["bulk_full_res"].get(
                    "calibration", st.session_state["bulk_full_res"])
                _lr   = _fcr2.get("linear_range", ["?","?"])
                st.markdown("### 📐 Calibration & LOD/LOQ")
                _mc1,_mc2,_mc3,_mc4,_mc5 = st.columns(5)
                _mc1.metric("Slope",   f"{_fcr2.get('slope',0):.4f} µA/{st.session_state.get('cal_unit','µM')}")
                _mc2.metric("r²",      f"{_fcr2.get('R2',_fcr2.get('r',0))**2:.5f}" if _fcr2.get('r') else f"{_fcr2.get('R2',0):.5f}")
                _mc3.metric("LOD",     f"{_fcr2.get('LOD',0):.4g} {st.session_state.get('cal_unit','µM')}")
                _mc4.metric("LOQ",     f"{_fcr2.get('LOQ',0):.4g} {st.session_state.get('cal_unit','µM')}")
                _mc5.metric("Lin. range", f"{_lr[0]:.3g}–{_lr[1]:.3g}")
                _bout2 = st.session_state.get("bulk_out_dir","")
                if _bout2:
                    _show_imgs(_bout2)

            # ICH Q2 Kesinlik özeti
            if "bulk_prec_res" in st.session_state:
                st.markdown("### 🔬 ICH Q2(R2) Precision")
                _pr = st.session_state["bulk_prec_res"]
                _intra_r = _pr.get("intraday", _pr.get("intraday_results", {}))
                _inter_r = _pr.get("interday", _pr.get("interday_results", {}))
                if _intra_r:
                    _intra_df = pd.DataFrame([
                        {"Level": k, "n": v.get("n", len(v.get("values",[]))),
                         "Mean (µA)": round(v.get("mean", v.get("mean_ip", 0)), 4),
                         "%RSD": round(v.get("rsd", 0), 2),
                         "Repeatability (≤2%)": "✅" if str(v.get("pass","False")) == "True" else "⚠️"}
                        for k,v in _intra_r.items() if isinstance(v, dict)
                    ])
                    st.caption("**Repeatability (Intraday)**")
                    st.dataframe(_intra_df, hide_index=True, use_container_width=True)
                if _inter_r:
                    _inter_df = pd.DataFrame([
                        {"Group": k, "n": v.get("n", len(v.get("values",[]))),
                         "Mean (µA)": round(v.get("mean", v.get("mean_ip", 0)), 4),
                         "%RSD": round(v.get("rsd", 0), 2),
                         "Intermediate Precision (≤5%)": "✅" if str(v.get("pass","False")) == "True" else "⚠️"}
                        for k,v in _inter_r.items() if isinstance(v, dict)
                    ])
                    st.caption("**Intermediate Precision (Interday)**")
                    st.dataframe(_inter_df, hide_index=True, use_container_width=True)

            # Accuracy özeti
            if "bulk_acc_rows" in st.session_state:
                st.markdown("### 🎯 Accuracy / Recovery")
                _acc_df2 = pd.DataFrame(st.session_state["bulk_acc_rows"])
                st.dataframe(_acc_df2, hide_index=True, use_container_width=True)
                _all_ok = all(98 <= r["Recovery (%)"] <= 102
                              for r in st.session_state["bulk_acc_rows"]
                              if not (isinstance(r["Recovery (%)"], float)
                                      and np.isnan(r["Recovery (%)"])))
                if _all_ok:
                    st.success("✅ All QC levels meet the ICH Q2(R2) accuracy criterion (98–102%)")
                else:
                    st.warning("⚠️ Some QC levels are outside the 98–102% range.")

            # Standart Ekleme özeti
            if "bulk_sa_res" in st.session_state:
                _sar = st.session_state["bulk_sa_res"]
                st.markdown("### ➕ Standard Addition")
                _sc1,_sc2,_sc3 = st.columns(3)
                _sc1.metric("Slope", f"{_sar.get('slope',0):.4f}")
                _sc2.metric("R²",    f"{_sar.get('R2',0):.5f}")
                _sc3.metric("Sample concentration",
                            f"{_sar.get('sample_concentration',0):.4g} "
                            f"{st.session_state.get('cal_unit','µM')}")

            # ── Rapor Üret ────────────────────────────────────────────────────
            st.divider()
            st.markdown("## 📄 Validation Report")
            st.caption("ICH Q2(R2)-compliant report — ready for journal submission (PDF + Word)")

            _rep_col1, _rep_col2 = st.columns(2)
            with _rep_col1:
                _rep_journal = st.selectbox(
                    "Target journal style",
                    ["Genel / ICH Q2(R2)", "Analytical Chemistry (ACS)",
                     "Biosensors & Bioelectronics (Elsevier)",
                     "ACS Sensors", "Talanta (Elsevier)",
                     "Journal of Electroanalytical Chemistry"],
                    key="bulk_journal_sel"
                )
            with _rep_col2:
                _rep_lang = st.selectbox("Report language", ["English", "Türkçe"],
                                          key="bulk_rep_lang")

            if st.button("📄 Generate PDF + Word Report", type="secondary",
                         use_container_width=True, key="btn_bulk_report"):
                with st.spinner("Generating report…"):
                    try:
                        # Mevcut projeye tüm sonuçları kaydet
                        _proj = _cur()
                        _rout = st.session_state.get("bulk_out_dir",
                                tempfile.mkdtemp(prefix="bulk_rep_"))

                        # Precision ve accuracy metriklerini de kaydet
                        if "bulk_prec_res" in st.session_state:
                            _pr2 = st.session_state["bulk_prec_res"]
                            _save_analysis("ich_precision", {
                                "intraday": _pr2.get("intraday", _pr2.get("intraday_results",{})),
                                "interday": _pr2.get("interday", _pr2.get("interday_results",{})),
                                "pass_fail": _pr2.get("pass_fail",{}),
                            }, _rout)
                        if "bulk_acc_rows" in st.session_state:
                            _save_analysis("cal_accuracy", {
                                r["QC Level"]: r["Recovery (%)"]
                                for r in st.session_state["bulk_acc_rows"]
                            }, _rout)

                        # Rapor oluştur
                        _rpaths = generate_reports(_proj, _rout)
                        st.success("✅ Report generated!")

                        _rdl1, _rdl2 = st.columns(2)
                        if "pdf" in _rpaths and os.path.exists(_rpaths["pdf"]):
                            with open(_rpaths["pdf"],"rb") as _pf:
                                _rdl1.download_button(
                                    "⬇️ Download PDF", _pf.read(),
                                    file_name=f"{st.session_state.get('bulk_analyte','Analyte')}_ICH_Q2R2_Validation.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                )
                        elif "pdf_error" in _rpaths:
                            _rdl1.error(f"PDF: {_rpaths['pdf_error']}")

                        if "docx" in _rpaths and os.path.exists(_rpaths["docx"]):
                            with open(_rpaths["docx"],"rb") as _wf:
                                _rdl2.download_button(
                                    "⬇️ Download Word (.docx)", _wf.read(),
                                    file_name=f"{st.session_state.get('bulk_analyte','Analyte')}_ICH_Q2R2_Validation.docx",
                                    mime="application/vnd.openxmlformats-officedocument"
                                          ".wordprocessingml.document",
                                    use_container_width=True,
                                )
                        elif "docx_error" in _rpaths:
                            _rdl2.error(f"DOCX: {_rpaths['docx_error']}")

                    except Exception as _re:
                        st.error(f"Report error: {_re}")
                        import traceback; st.code(traceback.format_exc())

            # ── AI Destekli Yorum ─────────────────────────────────────────────
            st.markdown("### 🤖 AI-Assisted Interpretation (High-Impact Journal Format)")
            st.caption(
                "The AI analyses all validation results and generates publication-ready English text "
                "suitable for the Methods & Results sections of a high-impact journal."
            )
            if "bulk_full_res" in st.session_state:
                _ai_kwargs_bulk = {
                    "results":  st.session_state["bulk_full_res"],
                    "analyte":  st.session_state.get("bulk_analyte","Analyte"),
                    "unit":     st.session_state.get("cal_unit","μM"),
                    "precision_results": st.session_state.get("bulk_prec_res"),
                    "accuracy_rows":     st.session_state.get("bulk_acc_rows"),
                    "sa_results":        st.session_state.get("bulk_sa_res"),
                    "journal":           st.session_state.get("bulk_journal_sel",""),
                    "language":          st.session_state.get("bulk_rep_lang","English"),
                }
                _ai_interpret_btn("cal", interpret_calibration, _ai_kwargs_bulk,
                                  result_key="bulk_ai_text",
                                  btn_key="ai_cal_bulk")

    left,right = st.columns([1,1])
    with left:
        st.markdown('<div class="sec">Parameters</div>',unsafe_allow_html=True)
        analyte   = st.text_input("Analyte name",value="Analyte")
        _UNITS    = ["μM","mM","nM","mg/mL","μg/mL","ng/mL","mg/L","μg/L","ng/L","mol/L","mmol/L"]
        _unit_sel = st.selectbox("Concentration unit", _UNITS,
                                  index=_UNITS.index(st.session_state.get("cal_unit","μM")),
                                  key="cal_unit_sel")
        st.session_state["cal_unit"] = _unit_sel
        conc_unit = _unit_sel
        area_cal  = st.number_input("Electrode area (cm²) — optional",
                                     value=0.0,min_value=0.0,format="%.4f",
                                     help="Geometric electrode area used to normalise current density. Leave at 0 to skip normalisation. For GCE ⌀3 mm → 0.0707 cm².")
    with right:
        st.markdown('<div class="sec">Input mode</div>',unsafe_allow_html=True)
        cal_mode = _input_mode("cal_mode")

    cal_data={}

    if cal_mode == "📁 Upload CSV":
        _csv_template_btn(
            "concentration_uM,ip_run1_uA,ip_run2_uA,ip_run3_uA\n"
            "0.5,0.312,0.318,0.315\n1.0,0.628,0.635,0.631\n2.0,1.254,1.261,1.258\n"
            "5.0,3.142,3.148,3.145\n10.0,6.285,6.291,6.288\n20.0,12.57,12.58,12.57\n"
            "50.0,31.42,31.45,31.44\n100.0,62.85,62.88,62.87\n",
            "calibration_template.csv",
            "⬇️ Download Calibration Template (concentration + 3 replicates)"
        )
        cal_up=st.file_uploader("Calibration CSV",type="csv")
        if cal_up:
            _csv_preview(cal_up, key_prefix="cal")
            df=pd.read_csv(cal_up); cc=df.columns[0]
            for _,row in df.iterrows():
                try:
                    conc_val = float(row[cc])
                except (ValueError, TypeError):
                    continue   # header / metadata row — skip
                vals=[]
                for c in df.columns[1:]:
                    try:
                        if pd.notna(row[c]):
                            vals.append(float(row[c]))
                    except (ValueError, TypeError):
                        pass   # non-numeric cell — skip
                if vals: cal_data[conc_val]=vals
    else:
        st.markdown('<div class="sec">Enter calibration data</div>',unsafe_allow_html=True)
        default_cal=pd.DataFrame({
            f'Concentration ({conc_unit})':[None]*8,
            'ip_1 (µA)':[None]*8,'ip_2 (µA)':[None]*8,'ip_3 (µA)':[None]*8,
        })
        cal_tbl=st.data_editor(default_cal,num_rows="dynamic",
                                use_container_width=True,key="cal_tbl")
        cc=cal_tbl.columns[0]
        for _,row in cal_tbl.dropna(subset=[cc,'ip_1 (µA)']).iterrows():
            vals=[float(row[c]) for c in cal_tbl.columns[1:] if pd.notna(row[c])]
            if vals: cal_data[float(row[cc])]=vals

    # Bulk şablondan kalibrasyon verisi varsa mevcut cal_data'ya ekle
    if "bulk_cal" in st.session_state:
        for _bc, _bv in st.session_state["bulk_cal"].items():
            if _bc not in cal_data:
                cal_data[_bc] = _bv

    # ── Kalibrasyon Renk Seçiciler ────────────────────────────
    _cal_base = st.session_state.get("plot_style",{}).get(
        "colors", ['#1a3a5c','#c55a11'])
    st.markdown('<div class="sec">🎨 Plot Colors</div>', unsafe_allow_html=True)
    _calc1, _calc2 = st.columns(2)
    _cal_data_col = _calc1.color_picker(
        "Data points & error bars", _cal_base[0], key="cal_col_data")
    _cal_fit_col  = _calc2.color_picker(
        "Fit line",  _cal_base[1] if len(_cal_base)>1 else '#c55a11',
        key="cal_col_fit")
    st.session_state["cal_custom_colors"] = [_cal_data_col, _cal_fit_col]

    # ── ICH Q2(R2) Precision Validation ──────────────────────
    st.divider()
    with st.expander("🔬 ICH Q2(R2) Precision Validation (Optional)", expanded=False):
        st.markdown("""
**ICH Q2(R2) Acceptance Criteria:**
| Precision Type | Minimum Requirement | %RSD Limit |
|---|---|---|
| **Repeatability** (Intraday) | ≥3 replicates, ≥3 concentrations | **≤ 2%** |
| **Intermediate Precision** (Interday) | ≥3 days, ≥3 replicates per day | **≤ 5%** |
        """)

        ich_tab1, ich_tab2 = st.tabs(["📅 Intraday (Repeatability)", "📆 Interday (Intermediate Precision)"])

        # --- Gün İçi -------------------------------------------------
        with ich_tab1:
            st.info("Replicate measurements at different concentration levels on the same day. "
                    "ICH Q2(R2): at least 3 levels (low/mid/high), ≥3 replicates per level.")

            ich_intra_mode = st.radio("Data entry", ["📋 Table (recommended)", "📁 Upload CSV"],
                                       horizontal=True, key="ich_intra_mode")

            intra_rows = []  # (etiket, list[float])

            if ich_intra_mode == "📁 Upload CSV":
                st.caption("Expected columns: **Level**, **ip(uA)** — one row per replicate.")
                ich_intra_up = st.file_uploader("Repeatability CSV", type="csv", key="ich_intra_up")
                if ich_intra_up:
                    try:
                        df_intra = pd.read_csv(ich_intra_up)
                        df_intra.columns = [c.strip() for c in df_intra.columns]
                        lbl_col = next((c for c in df_intra.columns if any(k in c.lower()
                                        for k in ('seviye','level','label','grup'))), df_intra.columns[0])
                        ip_col  = next((c for c in df_intra.columns if any(k in c.lower()
                                        for k in ('ip','current','i(','akım'))), None)  # 'akım'=current in Turkish (kept for file compatibility)
                        if ip_col is None:
                            st.error("No 'ip' column found in CSV.")
                        else:
                            for lbl, grp in df_intra.groupby(lbl_col, sort=False):
                                vals = grp[ip_col].dropna().astype(float).tolist()
                                if vals:
                                    intra_rows.append((str(lbl), vals))
                            st.success(f"✓ {len(intra_rows)} levels loaded")
                            st.dataframe(df_intra, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Could not read CSV: {e}")
            else:
                # Tablo modu — her satır bir konsantrasyon seviyesi, sütunlar tekrarlar
                st.caption("Each row = one concentration level. Empty cells are skipped automatically.")
                default_intra = pd.DataFrame({
                    'Level (Label)': ['Low', 'Mid', 'High'],
                    'ip_1 (µA)': [None, None, None],
                    'ip_2 (µA)': [None, None, None],
                    'ip_3 (µA)': [None, None, None],
                    'ip_4 (µA)': [None, None, None],
                    'ip_5 (µA)': [None, None, None],
                    'ip_6 (µA)': [None, None, None],
                })
                intra_tbl = st.data_editor(default_intra, num_rows="dynamic",
                                            use_container_width=True, key="ich_intra_tbl")
                ip_cols_intra = [c for c in intra_tbl.columns if c.startswith('ip_')]
                for _, row in intra_tbl.iterrows():
                    lbl = str(row['Level (Label)']).strip() if pd.notna(row['Level (Label)']) else ''
                    if not lbl or lbl == 'None':
                        continue
                    vals = []
                    for c in ip_cols_intra:
                        try:
                            if pd.notna(row[c]):
                                vals.append(float(row[c]))
                        except (ValueError, TypeError):
                            pass
                    if len(vals) >= 2:
                        intra_rows.append((lbl, vals))

        # --- Günler Arası --------------------------------------------
        with ich_tab2:
            st.info("Measurements on different days at the same concentration level. "
                    "At least 3 replicates per day are recommended.")

            ich_inter_mode = st.radio("Data entry", ["📋 Table (recommended)", "📁 Upload CSV"],
                                       horizontal=True, key="ich_inter_mode")

            inter_rows = []  # (etiket, list[float])

            if ich_inter_mode == "📁 Upload CSV":
                st.caption("Expected columns: **Day**, **ip(uA)** — one row per replicate.")
                ich_inter_up = st.file_uploader("Intermediate Precision CSV", type="csv", key="ich_inter_up")
                if ich_inter_up:
                    try:
                        df_inter = pd.read_csv(ich_inter_up)
                        df_inter.columns = [c.strip() for c in df_inter.columns]
                        day_col = next((c for c in df_inter.columns if any(k in c.lower()
                                        for k in ('gun','day','gün','date'))), df_inter.columns[0])
                        ip_col  = next((c for c in df_inter.columns if any(k in c.lower()
                                        for k in ('ip','current','i(','akım'))), None)  # 'akım'=current in Turkish (kept for file compatibility)
                        if ip_col is None:
                            st.error("No 'ip' column found in CSV.")
                        else:
                            for day, grp in df_inter.groupby(day_col, sort=False):
                                vals = grp[ip_col].dropna().astype(float).tolist()
                                if vals:
                                    inter_rows.append((str(day), vals))
                            st.success(f"✓ {len(inter_rows)} day groups loaded")
                            st.dataframe(df_inter, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Could not read CSV: {e}")
            else:
                # Tablo modu — her satır bir gün, sütunlar tekrarlar
                st.caption("Each row = one day. Empty cells are skipped automatically.")
                default_inter = pd.DataFrame({
                    'Day (Label)': ['Day 1', 'Day 2', 'Day 3'],
                    'ip_1 (µA)': [None, None, None],
                    'ip_2 (µA)': [None, None, None],
                    'ip_3 (µA)': [None, None, None],
                    'ip_4 (µA)': [None, None, None],
                    'ip_5 (µA)': [None, None, None],
                    'ip_6 (µA)': [None, None, None],
                })
                inter_tbl = st.data_editor(default_inter, num_rows="dynamic",
                                            use_container_width=True, key="ich_inter_tbl")
                ip_cols_inter = [c for c in inter_tbl.columns if c.startswith('ip_')]
                for _, row in inter_tbl.iterrows():
                    lbl = str(row['Day (Label)']).strip() if pd.notna(row['Day (Label)']) else ''
                    if not lbl or lbl == 'None':
                        continue
                    vals = []
                    for c in ip_cols_inter:
                        try:
                            if pd.notna(row[c]):
                                vals.append(float(row[c]))
                        except (ValueError, TypeError):
                            pass
                    if len(vals) >= 2:
                        inter_rows.append((lbl, vals))

        # Bulk şablondan gelen kesinlik verilerini otomatik ekle (henüz manuel giriş yoksa)
        if not intra_rows and "bulk_rep" in st.session_state:
            intra_rows = [(lbl, vals) for lbl, vals in st.session_state["bulk_rep"].items()]
            st.info(f"📦 Repeatability data loaded from bulk template: {[r[0] for r in intra_rows]}")
        if not inter_rows and "bulk_ipr" in st.session_state:
            # {gun: {seviye: [vals]}} → flatten to [(gun_sev_lbl, vals)]
            for _gun, _sevs in st.session_state["bulk_ipr"].items():
                for _sev, _vals in _sevs.items():
                    inter_rows.append((f"{_gun} – {_sev}", _vals))
            if inter_rows:
                st.info(f"📦 Intermediate precision data loaded from bulk template: {len(inter_rows)} groups")

        if st.button("▶ Run ICH Q2(R2) Validation", type="secondary", use_container_width=True,
                     key="btn_ich_q2"):
            # intra_rows ve inter_rows her modda zaten (etiket, list[float]) formatında
            intraday_data_ich = {lbl: vs for lbl, vs in intra_rows if len(vs) >= 2}

            # Parse inter-day
            interday_data_ich = {lbl: vs for lbl, vs in inter_rows if len(vs) >= 2}

            if not intraday_data_ich:
                st.error("Enter at least 1 intraday data group (label + comma-separated values).")
            else:
                with st.spinner("Computing ICH Q2(R2)…"):
                    ich_out = os.path.join(tempfile.mkdtemp(), "ich_out")
                    os.makedirs(ich_out, exist_ok=True)
                    try:
                        ich_res = ich_q2_precision_validation(
                            intraday_data=intraday_data_ich,
                            interday_data=interday_data_ich if interday_data_ich else None,
                            analyte_name=analyte,
                            unit=conc_unit,
                            output_dir=ich_out,
                            plot_style=st.session_state.get("plot_style")
                        )

                        # ── Metrikler ──
                        intra_ok = ich_res['overall_intra_pass']
                        inter_ok = ich_res['overall_inter_pass']
                        intra_rsd_max = ich_res['intra_max_rsd']
                        inter_rsd_tot = ich_res['inter_total_rsd']

                        c1, c2 = st.columns(2)
                        intra_icon = "✅" if intra_ok else "❌"
                        c1.metric(f"{intra_icon} Intraday Max %RSD",
                                  f"{intra_rsd_max:.2f}%",
                                  delta=f"Limit: 2.0% — {'Pass' if intra_ok else 'FAIL'}",
                                  delta_color="normal" if intra_ok else "inverse")
                        if inter_rsd_tot is not None:
                            inter_icon = "✅" if inter_ok else "❌"
                            c2.metric(f"{inter_icon} Interday Total %RSD",
                                      f"{inter_rsd_tot:.2f}%",
                                      delta=f"Limit: 5.0% — {'Pass' if inter_ok else 'FAIL'}",
                                      delta_color="normal" if inter_ok else "inverse")

                        # ── Detay tablosu ──
                        st.markdown("**Intraday Results:**")
                        rows_disp = []
                        for lbl, r in ich_res['intraday'].items():
                            rows_disp.append({
                                'Grup': lbl, 'n': r['n'],
                                'Mean (μA)': f"{r['mean']:.4f}",
                                'SD (μA)': f"{r['std']:.4f}",
                                '%RSD': f"{r['rsd']:.2f}%",
                                'ICH Limit': '≤2%',
                                'Status': '✅ Pass' if r['pass'] else '❌ Fail'
                            })
                        st.dataframe(pd.DataFrame(rows_disp),
                                     use_container_width=True, hide_index=True)

                        if ich_res['interday']:
                            st.markdown("**Interday Results:**")
                            rows_inter_disp = []
                            for day, r in ich_res['interday'].items():
                                if day.startswith('_'):
                                    continue
                                rows_inter_disp.append({
                                    'Day': day, 'n': r['n'],
                                    'Mean (μA)': f"{r['mean']:.4f}",
                                    'SD (μA)': f"{r['std']:.4f}",
                                    '%RSD (intraday)': f"{r['rsd']:.2f}%",
                                })
                            st.dataframe(pd.DataFrame(rows_inter_disp),
                                         use_container_width=True, hide_index=True)
                            if inter_rsd_tot is not None:
                                inter_icon = "✅" if inter_ok else "❌"
                                st.markdown(f"**Interday total %RSD:** {inter_rsd_tot:.2f}%  "
                                            f"(ICH limit ≤5%)  {inter_icon}")

                        # ── Grafik ──
                        _show_imgs(ich_out)
                        _dl_btn(ich_out, "ich_q2_validation.zip")

                    except Exception as e:
                        st.error(f"ICH Q2 computation error: {e}")
                        import traceback; st.code(traceback.format_exc())

    # ── QC Noktaları + Accuracy ──────────────────────────────
    st.divider()
    with st.expander("🎯 QC Points & Accuracy / Recovery (Optional)", expanded=False):

        # ── QC Önerisi ──────────────────────────────────────────
        st.markdown("### 💡 QC Point Suggestion")
        st.caption("Automatically computed from your calibration data. "
                   "Enter calibration data first, then click the button.")

        if st.button("🔢 Compute QC Points (LLOQ · LQC · MQC · HQC)",
                     key="btn_qc_suggest", type="secondary"):
            if not cal_data:
                st.warning("Enter calibration data first.")
            else:
                from scipy import stats as _stats
                _concs = sorted(cal_data.keys())
                _ips   = [float(np.mean(cal_data[c])) for c in _concs]
                _c_arr = np.array(_concs, dtype=float)
                _i_arr = np.array(_ips,   dtype=float)
                _slope, _intercept, _r, _, _se = _stats.linregress(_c_arr, _i_arr)
                _ip_pred   = _slope * _c_arr + _intercept
                _residuals = _i_arr - _ip_pred
                _syx = float(np.sqrt(np.sum(_residuals**2) / max(len(_c_arr)-2, 1)))
                _loq = 10 * _syx / _slope if _slope > 0 else _concs[0]
                _lod = 3  * _syx / _slope if _slope > 0 else _concs[0]
                _cmin, _cmax = float(_c_arr.min()), float(_c_arr.max())

                # ICH / FDA QC seviyeleri
                lloq = max(_loq, _cmin)
                lqc  = lloq * 3.0
                hqc  = _cmax * 0.80
                mqc  = (lloq * lqc * hqc) ** (1/3)   # geometrik orta

                # Kalibrasyon doğrusundan beklenen ip değerleri
                def _ip(c): return round(_slope * c + _intercept, 4)

                qc_df = pd.DataFrame([
                    {"QC Level": "LLOQ", "Nominal C": f"{lloq:.4g} {conc_unit}",
                     "Expected ip (µA)": _ip(lloq), "Description": "Lower limit of quantification (LOQ)"},
                    {"QC Level": "LQC",  "Nominal C": f"{lqc:.4g} {conc_unit}",
                     "Expected ip (µA)": _ip(lqc),  "Description": "Low QC ≈ 3× LLOQ"},
                    {"QC Level": "MQC",  "Nominal C": f"{mqc:.4g} {conc_unit}",
                     "Expected ip (µA)": _ip(mqc),  "Description": "Mid QC (geometric mean)"},
                    {"QC Level": "HQC",  "Nominal C": f"{hqc:.4g} {conc_unit}",
                     "Expected ip (µA)": _ip(hqc),  "Description": "High QC ≈ 80% of upper range"},
                ])
                st.dataframe(qc_df, use_container_width=True, hide_index=True)
                st.caption(f"Calibration: slope={_slope:.4f} µA/{conc_unit} | "
                           f"LOD={_lod:.4g} | LOQ={_loq:.4g} {conc_unit} | "
                           f"r={_r:.5f} | Range: {_cmin:.4g}–{_cmax:.4g} {conc_unit}")

                # ── Plotly interactive calibration curve ────────────
                try:
                    import plotly.graph_objects as _pgo
                    _fit_x = np.linspace(_cmin, _cmax, 200)
                    _fit_y = _slope * _fit_x + _intercept
                    _fig_pl = _pgo.Figure()
                    # Data points
                    _fig_pl.add_trace(_pgo.Scatter(
                        x=_c_arr.tolist(), y=_i_arr.tolist(),
                        mode="markers",
                        marker=dict(size=10, color="#1f77b4", symbol="circle"),
                        name="Data",
                        hovertemplate=(f"C = %{{x:.4g}} {conc_unit}<br>"
                                       "ip = %{y:.4f} µA<extra></extra>"),
                    ))
                    # Regression line
                    _fig_pl.add_trace(_pgo.Scatter(
                        x=_fit_x.tolist(), y=_fit_y.tolist(),
                        mode="lines",
                        line=dict(color="#e74c3c", width=2, dash="solid"),
                        name=f"Fit  r={_r:.4f}",
                        hovertemplate=(f"C = %{{x:.4g}} {conc_unit}<br>"
                                       "ip(fit) = %{y:.4f} µA<extra></extra>"),
                    ))
                    # LOD / LOQ vertical lines
                    for _lv, _lname, _lc in [(_lod,"LOD","#f39c12"), (_loq,"LOQ","#27ae60")]:
                        if _cmin <= _lv <= _cmax:
                            _fig_pl.add_vline(x=_lv, line_dash="dot",
                                              line_color=_lc, opacity=0.7,
                                              annotation_text=f"{_lname}={_lv:.3g}",
                                              annotation_position="top right")
                    _fig_pl.update_layout(
                        title=dict(text=f"{analyte} — Calibration Curve (interactive)",
                                   font=dict(size=14)),
                        xaxis_title=f"Concentration ({conc_unit})",
                        yaxis_title="Peak current ip (µA)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        height=400,
                        margin=dict(l=50, r=20, t=60, b=50),
                        hovermode="x unified",
                    )
                    st.plotly_chart(_fig_pl, use_container_width=True)
                except ImportError:
                    st.caption("💡 Install plotly (`pip install plotly`) for an interactive chart.")
                except Exception as _pe:
                    st.caption(f"Plotly chart skipped: {_pe}")

                # Session state'e kaydet — Accuracy tablosunda kullanmak için
                st.session_state["qc_points"] = {
                    "slope": _slope, "intercept": _intercept,
                    "lloq": lloq, "lqc": lqc, "mqc": mqc, "hqc": hqc,
                    "unit": conc_unit
                }

        st.divider()

        # ── Accuracy / Geri Kazanım Tablosu ─────────────────────
        st.markdown("### 📐 Accuracy (Recovery)")
        st.caption("Each row = one QC level. Each column = one replicate measurement (ip, µA). "
                   "Recovery is computed from the nominal concentration and measured ip.")

        acc_mode = st.radio("Data entry", ["📋 Table (recommended)", "📁 Upload CSV"],
                             horizontal=True, key="acc_mode")

        # QC noktaları öneriden geldiyse tablo etiketlerini doldur
        _qc = st.session_state.get("qc_points", {})
        _lloq_v = round(_qc.get("lloq", 0), 4) if _qc else 0
        _lqc_v  = round(_qc.get("lqc",  0), 4) if _qc else 0
        _mqc_v  = round(_qc.get("mqc",  0), 4) if _qc else 0
        _hqc_v  = round(_qc.get("hqc",  0), 4) if _qc else 0

        # acc_tbl_data: list of (qc_label, nominal_c, [ip values])
        acc_tbl_data = []

        # Bulk şablondan gelen accuracy verisi varsa önceden doldur
        if "bulk_acc" in st.session_state and not acc_tbl_data:
            for _lbl, (_nom, _vals) in st.session_state["bulk_acc"].items():
                acc_tbl_data.append((_lbl, _nom, _vals))
            st.info(f"📦 Accuracy data loaded from bulk template: {[d[0] for d in acc_tbl_data]}")

        if acc_mode == "📁 Upload CSV":
            st.caption(
                "Expected columns: **QC Level** (LLOQ/LQC/MQC/HQC), "
                f"**Nominal C ({conc_unit})**, **ip(uA)** — one row per replicate. "
                "The generated `4_accuracy_recovery.csv` file can be uploaded directly."
            )
            acc_up = st.file_uploader("Accuracy / Recovery CSV", type="csv", key="acc_up")
            if acc_up:
                try:
                    df_acc_up = pd.read_csv(acc_up)
                    df_acc_up.columns = [c.strip() for c in df_acc_up.columns]
                    # QC seviye sütunu
                    qc_col  = next((c for c in df_acc_up.columns if any(k in c.lower()
                                    for k in ('qc','seviye','level','label'))), df_acc_up.columns[0])
                    # Nominal konsantrasyon sütunu
                    nom_col_up = next((c for c in df_acc_up.columns if any(k in c.lower()
                                       for k in ('nominal','konsantrasyon','conc','c('))),
                                      df_acc_up.columns[1] if len(df_acc_up.columns) > 1 else None)
                    # ip sütunu
                    ip_col_up  = next((c for c in df_acc_up.columns if any(k in c.lower()
                                       for k in ('ip_olc','ip(','ip_m','current','akım',
                                                  'ip_run','ip_1','i(a'))), None)
                    # Geniş format: ip_run1, ip_run2 … sütunları varsa
                    ip_run_cols = [c for c in df_acc_up.columns
                                   if any(k in c.lower() for k in ('ip_run','run','tekrar'))
                                   and c != qc_col and c != nom_col_up]

                    if nom_col_up is None:
                        st.error("No nominal concentration column found.")
                    elif ip_col_up is None and not ip_run_cols:
                        st.error("No ip measurement column found.")
                    else:
                        st.dataframe(df_acc_up, use_container_width=True, hide_index=True)
                        if ip_run_cols:
                            # Geniş format (her sütun bir tekrar)
                            for _, row in df_acc_up.iterrows():
                                lbl = str(row[qc_col]).strip()
                                try:
                                    nom = float(row[nom_col_up])
                                except (ValueError, TypeError):
                                    continue
                                vals = []
                                for rc in ip_run_cols:
                                    try:
                                        if pd.notna(row[rc]):
                                            vals.append(float(row[rc]))
                                    except (ValueError, TypeError):
                                        pass
                                if vals:
                                    acc_tbl_data.append((lbl, nom, vals))
                        else:
                            # Uzun format (her satır bir tekrar)
                            for lbl, grp in df_acc_up.groupby(qc_col, sort=False):
                                try:
                                    nom = float(grp[nom_col_up].iloc[0])
                                except (ValueError, TypeError):
                                    continue
                                vals = grp[ip_col_up].dropna().astype(float).tolist()
                                if vals:
                                    acc_tbl_data.append((str(lbl), nom, vals))
                        st.success(f"✓ {len(acc_tbl_data)} QC levels loaded: "
                                   f"{[d[0] for d in acc_tbl_data]}")
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
        else:
            default_acc = pd.DataFrame({
                'QC Level':    ['LLOQ', 'LQC', 'MQC', 'HQC'],
                f'Nominal C ({conc_unit})': [_lloq_v or None, _lqc_v or None,
                                              _mqc_v or None, _hqc_v or None],
                'ip_1 (µA)': [None]*4, 'ip_2 (µA)': [None]*4,
                'ip_3 (µA)': [None]*4, 'ip_4 (µA)': [None]*4,
                'ip_5 (µA)': [None]*4, 'ip_6 (µA)': [None]*4,
            })
            acc_tbl = st.data_editor(default_acc, num_rows="dynamic",
                                      use_container_width=True, key="acc_tbl")
            nom_col  = acc_tbl.columns[1]
            ip_cols  = [c for c in acc_tbl.columns if c.startswith('ip_')]
            for _, row in acc_tbl.iterrows():
                lbl = str(row['QC Level']).strip() if pd.notna(row['QC Level']) else ''
                if not lbl or lbl == 'None':
                    continue
                try:
                    nom = float(row[nom_col])
                except (ValueError, TypeError):
                    continue
                vals = []
                for c in ip_cols:
                    try:
                        if pd.notna(row[c]):
                            vals.append(float(row[c]))
                    except (ValueError, TypeError):
                        pass
                if vals:
                    acc_tbl_data.append((lbl, nom, vals))

        if st.button("▶ Compute Accuracy", key="btn_acc", type="secondary"):
            _qc_now = st.session_state.get("qc_points", {})
            if not _qc_now and not cal_data:
                st.warning("Click 'Compute QC Points' first, or enter calibration data.")
            else:
                # Slope/intercept al — session'dan veya cal_data'dan hesapla
                if _qc_now:
                    _sl, _ic = _qc_now["slope"], _qc_now["intercept"]
                else:
                    from scipy import stats as _stats2
                    _c2 = np.array(sorted(cal_data.keys()), float)
                    _i2 = np.array([float(np.mean(cal_data[c])) for c in sorted(cal_data)], float)
                    _sl, _ic, *_ = _stats2.linregress(_c2, _i2)

                acc_rows = []
                for lbl, nom_c, vals in acc_tbl_data:
                    if not lbl or not vals or nom_c == 0:
                        continue
                    ip_mean  = float(np.mean(vals))
                    ip_std   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    ip_rsd   = (ip_std / ip_mean * 100) if ip_mean != 0 else 0.0
                    calc_c   = (ip_mean - _ic) / _sl if _sl != 0 else float('nan')
                    rec_pct  = (calc_c / nom_c * 100) if nom_c != 0 else float('nan')
                    acc_rows.append({
                        'QC Level': lbl,
                        f'Nominal C ({conc_unit})': nom_c,
                        'n': len(vals),
                        'ip ort. (µA)': round(ip_mean, 4),
                        'ip SD (µA)': round(ip_std, 4),
                        '%RSD': round(ip_rsd, 2),
                        f'Hesaplanan C ({conc_unit})': round(calc_c, 4),
                        'Recovery (%)': round(rec_pct, 2),
                        'ICH Kabul (98–102%)': "✅" if 98 <= rec_pct <= 102 else "⚠️"
                    })

                if acc_rows:
                    acc_res_df = pd.DataFrame(acc_rows)
                    st.dataframe(acc_res_df, use_container_width=True, hide_index=True)
                    all_pass = all(98 <= r['Recovery (%)'] <= 102 for r in acc_rows
                                   if not np.isnan(r['Recovery (%)']))
                    if all_pass:
                        st.success("✅ All QC levels meet the ICH Q2(R2) accuracy criterion (98–102%)")
                    else:
                        st.warning("⚠️ Some QC levels are outside the 98–102% range.")
                    # Kaydet
                    _save_analysis("cal_accuracy",
                                   {r['QC Level']: r['Recovery (%)'] for r in acc_rows},
                                   None)
                else:
                    st.warning("Enter at least 1 complete row of data in the table.")

    if cal_data:
        if "cal" in _cur().get("analyses", {}):
            st.warning("⚠️ A calibration analysis is already saved for this project. Running again will overwrite it.")
    if cal_data and st.button("▶ Run Calibration Analysis",type="primary",use_container_width=True):
        with st.spinner("⚙️ Computing calibration curve, LOD/LOQ, sensitivity…"):
            tmpdir=tempfile.mkdtemp()
            out=os.path.join(tmpdir,"cal_out"); os.makedirs(out)
            try:
                _cal_ps = dict(st.session_state.get("plot_style") or {})
                _cal_ps["colors"] = st.session_state.get(
                    "cal_custom_colors", _cal_ps.get("colors", ['#1a3a5c','#c55a11']))
                res=run_analytical_performance(
                    calibration_source=cal_data,analyte_name=analyte,
                    electrode_area_cm2=float(area_cal) if area_cal>0 else None,
                    unit=conc_unit,output_dir=out,
                    plot_style=_cal_ps)
                st.success("✅ Done — results saved to project")
                cr=res.get('calibration',res)
                lr=cr.get('linear_range',['?','?'])
                metrics={
                    "LOD":    f"{cr.get('LOD',0):.4g} {conc_unit}",
                    "LOQ":    f"{cr.get('LOQ',0):.4g} {conc_unit}",
                    "r":      f"{cr.get('r',cr.get('R2',0)):.5f}",
                    "Slope":  f"{cr.get('slope',0):.4f} µA/{conc_unit}",
                    "Intercept": f"{cr.get('intercept',0):.4f} µA",
                    "Linear range": f"{lr[0]:.4g}–{lr[1]:.4g} {conc_unit}",
                }
                m1,m2,m3,m4=st.columns(4)
                m1.metric("LOD",metrics["LOD"])
                m2.metric("LOQ",metrics["LOQ"])
                m3.metric("r",  metrics["r"])
                m4.metric("Slope",metrics["Slope"])
                _cal_notes = _analysis_notes_widget("cal")
                _save_analysis("cal",metrics,out, notes=_cal_notes)
                _metrics_csv_btn(metrics, "calibration_metrics.csv")
                st.markdown('<div class="sec">Plots</div>',unsafe_allow_html=True)
                _show_imgs(out); _dl_btn(out,"calibration.zip")
                _ai_interpret_btn("cal",interpret_calibration,
                                   {"results":res,"analyte":analyte,"unit":conc_unit},
                                   btn_key="ai_cal_run")
            except Exception as e:
                st.error(str(e))
                import traceback; st.code(traceback.format_exc())
    elif _cur().get("analyses",{}).get("cal"):
        a=_cur()["analyses"]["cal"]
        st.info(f"Previous calibration from {a.get('timestamp','')} stored.")
        _ai_interpret_btn("cal",interpret_calibration,{"results":{},"analyte":analyte,"unit":conc_unit},
                          btn_key="ai_cal_prev")


# ═══════════════════════════════════════════════════════════
# ➕  STANDARD ADDITION
# ═══════════════════════════════════════════════════════════
elif page == "➕  Standard Addition":
    st.title("➕ Standard Addition")
    st.caption(f"Project: **{_cur().get('name','')}**")

    left,right=st.columns([1,1])
    with left:
        st.markdown('<div class="sec">Parameters</div>',unsafe_allow_html=True)
        sa_analyte=st.text_input("Analyte name",value="Analyte")
        _SA_UNITS =["μM","mM","nM","mg/mL","μg/mL","ng/mL","mg/L","μg/L","ng/L","mol/L","mmol/L"]
        _sa_u_sel =st.selectbox("Concentration unit",_SA_UNITS,
                                 index=_SA_UNITS.index(st.session_state.get("sa_unit","μM")),
                                 key="sa_unit_sel")
        st.session_state["sa_unit"]=_sa_u_sel
        sa_cu     =_sa_u_sel
    with right:
        st.markdown('<div class="sec">Input mode</div>',unsafe_allow_html=True)
        sa_mode=_input_mode("sa_mode")

    additions,ip_vals=[],[]

    if sa_mode == "📁 Upload CSV":
        _csv_template_btn(
            "added_conc_uM,ip_uA\n"
            "0.0,2.145\n5.0,3.482\n10.0,4.819\n15.0,6.156\n20.0,7.493\n25.0,8.830\n",
            "standard_addition_template.csv",
            "⬇️ Download Standard Addition Template"
        )
        sa_up=st.file_uploader("Standard addition CSV",type="csv")
        if sa_up:
            _csv_preview(sa_up, key_prefix="sa")
            df_sa2=pd.read_csv(sa_up)
            additions=df_sa2.iloc[:,0].tolist()
            ip_vals  =df_sa2.iloc[:,1].tolist()
    else:
        st.markdown('<div class="sec">Enter standard addition data</div>',unsafe_allow_html=True)
        default_sa=pd.DataFrame({
            f'Addition ({sa_cu})':[0.0,None,None,None,None,None],
            'ip (µA)':[None]*6,
        })
        sa_tbl=st.data_editor(default_sa,num_rows="dynamic",
                               use_container_width=True,key="sa_tbl")
        clean_sa=sa_tbl.dropna()
        if len(clean_sa)>=3:
            additions=clean_sa.iloc[:,0].tolist()
            ip_vals  =clean_sa.iloc[:,1].tolist()

    if additions and ip_vals:
        if "sa" in _cur().get("analyses", {}):
            st.warning("⚠️ A standard addition analysis is already saved for this project. Running again will overwrite it.")
    if additions and ip_vals and st.button("▶ Run Standard Addition",type="primary",use_container_width=True):
        with st.spinner("Computing…"):
            tmpdir=tempfile.mkdtemp()
            out=os.path.join(tmpdir,"sa_out"); os.makedirs(out)
            try:
                res=standard_addition(additions_uM=additions,ip_uA=ip_vals,
                                       analyte_name=sa_analyte,output_dir=out,
                                       plot_style=st.session_state.get("plot_style"))
                st.success("✅ Done — results saved to project")
                _sa_slope = res.get("slope", 0)
                metrics={
                    f"C_sample ({sa_cu})": f"{res.get('C_sample_uM',0):.4f}",
                    "R²":   f"{res.get('R2',0):.5f}",
                    "Slope":f"{_sa_slope:.4f} µA/{sa_cu}",
                }
                m1,m2,m3=st.columns(3)
                m1.metric(f"C_sample ({sa_cu})",metrics[f"C_sample ({sa_cu})"])
                m2.metric("R²",metrics["R²"])
                m3.metric("Slope",metrics["Slope"])

                # ── Matrix effect: compare SA slope vs calibration slope ───
                _cal_metrics = _cur().get("analyses", {}).get("cal", {}).get("metrics", {})
                _cal_slope_str = str(_cal_metrics.get("Slope", ""))
                try:
                    # Parse "X.XXXX µA/µM" or plain number
                    _cal_slope_val = float(_cal_slope_str.split()[0])
                except (ValueError, IndexError):
                    _cal_slope_val = None

                # Also try DPV calibration
                if _cal_slope_val is None:
                    _dpv_metrics = _cur().get("analyses", {}).get("dpv", {}).get("metrics", {})
                    _dpv_slope_str = str(_dpv_metrics.get("Slope", ""))
                    try:
                        _cal_slope_val = float(_dpv_slope_str.split()[0])
                    except (ValueError, IndexError):
                        _cal_slope_val = None

                if _cal_slope_val and abs(_cal_slope_val) > 0:
                    _me_pct = (_sa_slope / _cal_slope_val - 1) * 100
                    _me_abs = abs(_me_pct)
                    _me_color = "✅" if _me_abs <= 10 else "⚠️"
                    _me_interp = ("negligible — no significant matrix effect" if _me_abs <= 5
                                  else "moderate — consider matrix-matched calibration" if _me_abs <= 15
                                  else "significant — matrix effect present; use standard addition or matrix-matched standards")
                    st.markdown('<div class="sec">Matrix Effect Assessment</div>', unsafe_allow_html=True)
                    _me1, _me2, _me3 = st.columns(3)
                    _me1.metric("Calibration slope", f"{_cal_slope_val:.4f}")
                    _me2.metric("SA slope", f"{_sa_slope:.4f}")
                    _me3.metric("Matrix effect", f"{_me_pct:+.1f}%", delta=f"{_me_color} {_me_interp.split('—')[0].strip()}")
                    if _me_abs > 10:
                        st.warning(f"⚠️ Matrix effect = {_me_pct:+.1f}% — {_me_interp}.")
                    else:
                        st.success(f"✅ Matrix effect = {_me_pct:+.1f}% — {_me_interp}.")
                    metrics["Calibration slope"] = f"{_cal_slope_val:.4f}"
                    metrics["SA slope"] = f"{_sa_slope:.4f}"
                    metrics["Matrix effect (%)"] = f"{_me_pct:+.1f}%"
                    metrics["Matrix effect verdict"] = _me_interp.split("—")[0].strip()
                else:
                    st.info("💡 Run Calibration or DPV Calibration first to enable matrix effect comparison (SA slope vs calibration slope).")
                _sa_notes = _analysis_notes_widget("sa")
                _save_analysis("sa",metrics,out, notes=_sa_notes)
                _metrics_csv_btn(metrics, "standard_addition_metrics.csv")
                st.markdown('<div class="sec">Plots</div>',unsafe_allow_html=True)
                _show_imgs(out); _dl_btn(out,"standard_addition.zip")
                _ai_interpret_btn("sa",interpret_standard_addition,
                                   {"results":res,"analyte":sa_analyte,"unit":sa_cu})
            except Exception as e:
                st.error(str(e))
                import traceback; st.code(traceback.format_exc())
    elif _cur().get("analyses",{}).get("sa"):
        a=_cur()["analyses"]["sa"]
        st.info(f"Previous std addition from {a.get('timestamp','')} stored.")
        _ai_interpret_btn("sa",interpret_standard_addition,
                           {"results":{},"analyte":sa_analyte,"unit":sa_cu})


# ═══════════════════════════════════════════════════════════
# 🧬  SEÇİCİLİK / İNTERFERANS
# ═══════════════════════════════════════════════════════════
elif page == "🧬  Selectivity":
    st.title("🧬 Selectivity / Interferent Study")
    st.caption(
        "Quantify the effect of potential interferents on the analyte signal. "
        "Acceptance criterion: % signal change ≤ ±5 % (user-adjustable)."
    )

    # ── Template indirme + CSV yükleme ──────────────────────────────────────
    with st.expander("📋 CSV Template — download & upload", expanded=False):
        _sel_tmpl_lines = [
            "# SilverstoneLab — Selectivity Template",
            "# Sütunlar: Interferent adı | Konsantrasyon (µM) | Ölçülen sinyal (µA)",
            "# Analite referans sinyalini aşağıdaki satıra girin (µA cinsinden):",
            "# Reference_Signal_uA: 0.0",
            "Interferent,Concentration_uM,Signal_uA",
            "Ascorbic Acid,100.0,0.0",
            "Uric Acid,100.0,0.0",
            "Epinephrine,100.0,0.0",
            "Glucose,500.0,0.0",
            "NaCl,500.0,0.0",
            "Urea,500.0,0.0",
        ]
        _sel_tmpl_bytes = "\n".join(_sel_tmpl_lines).encode("utf-8")
        _stcol1, _stcol2 = st.columns(2)
        with _stcol1:
            st.markdown("**1 · Download Template**")
            st.download_button(
                "⬇️ Download Selectivity Template (.csv)",
                data=_sel_tmpl_bytes,
                file_name="selectivity_template.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Sarı (Signal_uA) sütununu doldurun, diğerlerini gerekirse düzenleyin.")
        with _stcol2:
            st.markdown("**2 · Upload Filled Template**")
            _sel_up = st.file_uploader(
                "Upload selectivity CSV", type=["csv"], key="sel_csv_up"
            )
            if _sel_up:
                # Sadece yeni bir dosya yüklendiğinde işle; aynı dosyaysa rerun yok
                if st.session_state.get("sel_csv_loaded") != _sel_up.name:
                    try:
                        import io as _io_sel_up
                        _sel_raw = _sel_up.read().decode("utf-8", errors="replace")
                        # Metadata satırından referans sinyali oku
                        _ref_from_csv = None
                        for _line in _sel_raw.splitlines():
                            _line = _line.strip()
                            if _line.lower().startswith("# reference_signal_ua:"):
                                try:
                                    _ref_from_csv = float(_line.split(":", 1)[1].strip())
                                except ValueError:
                                    pass
                                break
                        _df_sel = pd.read_csv(_io_sel_up.BytesIO(_sel_raw.encode("utf-8")), comment="#")
                        # Normalize sütun isimleri
                        _df_sel.columns = [c.strip() for c in _df_sel.columns]
                        _name_col = _df_sel.columns[0]
                        _conc_col = _df_sel.columns[1] if len(_df_sel.columns) > 1 else None
                        _sig_col  = _df_sel.columns[2] if len(_df_sel.columns) > 2 else None
                        _n_loaded = len(_df_sel)
                        # Session state'e yaz → widget'lar otomatik dolacak
                        st.session_state["sel_n_int"] = _n_loaded
                        for _idx, _row in _df_sel.iterrows():
                            st.session_state[f"sel_int_name_{_idx}"] = str(_row[_name_col]).strip()
                            try:
                                st.session_state[f"sel_int_conc_{_idx}"] = float(_row[_conc_col]) if _conc_col else 100.0
                            except (TypeError, ValueError):
                                st.session_state[f"sel_int_conc_{_idx}"] = 100.0
                            try:
                                st.session_state[f"sel_int_sig_{_idx}"] = float(_row[_sig_col]) if _sig_col else 0.0
                            except (TypeError, ValueError):
                                st.session_state[f"sel_int_sig_{_idx}"] = 0.0
                        # Referans sinyal varsa widget'a yaz
                        if _ref_from_csv is not None:
                            st.session_state["sel_ref_sig"] = _ref_from_csv
                        # Yüklenen dosya adını kaydet — bir sonraki rerun'da tekrar işleme
                        st.session_state["sel_csv_loaded"] = _sel_up.name
                        _ref_msg = f" | Referans sinyal: {_ref_from_csv:.4f} µA" if _ref_from_csv is not None else " | ⚠️ Referans sinyal CSV'de bulunamadı — aşağıdaki alana giriniz."
                        st.success(f"✓ {_n_loaded} interferent yüklendi{_ref_msg}")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"CSV okunamadı: {_e}")

    _sel_analyte = st.text_input("Analyte", value=st.session_state.get("sel_analyte","Dopamine"), key="sel_analyte")
    _sel_unit    = st.text_input("Unit", value=st.session_state.get("cal_unit","µM"), key="sel_unit_inp")
    _sel_conc    = st.number_input("Analyte concentration (µM)", value=10.0, min_value=0.0001, format="%.4f", key="sel_conc",
                                    help="Concentration of the target analyte used in the selectivity test. Interferents are typically tested at a 10× excess (e.g. 100 µM interferent vs 10 µM analyte).")
    _sel_ref     = st.number_input("Reference signal — analyte alone (µA)", value=0.0, format="%.4f", key="sel_ref_sig",
                                    help="Peak current measured for the analyte alone (no interferent present). Used as the 100% reference for all tolerance calculations.")
    _sel_thresh  = st.number_input("Acceptance threshold (± %)", value=5.0, min_value=0.1, max_value=50.0, format="%.1f", key="sel_thresh",
                                    help="Maximum allowed signal change (%) due to an interferent. ICH Q2(R2) commonly uses ±5% for pharmaceutical sensors. Biosensors in complex matrices may use ±10–15%.")

    st.markdown("#### Interferent Table")
    st.caption("One interferent per row — veya yukarıdan CSV yükleyin.")

    _n_int = st.number_input("Number of interferents", min_value=1, max_value=20,
                             value=int(st.session_state.get("sel_n_int", 6)),
                             step=1, key="sel_n_int")
    _int_rows = []
    _icols = st.columns([3, 2, 2])
    _icols[0].markdown("**Interferent**")
    _icols[1].markdown("**Concentration (µM)**")
    _icols[2].markdown("**Signal (µA)**")
    for _i in range(int(_n_int)):
        _c1,_c2,_c3 = st.columns([3,2,2])
        _iname = _c1.text_input(f"Interferent {_i+1}", key=f"sel_int_name_{_i}",
                                 label_visibility="collapsed",
                                 placeholder=["Ascorbic Acid","Uric Acid","Epinephrine","Glucose","NaCl","Urea"][_i] if _i < 6 else f"Int-{_i+1}")
        _iconc = _c2.number_input("", value=100.0, min_value=0.0, format="%.2f",
                                   key=f"sel_int_conc_{_i}", label_visibility="collapsed")
        _isig  = _c3.number_input("", value=0.0, format="%.4f",
                                   key=f"sel_int_sig_{_i}", label_visibility="collapsed")
        if _iname.strip():          # 0.0 sinyal geçerli; sadece boş isim filtrele
            _int_rows.append((_iname.strip(), _iconc, _isig))

    if _sel_ref <= 0:
        st.warning("⚠️ Reference signal (analyte alone) = 0 — lütfen yukarıdaki alana analite sinyalini girin.")

    if st.button("▶ Run Selectivity", type="primary", use_container_width=True, key="btn_sel_run"):
        if not _int_rows:
            st.warning("En az bir interferent satırı doldurun (isim alanı boş olmamalı).")
        elif _sel_ref <= 0:
            st.warning("Referans sinyal > 0 olmalı — analiteyi yalnız ölçüp 'Reference signal' alanına girin.")
        else:
            _sel_results = []
            for _iname, _iconc, _isig in _int_rows:
                _delta_pct = (_isig - _sel_ref) / _sel_ref * 100
                _passed    = abs(_delta_pct) <= _sel_thresh
                _sel_results.append({
                    "Interferent":        _iname,
                    "Concentration (µM)": _iconc,
                    "Signal (µA)":        round(_isig, 4),
                    "% Signal Change":    round(_delta_pct, 2),
                    "Status":             "✅ Pass" if _passed else "⚠️ Fail",
                })
            st.session_state["sel_results"] = _sel_results
            st.session_state["sel_ref"]     = _sel_ref

    if "sel_results" in st.session_state:
        _sr = st.session_state["sel_results"]
        _sdf = pd.DataFrame(_sr)
        st.dataframe(_sdf, hide_index=True, use_container_width=True)

        _pass_all = all(r["Status"].startswith("✅") for r in _sr)
        if _pass_all:
            st.success(f"✅ All interferents within ± {st.session_state.get('sel_thresh',5):.1f}% threshold.")
        else:
            _fail_list = [r["Interferent"] for r in _sr if not r["Status"].startswith("✅")]
            st.warning(f"⚠️ Interferents outside criterion: {', '.join(_fail_list)}")

        import matplotlib.pyplot as plt
        import io as _io_sel
        _fig_sel, _ax_sel = plt.subplots(figsize=(max(6, len(_sr)*0.9), 4))
        _names   = [r["Interferent"] for r in _sr]
        _deltas  = [r["% Signal Change"] for r in _sr]
        _colors_bar = ["#2ecc71" if abs(d) <= st.session_state.get("sel_thresh",5) else "#e74c3c" for d in _deltas]
        _ax_sel.bar(_names, _deltas, color=_colors_bar, edgecolor="white", linewidth=0.5)
        _ax_sel.axhline(y= st.session_state.get("sel_thresh",5), color="#e74c3c", linestyle="--", linewidth=1.2, label=f"+{st.session_state.get('sel_thresh',5):.0f}%")
        _ax_sel.axhline(y=-st.session_state.get("sel_thresh",5), color="#e74c3c", linestyle="--", linewidth=1.2, label=f"-{st.session_state.get('sel_thresh',5):.0f}%")
        _ax_sel.axhline(y=0, color="gray", linewidth=0.8)
        _ax_sel.set_xlabel("Interferent", fontsize=10)
        _ax_sel.set_ylabel("% Signal Change", fontsize=10)
        _ax_sel.set_title(f"{_sel_analyte} — Selectivity Study", fontsize=11, fontweight="bold")
        plt.xticks(rotation=35, ha="right", fontsize=8)
        _ax_sel.legend(fontsize=8)
        plt.tight_layout()
        # Save 300 DPI bytes for report
        _sel_buf = _io_sel.BytesIO()
        _fig_sel.savefig(_sel_buf, format="png", dpi=300, bbox_inches="tight")
        st.session_state["sel_plot_bytes"] = [_sel_buf.getvalue()]
        st.pyplot(_fig_sel)
        plt.close(_fig_sel)

        _sel_csv = _sdf.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", _sel_csv,
                           file_name=f"{_sel_analyte}_selectivity.csv",
                           mime="text/csv", use_container_width=True)

# ═══════════════════════════════════════════════════════════
# 📅  STABİLİTE
# ═══════════════════════════════════════════════════════════
elif page == "📅  Stability":
    st.title("📅 Sensor Stability")
    st.caption(
        "Track performance loss of the sensor over time. "
        "Acceptance criterion: ≥ 90% of initial signal retained (user-adjustable)."
    )

    _stab_analyte = st.text_input("Analyte", value=st.session_state.get("sel_analyte","Dopamine"), key="stab_analyte")
    _stab_nominal = st.number_input("Test concentration (µM)", value=10.0, min_value=0.001, format="%.4f", key="stab_nominal",
                                     help="Known concentration of analyte used in stability tests. Recovery at each time point = (measured signal / day-0 signal) × 100%.")
    _stab_thresh  = st.number_input("Min. accepted % recovery", value=90.0, min_value=50.0, max_value=100.0, format="%.1f", key="stab_thresh",
                                     help="Minimum acceptable signal recovery (%). ICH Q2(R2) typically requires ≥ 90%. Lower thresholds may apply for complex matrix sensors or long storage studies.")
    _stab_unit    = st.text_input("Unit", value="µM", key="stab_unit")

    _csv_template_btn(
        "Day,Signal_uA,SD_uA\n"
        "0,8.542,0.112\n1,8.498,0.098\n3,8.421,0.134\n7,8.312,0.156\n"
        "14,8.187,0.189\n21,8.054,0.201\n28,7.923,0.223\n",
        "stability_template.csv",
        "⬇️ Download Stability Template (Day / Signal / SD)"
    )
    _stab_csv_up = st.file_uploader("Upload stability CSV (Day, Mean_signal_uA, [SD])", type="csv", key="stab_csv_up")
    _stab_rows = []
    if _stab_csv_up:
        _csv_preview(_stab_csv_up, key_prefix="stab")
        try:
            _stab_df_up = pd.read_csv(_stab_csv_up)
            _cols = [c.lower().strip() for c in _stab_df_up.columns]
            _day_col  = next((i for i,c in enumerate(_cols) if "day" in c or "gün" in c or "gun" in c), 0)
            _sig_col  = next((i for i,c in enumerate(_cols) if "signal" in c or "sinyal" in c or "ip" in c or "mean" in c), 1)
            _sd_col   = next((i for i,c in enumerate(_cols) if "sd" in c or "std" in c), None)
            for _, _row in _stab_df_up.iterrows():
                try:
                    _d  = int(float(str(_row.iloc[_day_col])))
                    _s  = float(str(_row.iloc[_sig_col]))
                    _sd = float(str(_row.iloc[_sd_col])) if _sd_col is not None else 0.0
                    _stab_rows.append((_d, _s, _sd))
                except (ValueError, TypeError):
                    continue
            st.success(f"✓ {len(_stab_rows)} days loaded")
        except Exception as _e:
            st.error(f"CSV read error: {_e}")

    if not _stab_rows:
        st.markdown("**or enter manually:**")
        _n_days = st.number_input("Number of measurement days", min_value=2, max_value=30, value=7, step=1, key="stab_n_days")
        _sc1, _sc2, _sc3 = st.columns([2, 2, 2])
        _sc1.markdown("**Day**"); _sc2.markdown("**Mean (µA)**"); _sc3.markdown("**SD (µA)**")
        for _i in range(int(_n_days)):
            _d1, _d2, _d3 = st.columns([2, 2, 2])
            _day_v  = _d1.number_input("", value=_i * (7 if _n_days > 7 else 1), min_value=0,
                                        key=f"stab_day_{_i}", label_visibility="collapsed")
            _sig_v  = _d2.number_input("", value=0.0, format="%.4f",
                                        key=f"stab_sig_{_i}", label_visibility="collapsed")
            _sd_v   = _d3.number_input("", value=0.0, min_value=0.0, format="%.4f",
                                        key=f"stab_sd_{_i}", label_visibility="collapsed")
            if _sig_v > 0:
                _stab_rows.append((int(_day_v), _sig_v, _sd_v))

    if st.button("▶ Run Stability", type="primary", use_container_width=True, key="btn_stab_run"):
        if len(_stab_rows) < 2:
            st.warning("At least 2 measurement days required.")
        else:
            _stab_rows_sorted = sorted(_stab_rows, key=lambda x: x[0])
            _ref_sig = _stab_rows_sorted[0][1]
            _stab_res = []
            for _day, _sig, _sd in _stab_rows_sorted:
                _rec = _sig / _ref_sig * 100 if _ref_sig != 0 else float("nan")
                _rsd = (_sd / _sig * 100) if _sig != 0 else 0.0
                _stab_res.append({
                    "Day":          _day,
                    "Signal (µA)":  round(_sig, 4),
                    "SD (µA)":      round(_sd, 4),
                    "%RSD":         round(_rsd, 2),
                    "% Recovery":   round(_rec, 2),
                    "Status":       "✅" if _rec >= _stab_thresh else "⚠️",
                })
            st.session_state["stab_results"] = _stab_res

    if "stab_results" in st.session_state:
        _sr2 = st.session_state["stab_results"]
        _thresh2 = st.session_state.get("stab_thresh", 90.0)
        st.dataframe(pd.DataFrame(_sr2), hide_index=True, use_container_width=True)

        _last_day_pass = _sr2[-1]["Status"] == "✅"
        _last_rec = _sr2[-1]["% Recovery"]
        if _last_day_pass:
            st.success(f"✅ Day {_sr2[-1]['Day']}: % recovery = {_last_rec:.1f}% — criterion met (≥{_thresh2:.0f}%)")
        else:
            st.warning(f"⚠️ Day {_sr2[-1]['Day']}: % recovery = {_last_rec:.1f}% — criterion not met (<{_thresh2:.0f}%)")

        # ── Linear trend test (signal drift significance) ───────────
        from scipy.stats import linregress as _lr_stab
        _stab_days_arr2  = [r["Day"]           for r in _sr2]
        _stab_sigs_arr2  = [r["Signal (µA)"]   for r in _sr2]
        _stab_lr = _lr_stab(_stab_days_arr2, _stab_sigs_arr2)
        _stab_slope_per_day = _stab_lr.slope
        _stab_p = _stab_lr.pvalue
        _stab_r2 = _stab_lr.rvalue ** 2
        _drift_pct_total = (_stab_slope_per_day * (_stab_days_arr2[-1] - _stab_days_arr2[0])
                            / _stab_sigs_arr2[0] * 100) if _stab_sigs_arr2[0] else 0

        st.markdown("**Signal Drift Analysis**")
        _sd1, _sd2, _sd3 = st.columns(3)
        _sd1.metric("Drift slope", f"{_stab_slope_per_day:.4f} µA/day")
        _sd2.metric("Total drift", f"{_drift_pct_total:+.1f}%")
        _sd3.metric("Trend p-value", f"{_stab_p:.4f}")

        if _stab_p < 0.05:
            if _stab_slope_per_day < 0:
                st.warning(f"⚠️ Statistically significant signal decline detected (slope = {_stab_slope_per_day:.4f} µA/day, p = {_stab_p:.4f}). "
                           f"This represents a {abs(_drift_pct_total):.1f}% total drop over the study period. "
                           "Report storage conditions and consider correcting for drift in long-term measurements.")
            else:
                st.info(f"ℹ️ Statistically significant positive drift detected (p = {_stab_p:.4f}). "
                        "This may indicate electrode conditioning or analyte accumulation.")
        else:
            st.success(f"✅ No statistically significant signal drift (p = {_stab_p:.4f} ≥ 0.05). "
                       "Signal is stable over the monitoring period.")

        import matplotlib.pyplot as plt
        import io as _io_stab
        _fig_stab, _ax_stab = plt.subplots(figsize=(8, 4))
        _days_arr = [r["Day"] for r in _sr2]
        _rec_arr  = [r["% Recovery"] for r in _sr2]
        _ax_stab.plot(_days_arr, _rec_arr, "o-", color="#3498db", linewidth=2, markersize=6, label="% Recovery")
        _ax_stab.axhline(y=_thresh2, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Threshold ({_thresh2:.0f}%)")
        _ax_stab.fill_between(_days_arr, _thresh2, _rec_arr,
                               where=[r >= _thresh2 for r in _rec_arr],
                               alpha=0.15, color="#2ecc71")
        _ax_stab.set_xlabel("Day", fontsize=10)
        _ax_stab.set_ylabel("% Recovery", fontsize=10)
        _ax_stab.set_title(f"{st.session_state.get('stab_analyte','Analyte')} — Sensor Stability", fontsize=11, fontweight="bold")
        _ax_stab.set_ylim(max(0, min(_rec_arr) - 10), 105)
        _ax_stab.legend(fontsize=9)
        plt.tight_layout()
        # Save 300 DPI bytes for report
        _stab_buf = _io_stab.BytesIO()
        _fig_stab.savefig(_stab_buf, format="png", dpi=300, bbox_inches="tight")
        st.session_state["stab_plot_bytes"] = [_stab_buf.getvalue()]
        st.pyplot(_fig_stab)
        plt.close(_fig_stab)

        _stab_csv_dl = pd.DataFrame(_sr2).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", _stab_csv_dl,
                           file_name=f"{st.session_state.get('stab_analyte','sensor')}_stability.csv",
                           mime="text/csv", use_container_width=True)

# ═══════════════════════════════════════════════════════════
# 🔬  YÜZEY KARAKTERİZASYONU
# ═══════════════════════════════════════════════════════════
elif page == "🔬  Surface Characterization":
    st.title("🔬 Surface Characterization")
    st.caption(
        "SEM images, EIS Nyquist comparison, and CV (bare vs modified) analysis — "
        "three key lines of evidence proving surface modification."
    )

    # ── Örnek veri seti (ZIP) ────────────────────────────────────────────────
    with st.expander("📦 Download Example Dataset (ZIP — 4 CSV files)", expanded=False):
        st.caption(
            "NiFe₂O₄/CSPE örnek verisi: EIS Nyquist (bare + modified) ve CV (bare + modified). "
            "Template olarak veya sonuçları karşılaştırmak için kullanın."
        )
        import io as _io_surf, zipfile as _zf_surf
        _surf_files = {
            "eis_bare_GCE.csv": (
                "# EIS Nyquist — Bare GCE | Z_real (Ω), -Z_imag (Ω)\n"
                "Zreal_Ohm,Zimag_Ohm\n"
                "12.4,0.6\n13.1,1.9\n14.8,5.2\n18.3,12.8\n28.6,28.4\n"
                "55.2,51.3\n112.4,78.6\n198.5,82.4\n285.3,65.8\n348.2,42.1\n"
                "392.6,24.8\n418.4,13.2\n430.1,6.8\n436.5,3.4\n439.2,1.8\n"
            ),
            "eis_NiFe2O4_CSPE.csv": (
                "# EIS Nyquist — NiFe₂O₄/CSPE | Z_real (Ω), -Z_imag (Ω)\n"
                "Zreal_Ohm,Zimag_Ohm\n"
                "8.2,0.4\n8.8,1.4\n10.1,3.8\n13.4,9.2\n20.1,19.8\n"
                "36.8,33.5\n62.4,46.2\n92.1,42.8\n118.3,30.4\n134.6,18.8\n"
                "143.2,10.6\n147.8,5.8\n150.1,3.1\n151.4,1.6\n152.0,0.9\n"
            ),
            "cv_bare_GCE.csv": (
                "# CV — Bare GCE | Potential (V vs Ag/AgCl), Current (µA)\n"
                "# Çözelti: 5 mM [Fe(CN)6]³⁻/⁴⁻ in 0.1 M KCl | v = 50 mV/s\n"
                "Potential_V,Current_uA\n"
                "-0.20,0.42\n-0.15,0.68\n-0.10,0.95\n-0.05,1.24\n"
                "0.00,1.52\n0.05,1.78\n0.10,2.08\n0.15,2.42\n"
                "0.20,2.84\n0.25,3.21\n0.30,3.48\n0.35,3.42\n"
                "0.40,3.08\n0.45,2.65\n0.50,2.24\n0.55,1.92\n"
                "0.60,1.68\n0.55,1.42\n0.50,1.18\n0.45,0.95\n"
                "0.40,0.71\n0.35,0.44\n0.30,0.14\n0.25,-0.18\n"
                "0.20,-0.52\n0.15,-0.88\n0.10,-1.24\n0.05,-1.48\n"
                "0.00,-1.58\n-0.05,-1.52\n-0.10,-1.38\n-0.15,-1.18\n"
                "-0.20,-0.95\n"
            ),
            "cv_NiFe2O4_CSPE.csv": (
                "# CV — NiFe₂O₄/CSPE | Potential (V vs Ag/AgCl), Current (µA)\n"
                "# Çözelti: 5 mM [Fe(CN)6]³⁻/⁴⁻ in 0.1 M KCl | v = 50 mV/s\n"
                "Potential_V,Current_uA\n"
                "-0.20,1.18\n-0.15,1.84\n-0.10,2.58\n-0.05,3.42\n"
                "0.00,4.28\n0.05,5.14\n0.10,6.08\n0.15,7.12\n"
                "0.20,8.28\n0.25,9.42\n0.30,10.14\n0.35,9.88\n"
                "0.40,8.95\n0.45,7.82\n0.50,6.68\n0.55,5.78\n"
                "0.60,5.08\n0.55,4.42\n0.50,3.78\n0.45,3.14\n"
                "0.40,2.44\n0.35,1.68\n0.30,0.82\n0.25,-0.14\n"
                "0.20,-1.28\n0.15,-2.52\n0.10,-3.68\n0.05,-4.52\n"
                "0.00,-4.84\n-0.05,-4.68\n-0.10,-4.28\n-0.15,-3.72\n"
                "-0.20,-3.08\n"
            ),
        }
        _zip_buf_surf = _io_surf.BytesIO()
        with _zf_surf.ZipFile(_zip_buf_surf, "w", _zf_surf.ZIP_DEFLATED) as _zf:
            for _fname, _fcontent in _surf_files.items():
                _zf.writestr(_fname, _fcontent)
        st.download_button(
            "⬇️ Download NiFe₂O₄/CSPE Example Dataset (.zip)",
            data=_zip_buf_surf.getvalue(),
            file_name="surface_charact_example_dataset.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.caption("ZIP içeriği: `eis_bare_GCE.csv`, `eis_NiFe2O4_CSPE.csv`, `cv_bare_GCE.csv`, `cv_NiFe2O4_CSPE.csv`")

    _surf_electrode = st.text_input("Electrode / material name",
                                     value=st.session_state.get("surf_electrode", _cur().get("name","Modified Electrode")),
                                     key="surf_electrode_inp",
                                     help="Name of the modified electrode system (e.g. 'Mn₃O₄/SPCE'). Used in AI interpretation and report headings.")
    st.session_state["surf_electrode"] = _surf_electrode

    _surf_tabs = st.tabs(["🖼️ SEM Images", "⚡ EIS Nyquist", "🌊 CV Comparison"])

    # ── SEM ──────────────────────────────────────────────────
    with _surf_tabs[0]:
        st.markdown("#### SEM Images")
        st.caption("Upload SEM images of the bare and modified electrode (PNG/JPG).")
        _sem_col1, _sem_col2 = st.columns(2)
        with _sem_col1:
            st.markdown("**Bare GCE**")
            _sem_bare = st.file_uploader("Bare GCE SEM", type=["png","jpg","jpeg","tif","tiff"], key="sem_bare_up")
            if _sem_bare:
                st.image(_sem_bare, caption="Bare GCE", use_container_width=True)
        with _sem_col2:
            st.markdown("**Modified Electrode**")
            _sem_mod = st.file_uploader("Modified SEM", type=["png","jpg","jpeg","tif","tiff"], key="sem_mod_up")
            if _sem_mod:
                st.image(_sem_mod, caption="Modified Electrode", use_container_width=True)
        _sem_note = st.text_area("SEM observation (optional)", height=80, key="sem_note",
                                  placeholder="e.g. The modified electrode surface shows MWCNT tube structures and a Nafion film layer.")
        if _sem_note:
            st.info(f"📝 Note: {_sem_note}")
        # Explicitly persist SEM upload state so other page sections can read it
        st.session_state["has_sem_bare"] = _sem_bare is not None
        st.session_state["has_sem_mod"]  = _sem_mod  is not None

    # ── EIS Nyquist ──────────────────────────────────────────
    with _surf_tabs[1]:
        st.markdown("#### EIS Nyquist Comparison")
        st.caption("Upload a CSV with Z_real (Ω) and -Z_imag (Ω) columns for each electrode, "
                   "or use the EIS module for full analysis.")

        _eis_col1, _eis_col2 = st.columns(2)
        _eis_data = {}
        with _eis_col1:
            st.markdown("**Bare GCE**")
            _csv_template_btn(
                "Zreal_Ohm,Zimag_Ohm\n"
                "15.2,0.9\n16.8,3.1\n19.4,8.5\n28.7,21.3\n62.4,55.8\n"
                "180.3,88.2\n340.5,72.1\n480.1,38.4\n510.2,22.0\n530.4,10.1\n",
                "eis_bare_template.csv",
                "⬇️ Download EIS Template"
            )
            _eis_bare_up = st.file_uploader("Bare GCE EIS CSV", type="csv", key="eis_bare_nyquist")
            if _eis_bare_up:
                try:
                    _eb_df = pd.read_csv(_eis_bare_up)
                    _ecols = [c.lower() for c in _eb_df.columns]
                    _zr_i  = next((i for i,c in enumerate(_ecols) if "zre" in c or "z_real" in c or "re" in c), 0)
                    _zi_i  = next((i for i,c in enumerate(_ecols) if "zim" in c or "z_imag" in c or "-z" in c or "im" in c), 1)
                    _eis_data["Bare GCE"] = (_eb_df.iloc[:, _zr_i].values, _eb_df.iloc[:, _zi_i].values)
                    st.success(f"✓ {len(_eb_df)} frequency points")
                except Exception as _e:
                    st.error(str(_e))
        with _eis_col2:
            st.markdown("**Modified Electrode**")
            _csv_template_btn(
                "Zreal_Ohm,Zimag_Ohm\n"
                "52.1,1.2\n54.3,4.5\n60.8,14.2\n85.4,38.7\n195.3,112.4\n"
                "480.2,195.1\n820.5,142.3\n1050.4,72.8\n1180.1,38.2\n1240.3,18.5\n",
                "eis_modified_template.csv",
                "⬇️ Download EIS Template"
            )
            _eis_mod_up = st.file_uploader("Modified EIS CSV", type="csv", key="eis_mod_nyquist")
            if _eis_mod_up:
                try:
                    _em_df = pd.read_csv(_eis_mod_up)
                    _ecols2 = [c.lower() for c in _em_df.columns]
                    _zr_i2  = next((i for i,c in enumerate(_ecols2) if "zre" in c or "z_real" in c or "re" in c), 0)
                    _zi_i2  = next((i for i,c in enumerate(_ecols2) if "zim" in c or "z_imag" in c or "-z" in c or "im" in c), 1)
                    _eis_data["Modified"] = (_em_df.iloc[:, _zr_i2].values, _em_df.iloc[:, _zi_i2].values)
                    st.success(f"✓ {len(_em_df)} frequency points")
                except Exception as _e:
                    st.error(str(_e))

        st.markdown("**Rct values (from equivalent circuit fit)**")
        # ── Auto-fill Rct from saved EIS analysis ──────────────
        _eis_saved = _cur().get("analyses", {}).get("eis", {})
        _eis_rct_list = _eis_saved.get("metrics", {}).get("Rct (Ω)", [])
        _eis_electrodes = _eis_saved.get("metrics", {}).get("electrodes", [])
        if _eis_rct_list and len(_eis_rct_list) >= 1:
            # Try to identify bare/modified by electrode filename
            def _is_bare(name: str) -> bool:
                return any(k in name.lower() for k in ("bare","gce","blank","ref","unmod"))
            if len(_eis_rct_list) >= 2 and _eis_electrodes:
                _bare_idx = next((i for i, n in enumerate(_eis_electrodes) if _is_bare(n)), 0)
                _mod_idx  = next((i for i, n in enumerate(_eis_electrodes) if not _is_bare(n)),
                                 1 if len(_eis_electrodes) > 1 else 0)
                _auto_rct_bare = float(_eis_rct_list[_bare_idx])
                _auto_rct_mod  = float(_eis_rct_list[_mod_idx])
                _rct_label = (f"🔗 EIS analizinden otomatik alındı — "
                              f"{_eis_electrodes[_bare_idx]} → Bare, "
                              f"{_eis_electrodes[_mod_idx]} → Modified")
            elif len(_eis_rct_list) == 1:
                _auto_rct_bare = float(_eis_rct_list[0])
                _auto_rct_mod  = 0.0
                _rct_label = "🔗 EIS analizinden 1 elektrot alındı — Modified değeri elle girin"
            else:
                _auto_rct_bare = _auto_rct_mod = 0.0
                _rct_label = ""
            if _auto_rct_bare > 0 and st.session_state.get("rct_bare", 0.0) == 0.0:
                st.session_state["rct_bare"] = _auto_rct_bare
            if _auto_rct_mod > 0 and st.session_state.get("rct_mod", 0.0) == 0.0:
                st.session_state["rct_mod"] = _auto_rct_mod
            if _rct_label:
                st.caption(_rct_label)
        # ───────────────────────────────────────────────────────
        _rc1, _rc2 = st.columns(2)
        _rct_bare = _rc1.number_input("Rct – Bare GCE (Ω)", value=0.0, min_value=0.0, format="%.1f", key="rct_bare",
                                       help="Charge-transfer resistance of the unmodified electrode from Randles equivalent circuit fit. Rct(bare) >> Rct(modified) means the modifier accelerates electron transfer.")
        _rct_mod  = _rc2.number_input("Rct – Modified (Ω)",  value=0.0, min_value=0.0, format="%.1f", key="rct_mod",
                                       help="Charge-transfer resistance after surface modification. A lower value indicates improved electrocatalytic activity. The ratio Rct(bare)/Rct(modified) quantifies the enhancement.")

        if _eis_data:
            import matplotlib.pyplot as plt
            _fig_eis2, _ax_eis2 = plt.subplots(figsize=(7, 5))
            _eis_colors = ["#2c3e50", "#e74c3c"]
            for (_lbl, (_zr, _zi)), _col in zip(_eis_data.items(), _eis_colors):
                _ax_eis2.plot(_zr, _zi, "o-", color=_col, markersize=4, linewidth=1.5, label=_lbl)
            _ax_eis2.set_xlabel("Z$_{real}$ (Ω)", fontsize=11)
            _ax_eis2.set_ylabel("-Z$_{imag}$ (Ω)", fontsize=11)
            _ax_eis2.set_title("EIS Nyquist — Surface Comparison", fontsize=12, fontweight="bold")
            _ax_eis2.set_aspect("equal", adjustable="box")
            _ax_eis2.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(_fig_eis2)
            plt.close(_fig_eis2)

        if _rct_bare > 0 and _rct_mod > 0:
            _rct_ratio = _rct_bare / _rct_mod
            _mc1b, _mc2b, _mc3b = st.columns(3)
            _mc1b.metric("Rct Bare (Ω)", f"{_rct_bare:.1f}")
            _mc2b.metric("Rct Modified (Ω)", f"{_rct_mod:.1f}")
            _mc3b.metric("Rct Ratio (Bare/Mod)", f"{_rct_ratio:.2f}×")
            if _rct_mod < _rct_bare:
                st.success("✅ Modified electrode shows lower Rct → improved electron transfer kinetics")
            else:
                st.info("ℹ️ Modified electrode shows higher Rct (expected for barrier-layer configurations)")

    # ── CV Comparison ─────────────────────────────────────────
    with _surf_tabs[2]:
        st.markdown("#### CV Comparison — Bare vs Modified")
        st.caption("Upload a CSV with potential (V) and current (µA) columns for each electrode, "
                   "or enter anodic/cathodic peak currents manually.")

        _cv_col1, _cv_col2 = st.columns(2)
        _cv_data = {}
        with _cv_col1:
            st.markdown("**Bare GCE**")
            _csv_template_btn(
                "Potential_V,Current_uA\n"
                "-0.6,1.42\n-0.5,0.95\n-0.4,0.48\n-0.3,-0.12\n-0.2,-0.65\n"
                "-0.1,-1.08\n0.0,-1.35\n0.1,-1.48\n0.2,-1.42\n0.3,-1.21\n"
                "0.4,-0.88\n0.5,-0.42\n0.6,-0.08\n",
                "cv_bare_template.csv",
                "⬇️ Download CV Template"
            )
            _cv_bare_up = st.file_uploader("Bare GCE CV CSV", type="csv", key="cv_bare_surf")
            if _cv_bare_up:
                try:
                    _cvb_df = pd.read_csv(_cv_bare_up)
                    _cvc1 = [c.lower() for c in _cvb_df.columns]
                    _vci  = next((i for i,c in enumerate(_cvc1) if "pot" in c or "v" in c or "e" in c), 0)
                    _ici  = next((i for i,c in enumerate(_cvc1) if "curr" in c or "i" in c or "µa" in c or "ua" in c or "a" in c), 1)
                    _cv_data["Bare GCE"] = (_cvb_df.iloc[:,_vci].values, _cvb_df.iloc[:,_ici].values)
                    st.success(f"✓ {len(_cvb_df)} points")
                except Exception as _e: st.error(str(_e))
        with _cv_col2:
            st.markdown("**Modified Electrode**")
            _csv_template_btn(
                "Potential_V,Current_uA\n"
                "-0.6,3.85\n-0.5,2.61\n-0.4,1.32\n-0.3,-0.31\n-0.2,-1.78\n"
                "-0.1,-2.95\n0.0,-3.68\n0.1,-4.02\n0.2,-3.87\n0.3,-3.31\n"
                "0.4,-2.42\n0.5,-1.15\n0.6,-0.21\n",
                "cv_modified_template.csv",
                "⬇️ Download CV Template"
            )
            _cv_mod_up = st.file_uploader("Modified CV CSV", type="csv", key="cv_mod_surf")
            if _cv_mod_up:
                try:
                    _cvm_df = pd.read_csv(_cv_mod_up)
                    _cvc2 = [c.lower() for c in _cvm_df.columns]
                    _vci2  = next((i for i,c in enumerate(_cvc2) if "pot" in c or "v" in c or "e" in c), 0)
                    _ici2  = next((i for i,c in enumerate(_cvc2) if "curr" in c or "i" in c or "µa" in c or "ua" in c or "a" in c), 1)
                    _cv_data["Modified"] = (_cvm_df.iloc[:,_vci2].values, _cvm_df.iloc[:,_ici2].values)
                    st.success(f"✓ {len(_cvm_df)} points")
                except Exception as _e: st.error(str(_e))

        if _cv_data:
            import matplotlib.pyplot as plt
            _fig_cv2, _ax_cv2 = plt.subplots(figsize=(8, 5))
            _cv_colors2 = ["#2c3e50", "#e74c3c"]
            for (_lbl, (_vv, _ii)), _col in zip(_cv_data.items(), _cv_colors2):
                _ax_cv2.plot(_vv, _ii, color=_col, linewidth=2, label=_lbl)
            _ax_cv2.set_xlabel("Potential (V)", fontsize=11)
            _ax_cv2.set_ylabel("Current (µA)", fontsize=11)
            _ax_cv2.set_title("CV — Bare GCE vs Modified Electrode", fontsize=12, fontweight="bold")
            _ax_cv2.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            _ax_cv2.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(_fig_cv2)
            plt.close(_fig_cv2)

        st.markdown("**Peak current comparison (manual entry):**")
        _pcc1, _pcc2 = st.columns(2)
        _pcc3, _pcc4 = st.columns(2)
        _ipa_bare = _pcc1.number_input("Ipa – Bare (µA)", value=0.0, format="%.3f", key="surf_ipa_bare",
                                        help="Anodic (oxidation) peak current of the unmodified electrode in the probe solution (e.g. [Fe(CN)₆]³⁻). Enter as a positive value.")
        _ipa_mod  = _pcc2.number_input("Ipa – Modified (µA)", value=0.0, format="%.3f", key="surf_ipa_mod",
                                        help="Anodic peak current after surface modification. Enhancement = Ipa(modified)/Ipa(bare) × 100%. Values > 100% indicate improved electron transfer kinetics.")
        _ipc_bare = _pcc3.number_input("Ipc – Bare (µA)", value=0.0, format="%.3f", key="surf_ipc_bare",
                                        help="Cathodic (reduction) peak current of the bare electrode. Enter as a positive value (absolute).")
        _ipc_mod  = _pcc4.number_input("Ipc – Modified (µA)", value=0.0, format="%.3f", key="surf_ipc_mod",
                                        help="Cathodic peak current after modification. The Ipc/Ipa ratio should be close to 1.0 for a reversible redox process.")
        if _ipa_bare > 0 and _ipa_mod > 0:
            _enhancement = (_ipa_mod - _ipa_bare) / _ipa_bare * 100
            _m1s, _m2s, _m3s = st.columns(3)
            _m1s.metric("Ipa Bare (µA)", f"{_ipa_bare:.3f}")
            _m2s.metric("Ipa Modified (µA)", f"{_ipa_mod:.3f}")
            _m3s.metric("Signal Enhancement", f"{_enhancement:+.1f}%",
                        delta=f"{_enhancement:+.1f}%",
                        delta_color="normal" if _enhancement > 0 else "inverse")
            if _enhancement > 0:
                st.success(f"✅ Modified electrode shows {_enhancement:.1f}% higher peak current → enhanced electrocatalytic activity")
            _metrics_csv_btn({
                "Ipa Bare (µA)": f"{_ipa_bare:.3f}",
                "Ipa Modified (µA)": f"{_ipa_mod:.3f}",
                "Ipc Bare (µA)": f"{_ipc_bare:.3f}",
                "Ipc Modified (µA)": f"{_ipc_mod:.3f}",
                "Signal Enhancement": f"{_enhancement:+.1f}%",
            }, "surface_cv_metrics.csv")

    # ── AI Interpretation ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🤖 AI Interpretation")
    st.caption("Generate a publication-ready interpretation of the surface characterization results.")

    # Mevcut metriklerden AI için özet dict oluştur
    _surf_ai_metrics = {}
    _rct_b_ai = st.session_state.get("rct_bare", 0.0)
    _rct_m_ai = st.session_state.get("rct_mod",  0.0)
    if _rct_b_ai > 0 and _rct_m_ai > 0:
        _surf_ai_metrics["Rct Bare (Ω)"]          = round(_rct_b_ai, 2)
        _surf_ai_metrics["Rct Modified (Ω)"]       = round(_rct_m_ai, 2)
        _surf_ai_metrics["Rct Ratio (Bare/Mod)"]   = round(_rct_b_ai / _rct_m_ai, 2)
    _ipa_b_ai = st.session_state.get("surf_ipa_bare", 0.0)
    _ipa_m_ai = st.session_state.get("surf_ipa_mod",  0.0)
    if _ipa_b_ai > 0 and _ipa_m_ai > 0:
        _surf_ai_metrics["Ipa Bare (µA)"]          = round(_ipa_b_ai, 3)
        _surf_ai_metrics["Ipa Modified (µA)"]      = round(_ipa_m_ai, 3)
        _surf_ai_metrics["Signal Enhancement"]     = f"{(_ipa_m_ai - _ipa_b_ai) / _ipa_b_ai * 100:+.1f}%"
    _sem_note_ai = st.session_state.get("sem_note", "")
    if _sem_note_ai:
        _surf_ai_metrics["SEM observation"] = _sem_note_ai
    # SEM görüntüsü yüklenmiş ama not girilmemişse yine de metriğe ekle
    _has_sem_bare = _sem_bare is not None
    _has_sem_mod  = _sem_mod  is not None
    if (_has_sem_bare or _has_sem_mod) and "SEM observation" not in _surf_ai_metrics:
        _sem_imgs = []
        if _has_sem_bare: _sem_imgs.append("bare electrode")
        if _has_sem_mod:  _sem_imgs.append("modified electrode")
        _surf_ai_metrics["SEM images available"] = ", ".join(_sem_imgs)
    # Flag: actual SEM image files are being sent — tell the prompt to do morphology analysis
    if _has_sem_bare or _has_sem_mod:
        _surf_ai_metrics["has_sem_images"] = True

    if not _surf_ai_metrics:
        st.info("💡 EIS veya CV verisi girin (Rct / Ipa alanlarını doldurun) ya da SEM gözlemini yazın — ardından AI yorumu üretebilirsiniz.")
    else:
        # Mevcut metrikleri göster
        with st.expander("📊 Metrics sent to AI", expanded=False):
            for _k, _v in _surf_ai_metrics.items():
                st.write(f"**{_k}:** {_v}")
        _surf_sem_files = [img for img in [_sem_bare, _sem_mod] if img is not None]
        if _surf_sem_files:
            st.caption(f"🖼️ {len(_surf_sem_files)} SEM görüntüsü AI'ya gönderilecek")
        _ai_interpret_btn(
            "surf",
            interpret_surface,
            {"metrics":        _surf_ai_metrics,
             "electrode_name": st.session_state.get("surf_electrode", "Modified Electrode")},
            result_key="surf_interp",
            btn_key="ai_surf_inline",
            images=_surf_sem_files,
        )
        if st.session_state.get("surf_interp"):
            st.markdown("#### 📝 AI Commentary")
            st.markdown(st.session_state["surf_interp"])

# ═══════════════════════════════════════════════════════════
# ⚡  DPV — SENSOR CALIBRATION
# ═══════════════════════════════════════════════════════════
elif page == "⚡  DPV — Sensor Calibration":
    st.title("⚡ DPV Sensor Calibration")
    st.caption(f"Project: **{_cur().get('name','')}**")
    st.markdown(
        "Upload one DPV file per concentration. The app auto-detects the peak current "
        "(Ipa) in your chosen potential window, fits a linear calibration curve, and "
        "calculates LOD, LOQ, and sensitivity."
    )

    # ── Helper: parse a DPV CSV ─────────────────────────────
    def _parse_dpv_file(f) -> tuple:
        """Return (potential_array, current_uA_array) or raise."""
        import io
        f.seek(0)
        raw = f.read().decode("utf-8", errors="replace")
        f.seek(0)
        import csv
        dialect = csv.Sniffer().sniff(raw[:2048], delimiters=",;\t ")
        reader = csv.DictReader(io.StringIO(raw), dialect=dialect)
        rows = list(reader)
        if not rows:
            raise ValueError("Empty file")
        hdr = [h.strip() for h in rows[0].keys()]
        # Find potential column
        pot_col = next((h for h in hdr if any(k in h.lower() for k in
                        ("potential","e_","e /","e(v)","volt","pot"))), hdr[0])
        # Find current column
        cur_col = next((h for h in hdr if any(k in h.lower() for k in
                        ("current","i_","i /","curr","i(","ia","uA","µa","μa"))), hdr[1])
        import numpy as np
        pot = np.array([float(r[pot_col]) for r in rows if r[pot_col].strip()])
        cur = np.array([float(r[cur_col]) for r in rows if r[cur_col].strip()])
        # Convert A → µA if values look like they're in Amperes
        if np.max(np.abs(cur)) < 0.01:
            cur = cur * 1e6
        return pot, cur

    def _find_peak(pot, cur, e_min, e_max, peak_type="anodic"):
        """Return peak current in the given potential window."""
        import numpy as np
        mask = (pot >= e_min) & (pot <= e_max)
        if not np.any(mask):
            raise ValueError(f"No data in window {e_min}–{e_max} V")
        cur_w = cur[mask]
        return float(np.max(cur_w)) if peak_type == "anodic" else float(np.min(cur_w))

    # ── Settings row ────────────────────────────────────────
    _dset1, _dset2, _dset3, _dset4 = st.columns([1,1,1,1])
    _dpv_emin   = _dset1.number_input("Peak window E min (V)", value=0.0,  format="%.3f", key="dpv_emin")
    _dpv_emax   = _dset2.number_input("Peak window E max (V)", value=0.8,  format="%.3f", key="dpv_emax")
    _dpv_ptype  = _dset3.selectbox("Peak type", ["anodic (max)", "cathodic (min)"], key="dpv_ptype")
    _CONC_UNITS = ["µM","nM","mM","mg/L","µg/L","ng/mL","mol/L"]
    _dpv_unit   = _dset4.selectbox("Concentration unit", _CONC_UNITS, key="dpv_unit")
    _peak_type  = "anodic" if "anodic" in _dpv_ptype else "cathodic"

    _dpv_analyte = st.text_input("Analyte name", value=st.session_state.get("dpv_analyte",""), key="dpv_analyte")

    # ── ECSA from CV (optional) for sensitivity per cm² ────
    _dpv_ecsa = None
    _cv_saved = _cur().get("analyses", {}).get("cv", {}).get("metrics", {})
    if _cv_saved.get("ecsa") or _cv_saved.get("ECSA_cm2"):
        _dpv_ecsa = float(_cv_saved.get("ecsa") or _cv_saved.get("ECSA_cm2") or 0) or None
        if _dpv_ecsa:
            st.info(f"🔗 ECSA from CV module: **{_dpv_ecsa:.4f} cm²** — will be used for normalised sensitivity.")

    st.markdown("---")

    # ── File uploader ────────────────────────────────────────
    st.markdown("#### 1. Upload DPV files (one per concentration)")
    _dpv_files = st.file_uploader(
        "Select all DPV CSV files at once",
        type=["csv","txt","dat"],
        accept_multiple_files=True,
        key="dpv_files_up",
    )

    # ── Concentration labels ─────────────────────────────────
    _dpv_concs: list[float] = []
    _dpv_ipa:   list[float] = []
    _dpv_names: list[str]   = []

    if _dpv_files:
        st.markdown("#### 2. Assign concentrations & verify peaks")
        _n = len(_dpv_files)
        _cols_hdr = st.columns([3, 2, 2])
        _cols_hdr[0].markdown("**File**")
        _cols_hdr[1].markdown(f"**Concentration ({_dpv_unit})**")
        _cols_hdr[2].markdown("**Detected Ipa (µA)**")

        _parse_errors = []
        for _idx, _fobj in enumerate(_dpv_files):
            _fc1, _fc2, _fc3 = st.columns([3, 2, 2])
            _fc1.caption(_fobj.name)
            _conc_val = _fc2.number_input(
                f"conc_{_idx}", label_visibility="collapsed",
                min_value=0.0, value=0.0, format="%.4f",
                key=f"dpv_conc_{_idx}",
            )
            try:
                _pot, _cur = _parse_dpv_file(_fobj)
                _ipa = _find_peak(_pot, _cur, _dpv_emin, _dpv_emax, _peak_type)
                _fc3.metric("", f"{_ipa:.3f} µA")
                _dpv_concs.append(_conc_val)
                _dpv_ipa.append(_ipa)
                _dpv_names.append(_fobj.name)
            except Exception as _pe:
                _fc3.error(f"⚠ {_pe}")
                _parse_errors.append(str(_pe))

    # ── Manual table fallback ────────────────────────────────
    with st.expander("📝 Or enter data manually (no files needed)", expanded=not _dpv_files):
        st.markdown(f"Enter concentration ({_dpv_unit}) and peak current (µA) pairs:")
        _n_rows = st.number_input("Number of points", min_value=3, max_value=30, value=6, key="dpv_nrows_manual")
        _man_concs, _man_ipa = [], []
        _mcols = st.columns(2)
        for _mi in range(int(_n_rows)):
            _mc = _mcols[0].number_input(f"C{_mi+1} ({_dpv_unit})", value=0.0, min_value=0.0,
                                          format="%.4f", key=f"dpv_mc_{_mi}")
            _mi2 = _mcols[1].number_input(f"Ipa{_mi+1} (µA)", value=0.0,
                                           format="%.4f", key=f"dpv_mi_{_mi}")
            if _mc > 0 and _mi2 != 0.0:
                _man_concs.append(_mc)
                _man_ipa.append(_mi2)
        if _man_concs and not _dpv_files:
            _dpv_concs = _man_concs
            _dpv_ipa   = _man_ipa
            _dpv_names = [f"Point {i+1}" for i in range(len(_man_concs))]

    # ── Calibration calculation ──────────────────────────────
    _can_run = len(_dpv_concs) >= 3 and all(c > 0 for c in _dpv_concs)
    if _can_run:
        st.markdown("---")
        st.markdown("#### 3. Calibration curve & results")
        import numpy as np
        from scipy import stats as _stats
        _x = np.array(_dpv_concs)
        _y = np.array(_dpv_ipa)
        _sort_idx = np.argsort(_x)
        _x, _y = _x[_sort_idx], _y[_sort_idx]

        _slope, _intercept, _r, _pval, _stderr = _stats.linregress(_x, _y)
        _r2 = _r ** 2
        _residuals = _y - (_slope * _x + _intercept)
        _sigma = float(np.std(_residuals, ddof=2)) if len(_residuals) > 2 else float(np.std(_residuals))

        _lod = 3.0 * _sigma / abs(_slope)
        _loq = 10.0 * _sigma / abs(_slope)
        _linear_range_low  = float(_x.min())
        _linear_range_high = float(_x.max())
        _sensitivity_per_cm2 = (abs(_slope) / _dpv_ecsa) if _dpv_ecsa else None

        # Plot
        import plotly.graph_objects as go
        _fig_dpv = go.Figure()
        _x_fit = np.linspace(_x.min(), _x.max(), 200)
        _y_fit = _slope * _x_fit + _intercept
        _fig_dpv.add_trace(go.Scatter(x=_x, y=_y, mode="markers",
                                       marker=dict(size=9, color="#2563EB"),
                                       name="Measured Ipa"))
        _fig_dpv.add_trace(go.Scatter(x=_x_fit, y=_y_fit, mode="lines",
                                       line=dict(color="#EF4444", width=2),
                                       name=f"Linear fit (R²={_r2:.5f})"))
        _fig_dpv.update_layout(
            xaxis_title=f"Concentration ({_dpv_unit})",
            yaxis_title="Ipa (µA)",
            title=f"DPV Calibration — {_dpv_analyte or 'Analyte'}",
            height=400,
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=60, r=20, t=50, b=50),
        )
        st.plotly_chart(_fig_dpv, use_container_width=True)

        # Results table
        _res_col1, _res_col2 = st.columns(2)
        with _res_col1:
            st.markdown("**Regression**")
            st.markdown(f"- Slope: **{_slope:.4f} µA/{_dpv_unit}**")
            st.markdown(f"- Intercept: **{_intercept:.4f} µA**")
            st.markdown(f"- R²: **{_r2:.5f}**")
            st.markdown(f"- Linear range: **{_linear_range_low:.4g}–{_linear_range_high:.4g} {_dpv_unit}**")
        with _res_col2:
            st.markdown("**Figures of merit**")
            st.markdown(f"- LOD: **{_lod:.4g} {_dpv_unit}**")
            st.markdown(f"- LOQ: **{_loq:.4g} {_dpv_unit}**")
            if _sensitivity_per_cm2:
                st.markdown(f"- Sensitivity: **{abs(_slope):.4f} µA/{_dpv_unit}**")
                st.markdown(f"- Norm. sensitivity: **{_sensitivity_per_cm2:.4f} µA/{_dpv_unit}·cm²**")
            else:
                st.markdown(f"- Sensitivity: **{abs(_slope):.4f} µA/{_dpv_unit}**")
                st.caption("Add electrode area or run CV first for normalised sensitivity.")

        # ── Mandel linearity test (Lack-of-Fit F-test) ──────
        if len(_x) >= 5:
            # Fit quadratic
            _p2 = np.polyfit(_x, _y, 2)
            _y_lin  = _slope * _x + _intercept
            _y_quad = np.polyval(_p2, _x)
            _rss_lin  = float(np.sum((_y - _y_lin)  ** 2))
            _rss_quad = float(np.sum((_y - _y_quad) ** 2))
            _n_pts = len(_x)
            # F = (RSS_lin - RSS_quad) / 1  /  (RSS_quad / (n-3))
            if _rss_quad > 0 and _n_pts > 3:
                _F_mandel = (_rss_lin - _rss_quad) / 1.0 / (_rss_quad / (_n_pts - 3))
                from scipy.stats import f as _f_dist
                _p_mandel = 1 - _f_dist.cdf(_F_mandel, 1, _n_pts - 3)
                st.markdown("**Mandel Linearity Test (Lack-of-Fit F-test)**")
                _ml1, _ml2, _ml3 = st.columns(3)
                _ml1.metric("F statistic", f"{_F_mandel:.3f}")
                _ml2.metric("p-value", f"{_p_mandel:.4f}")
                _ml3.metric("Linearity", "✅ Confirmed" if _p_mandel >= 0.05 else "⚠️ Rejected")
                if _p_mandel < 0.05:
                    st.warning(f"⚠️ Mandel test: F = {_F_mandel:.3f}, p = {_p_mandel:.4f} < 0.05 — "
                               "the quadratic term significantly improves the fit. Consider narrowing the "
                               "linear range or reporting a quadratic calibration equation.")
                else:
                    st.success(f"✅ Mandel test: F = {_F_mandel:.3f}, p = {_p_mandel:.4f} ≥ 0.05 — "
                               "linear model is adequate over the stated concentration range.")
                # Store for metrics
                _mandel_verdict = "confirmed" if _p_mandel >= 0.05 else "rejected"
            else:
                _F_mandel, _p_mandel, _mandel_verdict = None, None, "not tested"
        else:
            st.caption("💡 Mandel linearity test requires ≥ 5 calibration points.")
            _F_mandel, _p_mandel, _mandel_verdict = None, None, "not tested (< 5 pts)"

        # Save to project
        _dpv_metrics = {
            "analyte":        _dpv_analyte,
            "unit":           _dpv_unit,
            "slope":          round(_slope, 6),
            "intercept":      round(_intercept, 6),
            "R2":             round(_r2, 6),
            "LOD":            round(_lod, 6),
            "LOQ":            round(_loq, 6),
            "linear_range":   f"{_linear_range_low:.4g}–{_linear_range_high:.4g}",
            "sensitivity":    round(abs(_slope), 6),
            "n_points":       len(_x),
            "peak_window_V":  f"{_dpv_emin}–{_dpv_emax}",
        }
        if _sensitivity_per_cm2:
            _dpv_metrics["sensitivity_norm"] = round(_sensitivity_per_cm2, 6)
        if _F_mandel is not None:
            _dpv_metrics["Mandel F"] = round(_F_mandel, 4)
            _dpv_metrics["Mandel p"] = round(_p_mandel, 4)
            _dpv_metrics["Linearity"] = _mandel_verdict

        _save_analysis("dpv", _dpv_metrics, None)

        # ── AI interpretation ────────────────────────────────
        st.markdown("---")
        st.markdown("#### 4. AI Interpretation")
        _ai_interpret_btn(
            "dpv",
            interpret_dpv_calibration,
            {"metrics": _dpv_metrics},
            result_key="dpv_interp",
            btn_key="ai_dpv_inline",
        )
        if st.session_state.get("dpv_interp"):
            st.markdown(st.session_state["dpv_interp"])

    elif _dpv_files or _dpv_concs:
        st.info("💡 Assign concentrations to all files (must be > 0) and ensure at least 3 points.")
    else:
        st.info("💡 Upload DPV CSV files above or enter data manually to begin.")


# ═══════════════════════════════════════════════════════════
# 🔧  DPV — OPTIMIZATION (OFAT)
# ═══════════════════════════════════════════════════════════
elif page == "🔧  DPV — Optimization":
    st.title("🔧 DPV Parameter Optimization")
    st.caption(f"Project: **{_cur().get('name','')}**")
    st.markdown(
        "One-factor-at-a-time (OFAT) optimization of DPV parameters. "
        "For each parameter, record the peak current (Ipa) at each tested value — "
        "the app marks the optimum and builds a summary table for the Methods section."
    )

    # ── Predefined parameters ───────────────────────────────
    _OFAT_PARAMS = {
        "Pulse Amplitude (mV)":      {"unit": "mV",   "hint": "10, 20, 30, 40, 50, 60, 70, 80 mV"},
        "Pulse Width (ms)":          {"unit": "ms",   "hint": "10, 20, 30, 40, 50, 60 ms"},
        "Step Potential (mV)":       {"unit": "mV",   "hint": "2, 4, 6, 8, 10 mV"},
        "Accumulation Time (s)":     {"unit": "s",    "hint": "0, 30, 60, 90, 120, 150 s"},
        "Accumulation Potential (V)":{"unit": "V",    "hint": "-0.4, -0.2, 0.0, 0.2 V"},
        "Modifier Loading (µL)":     {"unit": "µL",   "hint": "2, 4, 6, 8, 10 µL"},
        "Modifier Concentration (mg/mL)": {"unit": "mg/mL", "hint": "0.5, 1.0, 1.5, 2.0 mg/mL"},
        "Electrolyte Concentration (mM)": {"unit": "mM",   "hint": "10, 25, 50, 100, 200 mM"},
        "Scan Rate (mV/s)":          {"unit": "mV/s", "hint": "10, 25, 50, 75, 100 mV/s"},
        "Custom parameter…":         {"unit": "",     "hint": ""},
    }

    # ── Load saved optimizations from project ────────────────
    _opt_saved = _cur().get("analyses", {}).get("dpvopt", {}).get("params", {})

    # ── Add new parameter scan ───────────────────────────────
    st.markdown("#### Add an optimization run")
    _opt_sel = st.selectbox("Parameter to optimize", list(_OFAT_PARAMS.keys()), key="opt_param_sel")
    if _opt_sel == "Custom parameter…":
        _cust_col1, _cust_col2 = st.columns(2)
        _opt_name = _cust_col1.text_input("Parameter name", key="opt_custom_name",
                                           placeholder="e.g. Incubation time (min)")
        _opt_unit = _cust_col2.text_input("Unit", key="opt_custom_unit", placeholder="min")
    else:
        _opt_name = _opt_sel
        _opt_unit = _OFAT_PARAMS[_opt_sel]["unit"]

    _opt_hint = _OFAT_PARAMS.get(_opt_sel, {}).get("hint", "")
    if _opt_hint:
        st.caption(f"Typical range: {_opt_hint}")

    _opt_n = st.number_input("Number of tested values", min_value=2, max_value=20,
                              value=6, key="opt_n_vals")

    _ov_cols = st.columns(2)
    _ov_cols[0].markdown(f"**{_opt_name or 'Value'} ({_opt_unit})**")
    _ov_cols[1].markdown("**Ipa (µA)**")

    _opt_vals, _opt_ipa = [], []
    for _oi in range(int(_opt_n)):
        _oc1, _oc2 = st.columns(2)
        _v = _oc1.number_input(f"val_{_oi}", value=0.0, format="%.4f",
                                label_visibility="collapsed", key=f"opt_v_{_oi}")
        _i = _oc2.number_input(f"ipa_{_oi}", value=0.0, format="%.4f",
                                label_visibility="collapsed", key=f"opt_i_{_oi}")
        if _v != 0.0 or _i != 0.0:
            _opt_vals.append(_v)
            _opt_ipa.append(_i)

    _can_opt = len(_opt_vals) >= 2 and _opt_name and any(i != 0.0 for i in _opt_ipa)

    if _can_opt:
        import numpy as np
        _ov = np.array(_opt_vals)
        _oi_arr = np.array(_opt_ipa)
        _sort_i = np.argsort(_ov)
        _ov, _oi_arr = _ov[_sort_i], _oi_arr[_sort_i]

        _best_idx = int(np.argmax(np.abs(_oi_arr)))
        _opt_value = float(_ov[_best_idx])
        _opt_ipa_best = float(_oi_arr[_best_idx])

        import plotly.graph_objects as go
        _fig_opt = go.Figure()
        _fig_opt.add_trace(go.Scatter(
            x=list(_ov), y=list(_oi_arr),
            mode="lines+markers",
            marker=dict(size=8, color="#2563EB"),
            line=dict(color="#2563EB", width=2),
            name="Ipa",
        ))
        _fig_opt.add_trace(go.Scatter(
            x=[_opt_value], y=[_opt_ipa_best],
            mode="markers",
            marker=dict(size=14, color="#EF4444", symbol="star"),
            name=f"Optimum: {_opt_value} {_opt_unit}",
        ))
        _fig_opt.update_layout(
            xaxis_title=f"{_opt_name} ({_opt_unit})",
            yaxis_title="Ipa (µA)",
            title=f"Optimization: {_opt_name}",
            height=380,
            margin=dict(l=60, r=20, t=50, b=50),
        )
        st.plotly_chart(_fig_opt, use_container_width=True)
        st.success(f"⭐ Optimum: **{_opt_value} {_opt_unit}** → Ipa = **{_opt_ipa_best:.3f} µA**")

        if st.button("💾 Save this optimization run", key="btn_save_opt"):
            _existing = _cur().get("analyses", {}).get("dpvopt", {})
            _existing_params = _existing.get("params", {})
            _existing_params[_opt_name] = {
                "unit":        _opt_unit,
                "values":      list(_ov),
                "ipa":         list(_oi_arr),
                "optimum":     _opt_value,
                "optimum_ipa": _opt_ipa_best,
            }
            _all_metrics = {k: v for k, v in _existing.get("metrics", {}).items()}
            _all_metrics[_opt_name] = f"{_opt_value} {_opt_unit}"
            _save_analysis("dpvopt", _all_metrics, None)
            # Also update params dict manually
            _proj_data = st.session_state.get("projects", [])
            for _p in _proj_data:
                if _p.get("name") == _cur().get("name"):
                    if "analyses" not in _p: _p["analyses"] = {}
                    if "dpvopt" not in _p["analyses"]: _p["analyses"]["dpvopt"] = {}
                    _p["analyses"]["dpvopt"]["params"] = _existing_params
            st.success(f"✅ Saved: {_opt_name} = {_opt_value} {_opt_unit}")
            st.rerun()

    # ── Summary of all saved optimizations ──────────────────
    _opt_saved = _cur().get("analyses", {}).get("dpvopt", {}).get("params", {})
    if _opt_saved:
        st.markdown("---")
        st.markdown("#### Optimized parameters summary")
        import pandas as pd
        _sum_rows = []
        _opt_figs = []
        for _pname, _pdata in _opt_saved.items():
            _sum_rows.append({
                "Parameter":      _pname,
                "Optimum value":  f"{_pdata['optimum']} {_pdata['unit']}",
                "Ipa at optimum": f"{_pdata['optimum_ipa']:.3f} µA",
            })
        st.dataframe(pd.DataFrame(_sum_rows), use_container_width=True, hide_index=True)

        # Multi-panel figure
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots as _msp
        _np2 = len(_opt_saved)
        _ncols_fig = min(3, _np2)
        _nrows_fig = (_np2 + _ncols_fig - 1) // _ncols_fig
        _fig_all = _msp(rows=_nrows_fig, cols=_ncols_fig,
                        subplot_titles=list(_opt_saved.keys()))
        for _pi, (_pname, _pdata) in enumerate(_opt_saved.items()):
            _r = _pi // _ncols_fig + 1
            _c = _pi % _ncols_fig + 1
            _fig_all.add_trace(go.Scatter(
                x=_pdata["values"], y=_pdata["ipa"],
                mode="lines+markers",
                marker=dict(size=7, color="#2563EB"),
                line=dict(color="#2563EB", width=1.5),
                showlegend=False,
            ), row=_r, col=_c)
            _fig_all.add_trace(go.Scatter(
                x=[_pdata["optimum"]], y=[_pdata["optimum_ipa"]],
                mode="markers",
                marker=dict(size=12, color="#EF4444", symbol="star"),
                showlegend=False,
            ), row=_r, col=_c)
        _fig_all.update_layout(height=300 * _nrows_fig,
                                title_text="DPV Parameter Optimization Overview",
                                margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(_fig_all, use_container_width=True)

        # AI interpretation
        st.markdown("---")
        st.markdown("#### AI Interpretation")
        _opt_metrics = {
            "optimized_parameters": {k: f"{v['optimum']} {v['unit']}" for k, v in _opt_saved.items()},
            "analyte": _cur().get("analyses", {}).get("dpv", {}).get("metrics", {}).get("analyte", "analyte"),
        }
        _ai_interpret_btn(
            "dpvopt",
            interpret_dpv_optimization,
            {"metrics": _opt_metrics},
            result_key="dpvopt_interp",
            btn_key="ai_dpvopt_inline",
        )
        if st.session_state.get("dpvopt_interp"):
            st.markdown(st.session_state["dpvopt_interp"])
    elif not _can_opt:
        st.info("💡 Enter values and Ipa measurements above, then save each optimization run. "
                "A summary table with all parameters will appear here.")


# ═══════════════════════════════════════════════════════════
# 🧫  REAL SAMPLE ANALYSIS
# ═══════════════════════════════════════════════════════════
elif page == "🧫  Real Sample Analysis":
    st.title("🧫 Real Sample Analysis")
    st.caption(f"Project: **{_cur().get('name','')}**")
    st.markdown(
        "Enter the known (spiked) and found (measured from calibration) concentrations "
        "for each sample to calculate recovery % and RSD."
    )

    # Pull LOD/LOQ/unit from DPV module if available
    _dpv_saved = _cur().get("analyses", {}).get("dpv", {}).get("metrics", {})
    _rs_unit   = _dpv_saved.get("unit", "µM")
    _rs_slope  = _dpv_saved.get("slope")
    _rs_int    = _dpv_saved.get("intercept")
    if _dpv_saved:
        st.info(f"🔗 Calibration from DPV module: slope={_rs_slope:.4f} µA/{_rs_unit}, "
                f"intercept={_rs_int:.4f} µA — concentrations will be back-calculated from Ipa if needed.")

    # Settings
    _rsc1, _rsc2, _rsc3 = st.columns(3)
    _rs_analyte = _rsc1.text_input("Analyte", value=_dpv_saved.get("analyte",""), key="rs_analyte")
    _UNITS2 = ["µM","nM","mM","mg/L","µg/L","ng/mL","mol/L"]
    _rs_unit_sel = _rsc2.selectbox("Unit", _UNITS2,
                                    index=_UNITS2.index(_rs_unit) if _rs_unit in _UNITS2 else 0,
                                    key="rs_unit_sel")
    _MATRIX_TYPES = ["Urine","Blood serum","Tap water","River water","Food extract",
                     "Pharmaceutical","Soil extract","Other"]
    _rs_matrix = _rsc3.selectbox("Sample matrix", _MATRIX_TYPES, key="rs_matrix")

    st.markdown("---")
    st.markdown("#### Sample data entry")

    _n_samples = st.number_input("Number of samples", min_value=1, max_value=20, value=3, key="rs_n")
    _input_mode = st.radio("Input mode", ["Concentration (direct)", "Current → back-calculate"],
                            horizontal=True, key="rs_input_mode")

    _rs_known, _rs_found, _rs_labels = [], [], []
    _hdr1, _hdr2, _hdr3 = st.columns([2, 2, 2])
    _hdr1.markdown(f"**Sample label**")
    _hdr2.markdown(f"**Spiked / Known ({_rs_unit_sel})**")
    _hdr3.markdown(f"**{'Found (' + _rs_unit_sel + ')' if _input_mode.startswith('Conc') else 'Measured Ipa (µA)'}**")

    for _si in range(int(_n_samples)):
        _sc1, _sc2, _sc3 = st.columns([2, 2, 2])
        _lbl = _sc1.text_input(f"Label {_si+1}", value=f"S{_si+1}", key=f"rs_lbl_{_si}",
                                label_visibility="collapsed")
        _known = _sc2.number_input(f"Known {_si+1}", min_value=0.0, value=0.0,
                                    format="%.4f", key=f"rs_known_{_si}",
                                    label_visibility="collapsed")
        _meas = _sc3.number_input(f"Found {_si+1}", value=0.0, format="%.4f",
                                   key=f"rs_meas_{_si}", label_visibility="collapsed")
        if _known > 0 and _meas != 0.0:
            _rs_known.append(_known)
            _rs_labels.append(_lbl)
            if _input_mode.startswith("Current") and _rs_slope and abs(_rs_slope) > 0:
                _found_c = (_meas - _rs_int) / _rs_slope
                _rs_found.append(max(0.0, _found_c))
            else:
                _rs_found.append(_meas)

    if len(_rs_known) >= 1:
        import numpy as np
        _recoveries = [(_rs_found[i] / _rs_known[i]) * 100 for i in range(len(_rs_known))]
        _rec_arr = np.array(_recoveries)
        _mean_rec = float(np.mean(_rec_arr))
        _rsd      = float(np.std(_rec_arr, ddof=1) / np.mean(_rec_arr) * 100) if len(_rec_arr) > 1 else 0.0

        st.markdown("---")
        st.markdown("#### Results")

        import pandas as pd
        _rs_df = pd.DataFrame({
            "Sample":        _rs_labels,
            f"Added ({_rs_unit_sel})": _rs_known,
            f"Found ({_rs_unit_sel})": [round(v, 4) for v in _rs_found],
            "Recovery (%)": [round(r, 2) for r in _recoveries],
        })
        st.dataframe(_rs_df, use_container_width=True, hide_index=True)

        _r1, _r2, _r3 = st.columns(3)
        _r1.metric("Mean Recovery", f"{_mean_rec:.2f}%")
        if len(_rec_arr) > 1:
            _r2.metric("RSD", f"{_rsd:.2f}%")
        _r3.metric("n samples", len(_rs_known))

        # Acceptability flag
        if 95 <= _mean_rec <= 105 and _rsd <= 5:
            st.success("✅ Acceptable: recovery 95–105 % and RSD ≤ 5 % — meets typical analytical criteria.")
        elif 90 <= _mean_rec <= 110 and _rsd <= 10:
            st.warning("⚠️ Marginal: recovery within 90–110 % but criteria are relaxed — verify method selectivity.")
        else:
            st.error("❌ Outside acceptable range — check calibration, matrix effects, or extraction efficiency.")

        # Save
        _rs_metrics = {
            "analyte":     _rs_analyte,
            "matrix":      _rs_matrix,
            "unit":        _rs_unit_sel,
            "mean_recovery_%": round(_mean_rec, 3),
            "RSD_%":       round(_rsd, 3) if len(_rec_arr) > 1 else "N/A",
            "n_samples":   len(_rs_known),
            "samples":     _rs_labels,
            "recoveries":  [round(r, 2) for r in _recoveries],
        }
        _save_analysis("realsample", _rs_metrics, None)

        # AI
        st.markdown("---")
        st.markdown("#### AI Interpretation")
        _ai_interpret_btn(
            "realsample",
            interpret_real_sample,
            {"metrics": _rs_metrics},
            result_key="rs_interp",
            btn_key="ai_rs_inline",
        )
        if st.session_state.get("rs_interp"):
            st.markdown(st.session_state["rs_interp"])
    else:
        st.info("💡 Fill at least one row (spiked and found concentrations both > 0).")


# ═══════════════════════════════════════════════════════════
# 📚  LITERATURE COMPARISON
# ═══════════════════════════════════════════════════════════
elif page == "📚  Literature Comparison":
    st.title("📚 Literature Comparison Table")
    st.caption(f"Project: **{_cur().get('name','')}**")
    st.markdown(
        "Build the standard 'comparison with previously reported sensors' table required "
        "by most electroanalytical journals. Add rows manually — Scopus/WoS integration coming soon."
    )

    # Pull own sensor data
    _dpv_s = _cur().get("analyses", {}).get("dpv", {}).get("metrics", {})
    _lit_analyte = _dpv_s.get("analyte", "")
    _lit_unit    = _dpv_s.get("unit", "µM")
    _lit_own_lod = _dpv_s.get("LOD")
    _lit_own_lr  = _dpv_s.get("linear_range")
    _lit_own_sens= _dpv_s.get("sensitivity")

    if _dpv_s:
        st.info(
            f"🔗 Your sensor — Analyte: **{_lit_analyte}** | "
            f"LOD: **{_lit_own_lod:.4g} {_lit_unit}** | "
            f"Linear range: **{_lit_own_lr} {_lit_unit}** | "
            f"Sensitivity: **{_lit_own_sens:.4f} µA/{_lit_unit}**"
        )

    st.markdown("---")
    st.markdown("#### Add literature entries")

    _lit_n = st.number_input("Number of literature rows", min_value=1, max_value=20, value=5, key="lit_n")

    # Header
    _lhdr = st.columns([2, 2, 1.5, 1.5, 1.5, 1.5])
    for _h, _t in zip(_lhdr, ["Electrode / Modifier", "Method", f"LOD ({_lit_unit})",
                                f"Linear range ({_lit_unit})", f"Sensitivity (µA/{_lit_unit})", "Reference"]):
        _h.markdown(f"**{_t}**")

    _lit_rows = []
    for _li in range(int(_lit_n)):
        _lc = st.columns([2, 2, 1.5, 1.5, 1.5, 1.5])
        _electrode = _lc[0].text_input(f"Electrode {_li+1}", value="", key=f"lit_elec_{_li}",
                                        label_visibility="collapsed", placeholder="e.g. GCE/MoS₂")
        _method    = _lc[1].text_input(f"Method {_li+1}", value="", key=f"lit_meth_{_li}",
                                        label_visibility="collapsed", placeholder="e.g. DPV")
        _lod_v     = _lc[2].text_input(f"LOD {_li+1}", value="", key=f"lit_lod_{_li}",
                                        label_visibility="collapsed", placeholder="0.01")
        _lr_v      = _lc[3].text_input(f"LR {_li+1}", value="", key=f"lit_lr_{_li}",
                                        label_visibility="collapsed", placeholder="0.1–100")
        _sens_v    = _lc[4].text_input(f"Sens {_li+1}", value="", key=f"lit_sens_{_li}",
                                        label_visibility="collapsed", placeholder="1.23")
        _ref_v     = _lc[5].text_input(f"Ref {_li+1}", value="", key=f"lit_ref_{_li}",
                                        label_visibility="collapsed", placeholder="Author, Year")
        if _electrode.strip():
            _lit_rows.append({
                "Electrode/Modifier": _electrode,
                "Method": _method,
                f"LOD ({_lit_unit})": _lod_v,
                f"Linear range ({_lit_unit})": _lr_v,
                f"Sensitivity": _sens_v,
                "Reference": _ref_v,
            })

    # Always append own sensor as last row
    if _dpv_s:
        _lit_rows.append({
            "Electrode/Modifier": f"**This work** ({_cur().get('name','')})",
            "Method": "DPV",
            f"LOD ({_lit_unit})": f"{_lit_own_lod:.4g}" if _lit_own_lod else "",
            f"Linear range ({_lit_unit})": _lit_own_lr or "",
            f"Sensitivity": f"{_lit_own_sens:.4f}" if _lit_own_sens else "",
            "Reference": "This work",
        })

    if _lit_rows:
        st.markdown("---")
        st.markdown("#### Comparison table")
        import pandas as pd
        _lit_df = pd.DataFrame(_lit_rows)
        st.dataframe(_lit_df, use_container_width=True, hide_index=True)

        # Save
        _save_analysis("litcomp", {
            "analyte":    _lit_analyte,
            "unit":       _lit_unit,
            "n_entries":  len(_lit_rows),
            "entries":    _lit_rows,
        }, None)

        # AI summary
        st.markdown("---")
        st.markdown("#### AI comparison summary")
        _lit_metrics = {
            "analyte": _lit_analyte,
            "unit":    _lit_unit,
            "own_LOD": _lit_own_lod,
            "own_linear_range": _lit_own_lr,
            "own_sensitivity": _lit_own_sens,
            "literature": _lit_rows[:-1] if _dpv_s else _lit_rows,
        }
        _ai_interpret_btn(
            "litcomp",
            interpret_literature_comparison,
            {"metrics": _lit_metrics},
            result_key="lit_interp",
            btn_key="ai_lit_inline",
        )
        if st.session_state.get("lit_interp"):
            st.markdown(st.session_state["lit_interp"])
    else:
        st.info("💡 Fill at least one electrode/modifier field to build the table.")


# ═══════════════════════════════════════════════════════════
# 📄  REPORT
# ═══════════════════════════════════════════════════════════
elif page == "📄  Report":
    st.title("📄 Generate Publication Report")
    proj = _cur()
    analyses = proj.get("analyses", {})

    st.markdown(f"### Project: **{proj.get('name','')}**")
    if proj.get("notes"):
        st.caption(proj["notes"])

    if not analyses:
        st.warning("No analyses found in this project. Run at least one module first.")
        st.stop()

    # ── Durum tablosu ────────────────────────────────────────
    _mod_labels = {"cv":"📈 CV","eis":"🔵 EIS","ph":"🧪 pH",
                   "cal":"📊 Calibration","sa":"➕ Std Addition",
                   "sel":"🧬 Selectivity","stab":"📅 Stability","surf":"🔬 Surface Charact.",
                   "dpv":"⚡ DPV Calibration","dpvopt":"🔧 DPV Optimization",
                   "realsample":"🧫 Real Sample","litcomp":"📚 Literature"}
    st.markdown("**Analyses ready for report:**")
    _missing_interp = []
    for k, a in analyses.items():
        interp = a.get("interpretation","")
        _note_badge = " 📝" if a.get("notes","") else ""
        if interp:
            st.markdown(f"- 🤖 {_mod_labels.get(k,k)} — {a.get('timestamp','')} ✅ *AI interpretation available*{_note_badge}")
        else:
            st.markdown(f"- 📊 {_mod_labels.get(k,k)} — {a.get('timestamp','')} ⚪ *No AI interpretation*{_note_badge}")
            _missing_interp.append(k)
        if a.get("notes",""):
            st.caption(f"  📝 _{a['notes'][:120]}{'…' if len(a.get('notes','')) > 120 else ''}_")

    # ── session-state only modules (sel / stab / surf) ───────────────────────
    _ss_checks = [
        ("sel",  "sel_results",  "sel_interp"),
        ("stab", "stab_results", "stab_interp"),
        ("surf", "rct_bare",     "surf_interp"),   # rct_bare > 0 → data present
    ]
    for _mk, _data_key, _interp_key in _ss_checks:
        if _mk in analyses:
            continue   # already shown above (saved via _save_analysis)
        # Determine if data is present
        _has_data = False
        if _mk == "surf":
            _has_data = (st.session_state.get("rct_bare", 0) > 0
                         or st.session_state.get("surf_ipa_bare", 0) > 0
                         or bool(st.session_state.get("sem_note", ""))
                         or st.session_state.get("has_sem_bare", False)
                         or st.session_state.get("has_sem_mod",  False))
        else:
            _has_data = bool(st.session_state.get(_data_key))
        if not _has_data:
            continue
        _has_interp = bool(st.session_state.get(_interp_key, ""))
        if _has_interp:
            st.markdown(f"- 🤖 {_mod_labels.get(_mk,_mk)} ✅ *AI interpretation available*")
        else:
            st.markdown(f"- 📊 {_mod_labels.get(_mk,_mk)} ⚪ *No AI interpretation*")
            _missing_interp.append(_mk)

    st.divider()

    # ── AI Toplu Yorum Üretimi ─────────────────────────────
    _report_ak = (st.session_state.get("api_key","") or
                  st.session_state.get("openai_api_key","") or
                  st.session_state.get("gemini_api_key",""))
    _report_provider = st.session_state.get("ai_provider", "Claude (Anthropic)")

    st.markdown('<div class="sec">🤖 AI Module Interpretations</div>', unsafe_allow_html=True)

    if not _report_ak:
        st.info("Enter an AI API key in the sidebar — you can generate interpretations for all modules here in one click.")
    else:
        # Interp fonksiyonu + argüman eşlemesi
        _INTERP_MAP = {
            "cv":  (interpret_cv,  lambda a: {"results": a.get("metrics",{}),
                                               "electrode_name": proj.get("name","")}),
            "eis": (interpret_eis, lambda a: {"param_df": None,
                                               "electrode_labels": a.get("metrics",{}).get("electrodes",[])}),
            "ph":  (interpret_ph,  lambda a: {"results": a.get("metrics",{}),
                                               "analyte": proj.get("notes","analyte")}),
            "cal": (interpret_calibration,
                                   lambda a: {"results": a.get("metrics",{}),
                                               "analyte": proj.get("notes","analyte"),
                                               "unit": "µM"}),
            "sa":  (interpret_standard_addition,
                                   lambda a: {"results": a.get("metrics",{}),
                                               "analyte": proj.get("notes","analyte"),
                                               "unit": "µM"}),
            "sel": (interpret_selectivity,
                                   lambda a: {"results": a.get("rows", []),
                                               "analyte": proj.get("notes","analyte"),
                                               "threshold_pct": float(
                                                   str(a.get("metrics",{}).get("Threshold","5%"))
                                                   .replace("±","").replace("%","").strip() or 5)}),
            "stab":(interpret_stability,
                                   lambda a: {"results": a.get("rows", []),
                                               "analyte": a.get("metrics",{}).get("Analyte","analyte"),
                                               "threshold_pct": float(
                                                   str(a.get("metrics",{}).get("Min. threshold","90%"))
                                                   .replace("%","").strip() or 90)}),
            "surf":(interpret_surface,
                                   lambda a: {"metrics": a.get("metrics",{}),
                                               "electrode_name": st.session_state.get("surf_electrode",
                                                   proj.get("name","modified electrode"))}),
            "dpv":(interpret_dpv_calibration,
                                   lambda a: {"metrics": a.get("metrics",{})}),
            "dpvopt":(interpret_dpv_optimization,
                                   lambda a: {"metrics": a.get("metrics",{})}),
            "realsample":(interpret_real_sample,
                                   lambda a: {"metrics": a.get("metrics",{})}),
            "litcomp":(interpret_literature_comparison,
                                   lambda a: {"metrics": a.get("metrics",{})}),
        }

        def _run_interp(key, fn, kwargs):
            """Generate AI interpretation (route by provider)."""
            if _report_provider == "ChatGPT (OpenAI)":
                return _call_openai(
                    st.session_state.get("openai_api_key",""),
                    st.session_state.get("gpt_model","gpt-4o"),
                    fn, kwargs)
            elif _report_provider == "Gemini (Google)":
                return _call_gemini(
                    st.session_state.get("gemini_api_key",""),
                    st.session_state.get("gemini_model","gemini-3.8-flash"),
                    fn, kwargs)
            else:
                return fn(api_key=st.session_state.get("api_key",""),
                          model=st.session_state.get("ai_model","claude-opus-5-5"),
                          **kwargs)

        # Mevcut yorumlar (göster)
        for k, a in analyses.items():
            interp = a.get("interpretation","")
            if interp:
                with st.expander(f"✅ {_mod_labels.get(k,k)} — saved interpretation"):
                    st.markdown(f'<div class="interp-box">{interp.replace(chr(10),"<br>")}</div>',
                                unsafe_allow_html=True)

        # Eksik yorumlar için "üret" butonları
        if _missing_interp:
            st.info(f"Missing interpretations: {', '.join(_mod_labels.get(k,k) for k in _missing_interp)}")

            _col_all, _col_sel = st.columns(2)
            if _col_all.button("🤖 Generate AI Interpretation for All", use_container_width=True,
                                key="ai_all_modules"):
                _progress = st.progress(0, text="Starting…")
                _total = len(_missing_interp)
                for _i, _mk in enumerate(_missing_interp):
                    _progress.progress((_i) / _total,
                                       text=f"Interpreting {_mod_labels.get(_mk,_mk)}…")
                    try:
                        _fn, _kw_fn = _INTERP_MAP[_mk]
                        _kw = _kw_fn(analyses[_mk])
                        _text = _run_interp(_mk, _fn, _kw)
                        analyses[_mk]["interpretation"] = _text
                        st.session_state[f"interp_{_mk}"] = _text
                    except Exception as _e:
                        st.warning(f"{_mod_labels.get(_mk,_mk)}: {_e}")
                _progress.progress(1.0, text="Done ✅")
                _persist_user_data(_current_user, _projects(),
                                   st.session_state.get(_user_key("current_project"), ""))
                st.rerun()

            # Bireysel seçim
            _sel_missing = _col_sel.multiselect(
                "Generate only for these",
                options=_missing_interp,
                format_func=lambda x: _mod_labels.get(x,x),
                key="ai_sel_modules"
            )
            if _sel_missing and st.button("🤖 Generate for Selected", use_container_width=True,
                                           key="ai_sel_run"):
                _progress2 = st.progress(0, text="Starting…")
                for _i, _mk in enumerate(_sel_missing):
                    _progress2.progress(_i / len(_sel_missing),
                                        text=f"Interpreting {_mod_labels.get(_mk,_mk)}…")
                    try:
                        _fn, _kw_fn = _INTERP_MAP[_mk]
                        _kw = _kw_fn(analyses[_mk])
                        _text = _run_interp(_mk, _fn, _kw)
                        analyses[_mk]["interpretation"] = _text
                        st.session_state[f"interp_{_mk}"] = _text
                    except Exception as _e:
                        st.warning(f"{_mod_labels.get(_mk,_mk)}: {_e}")
                _progress2.progress(1.0, text="Done ✅")
                _persist_user_data(_current_user, _projects(),
                                   st.session_state.get(_user_key("current_project"), ""))
                st.rerun()
        else:
            st.success("✅ AI interpretations available for all modules.")

    st.divider()

    # ── Toplu / Executive Summary ──────────────────────────
    # Resolve active key/model for whichever provider is selected
    _exec_provider = st.session_state.get("ai_provider", "Claude (Anthropic)")
    if _exec_provider == "ChatGPT (OpenAI)":
        ak = st.session_state.get("openai_api_key", "")
        _exec_model = st.session_state.get("gpt_model", "gpt-4o")
    elif _exec_provider == "Gemini (Google)":
        ak = st.session_state.get("gemini_api_key", "")
        _exec_model = st.session_state.get("gemini_model", "gemini-2.0-flash")
    else:
        ak = st.session_state.get("api_key", "")
        _exec_model = st.session_state.get("ai_model", "claude-opus-5-5")

    st.markdown('<div class="sec">📋 Bulk AI Report (Executive Summary)</div>',
                unsafe_allow_html=True)
    exec_sum = proj.get("executive_summary","")
    if exec_sum:
        st.markdown(f'<div class="interp-box">{exec_sum.replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True)
        if st.button("🔄 Regenerate"):
            exec_sum = ""
    if not exec_sum:
        if ak:
            _has_all = not _missing_interp
            _btn_lbl = ("🤖 Generate Bulk Report (all modules included)"
                        if _has_all else
                        "🤖 Generate Executive Summary (with available data)")
            if st.button(_btn_lbl, use_container_width=True, key="exec_sum_btn"):
                with st.spinner("Writing bulk report…"):
                    try:
                        # Route through the active AI provider
                        if _exec_provider == "ChatGPT (OpenAI)":
                            text = _call_openai(ak, _exec_model,
                                                interpret_full_project,
                                                {"project": proj})
                        elif _exec_provider == "Gemini (Google)":
                            text = _call_gemini(ak, _exec_model,
                                                interpret_full_project,
                                                {"project": proj})
                        else:
                            text = interpret_full_project(ak, proj, model=_exec_model)
                        proj["executive_summary"] = text
                        _persist_user_data(_current_user, _projects(),
                                           st.session_state.get(
                                               _user_key("current_project"), ""))
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            if not _has_all:
                st.caption("💡 Generate AI interpretations for all modules first to produce a more comprehensive bulk report.")
        else:
            proj_sum = st.text_area("Write your own summary (optional)",
                                     height=120, key="manual_exec_sum")
            if proj_sum.strip():
                proj["executive_summary"] = proj_sum.strip()

    st.divider()
    st.markdown('<div class="sec">Generate Report</div>', unsafe_allow_html=True)

    # ── Report type selector ──────────────────────────────────
    _rtype = st.radio(
        "Report type",
        ["📋  Summary", "📄  Detailed"],
        horizontal=True,
        key="report_type_radio",
        help=(
            "**Summary** — title page · executive summary · key metrics table · "
            "AI interpretation for each module. No figures. ~2–3 pages.  \n"
            "**Detailed** — everything in Summary plus embedded figures, raw data "
            "tables, EIS fit parameters, ICH Q2 precision breakdown. Full publication output."
        ),
    )
    _report_type_val = "summary" if _rtype.startswith("📋") else "detailed"

    # explain contents
    if _report_type_val == "summary":
        st.caption("📋 Summary includes: title · executive summary · metrics tables · AI interpretation. No figures.")
    else:
        st.caption("📄 Detailed includes: all Summary content + all embedded figures + raw data tables + EIS fit params.")

    col1, col2 = st.columns(2)
    gen_pdf  = col1.checkbox("PDF (.pdf)",   value=True)
    gen_docx = col2.checkbox("Word (.docx)", value=True)

    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        if not gen_pdf and not gen_docx:
            st.warning("Select at least one format.")
        else:
            with st.spinner("Building report…"):
                # ── Collect new-module session-state results into project ──
                import copy
                _proj_for_report = copy.deepcopy(proj)
                _proj_for_report.setdefault("analyses", {})

                # Selectivity
                if "sel_results" in st.session_state and st.session_state["sel_results"]:
                    _sr = st.session_state["sel_results"]
                    _pass_pct = sum(1 for r in _sr if r.get("Status","").startswith("✅")) / len(_sr) * 100 if _sr else 0
                    _proj_for_report["analyses"]["sel"] = {
                        "metrics": {
                            "Analyte":          st.session_state.get("sel_analyte","—"),
                            "Interferents tested": len(_sr),
                            "Pass rate":        f"{_pass_pct:.0f}%",
                            "Threshold":        f"±{st.session_state.get('sel_thresh',5):.0f}%",
                        },
                        "rows":        _sr,
                        "plot_bytes":  st.session_state.get("sel_plot_bytes", []),
                        "interpretation": st.session_state.get("sel_interp",""),
                    }

                # Stability
                if "stab_results" in st.session_state and st.session_state["stab_results"]:
                    _stbr = st.session_state["stab_results"]
                    _last_rec = _stbr[-1].get("% Recovery", 0) if _stbr else 0
                    _proj_for_report["analyses"]["stab"] = {
                        "metrics": {
                            "Analyte":          st.session_state.get("stab_analyte","—"),
                            "Days monitored":   len(_stbr),
                            "Min. threshold":   f"{st.session_state.get('stab_thresh',90):.0f}%",
                            "Final recovery":   f"{_last_rec:.1f}%",
                            "Final status":     _stbr[-1].get("Status","—") if _stbr else "—",
                        },
                        "rows":        _stbr,
                        "plot_bytes":  st.session_state.get("stab_plot_bytes", []),
                        "interpretation": st.session_state.get("stab_interp",""),
                    }

                # DPV Calibration
                _dpv_saved = _cur().get("analyses", {}).get("dpv", {})
                if _dpv_saved:
                    _proj_for_report["analyses"]["dpv"] = _dpv_saved

                # DPV Optimization
                _dpvopt_saved = _cur().get("analyses", {}).get("dpvopt", {})
                if _dpvopt_saved:
                    _proj_for_report["analyses"]["dpvopt"] = _dpvopt_saved

                # Real Sample Analysis
                _rs_saved = _cur().get("analyses", {}).get("realsample", {})
                if not _rs_saved and "rs_results" in st.session_state and st.session_state["rs_results"]:
                    _rsr = st.session_state["rs_results"]
                    _rs_saved = {
                        "metrics": {
                            "Analyte":          st.session_state.get("rs_analyte", "—"),
                            "Matrix":           st.session_state.get("rs_matrix", "—"),
                            "Samples tested":   len(_rsr),
                            "Mean recovery":    f"{sum(r.get('Recovery (%)', 0) for r in _rsr)/len(_rsr):.1f}%" if _rsr else "—",
                        },
                        "rows": _rsr,
                        "plot_bytes": st.session_state.get("rs_plot_bytes", []),
                        "interpretation": st.session_state.get("rs_interp", ""),
                    }
                if _rs_saved:
                    _proj_for_report["analyses"]["realsample"] = _rs_saved

                # Literature Comparison
                _lit_saved = _cur().get("analyses", {}).get("litcomp", {})
                if _lit_saved:
                    _proj_for_report["analyses"]["litcomp"] = _lit_saved

                # Surface Characterization
                _surf_metrics = {}
                _rct_b = st.session_state.get("rct_bare", 0.0)
                _rct_m = st.session_state.get("rct_mod",  0.0)
                if _rct_b > 0 and _rct_m > 0:
                    _surf_metrics["Rct Bare (Ω)"]    = round(_rct_b, 2)
                    _surf_metrics["Rct Modified (Ω)"] = round(_rct_m, 2)
                    _surf_metrics["Rct Ratio (Bare/Mod)"] = round(_rct_b / _rct_m, 2)
                _ipa_b = st.session_state.get("surf_ipa_bare", 0.0)
                _ipa_m = st.session_state.get("surf_ipa_mod",  0.0)
                if _ipa_b > 0 and _ipa_m > 0:
                    _surf_metrics["Ipa Bare (µA)"]    = round(_ipa_b, 3)
                    _surf_metrics["Ipa Modified (µA)"] = round(_ipa_m, 3)
                    _surf_metrics["Signal Enhancement"] = f"{(_ipa_m-_ipa_b)/_ipa_b*100:+.1f}%"
                _sem_note_val = st.session_state.get("sem_note","")
                if _sem_note_val:
                    _surf_metrics["SEM observation"] = _sem_note_val
                if _surf_metrics:
                    _proj_for_report["analyses"]["surf"] = {
                        "metrics": _surf_metrics,
                        "plot_bytes": st.session_state.get("surf_plot_bytes", []),
                        "interpretation": st.session_state.get("surf_interp",""),
                    }

                # ── Stable per-project output directory; purge old reports before regen ──
                import glob as _glob
                _proj_id   = re.sub(r"[^\w\-]", "_", proj.get("name", "project"))[:40]
                _report_dir = os.path.join(tempfile.gettempdir(), "silverstonelab", _proj_id)
                os.makedirs(_report_dir, exist_ok=True)
                # Delete any previously generated PDF/DOCX for this project
                for _old in _glob.glob(os.path.join(_report_dir, "*.pdf")) + \
                             _glob.glob(os.path.join(_report_dir, "*.docx")):
                    try:
                        os.remove(_old)
                    except OSError:
                        pass

                try:
                    paths = generate_reports(_proj_for_report, _report_dir,
                                             report_type=_report_type_val)
                    st.success("✅ Report ready!")
                    if gen_pdf and "pdf" in paths:
                        with open(paths["pdf"],"rb") as f:
                            st.download_button("⬇ Download PDF", f.read(),
                                               Path(paths["pdf"]).name,
                                               "application/pdf",
                                               use_container_width=True)
                    elif "pdf_error" in paths:
                        st.error(f"PDF error: {paths['pdf_error']}")

                    if gen_docx and "docx" in paths:
                        with open(paths["docx"],"rb") as f:
                            st.download_button(
                                "⬇ Download Word (.docx)", f.read(),
                                Path(paths["docx"]).name,
                                "application/vnd.openxmlformats-officedocument"
                                ".wordprocessingml.document",
                                use_container_width=True)
                    elif "docx_error" in paths:
                        st.error(f"Word error: {paths['docx_error']}")

                except Exception as e:
                    st.error(str(e))
                    import traceback; st.code(traceback.format_exc())

    st.divider()

    # ── Project ZIP export ────────────────────────────────────
    st.markdown('<div class="sec">📦 Export Project Data</div>', unsafe_allow_html=True)
    st.caption("Download all analysis metrics and embedded plots from this project as a single ZIP archive.")
    if st.button("📦 Download Project ZIP", use_container_width=False, key="proj_zip_btn"):
        import zipfile as _zf, io as _zio, json as _zjson
        _zip_buf = _zio.BytesIO()
        _proj_name_safe = re.sub(r"[^\w\-]","_", proj.get("name","project"))[:40]
        _date_str_zip = datetime.date.today().strftime("%Y%m%d")
        with _zf.ZipFile(_zip_buf, "w", _zf.ZIP_DEFLATED) as _zzip:
            # 1. Project metadata as JSON (no raw plot bytes — keep it readable)
            _meta = {k: v for k, v in proj.items() if k != "analyses"}
            _analyses_clean = {}
            for _zk, _za in analyses.items():
                _analyses_clean[_zk] = {kk: vv for kk, vv in _za.items()
                                         if kk not in ("plot_bytes",)}
            _meta["analyses"] = _analyses_clean
            _zzip.writestr(
                f"{_proj_name_safe}_metrics.json",
                _zjson.dumps(_meta, ensure_ascii=False, indent=2)
            )
            # 2. Metrics CSV per module
            for _zk, _za in analyses.items():
                _zm = _za.get("metrics", {})
                if _zm:
                    _zcsv = "Metric,Value\n" + "".join(
                        f"{str(k).replace(',','')},{str(v).replace(',','')}\n"
                        for k, v in _zm.items()
                        if isinstance(v, (int, float, str))
                    )
                    _zzip.writestr(f"{_zk}_metrics.csv", _zcsv)
            # 3. Plot PNGs per module
            for _zk, _za in analyses.items():
                for _pi, _pb in enumerate(_za.get("plot_bytes", []) or []):
                    if isinstance(_pb, bytes) and _pb:
                        _zzip.writestr(f"{_zk}_plot_{_pi+1}.png", _pb)
            # 4. AI interpretations as text
            _interp_txt = ""
            for _zk, _za in analyses.items():
                _zi = _za.get("interpretation","")
                if _zi:
                    _interp_txt += f"=== {_mod_labels.get(_zk,_zk)} ===\n{_zi}\n\n"
            if _interp_txt:
                _zzip.writestr("ai_interpretations.txt", _interp_txt)
        _zip_buf.seek(0)
        st.download_button(
            "⬇ Download ZIP",
            data=_zip_buf.getvalue(),
            file_name=f"{_proj_name_safe}_{_date_str_zip}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.divider()
    st.caption(
        "**Summary** reports are ideal for quick sharing and lab notebooks. "
        "**Detailed** reports are publication-ready with all figures and raw data embedded. "
        "File names include `_summary` or `_detailed` to distinguish them. "
        "©SilverstoneLab watermark is included on every page."
    )
