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
                                  interpret_full_project)
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
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1b2a 0%,#1b2838 60%,#0d1b2a 100%);
}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stMarkdown
    { color:#c8d8e8 !important; }
[data-testid="stSidebar"] h2 { color:#7ec8e3 !important; }
[data-testid="stSidebar"] h3 { color:#a8c8e0 !important; }
.block-container { padding-top:1.4rem; }
.sec { font-size:1rem;font-weight:600;color:#1f4e79;
       border-bottom:2px solid #1f4e79;padding-bottom:3px;margin-top:1rem; }
.wm  { position:fixed;bottom:10px;right:14px;font-size:11px;
       color:#1f4e79;opacity:.28;font-style:italic; }
.interp-box { background:#f8f9ff;border-left:4px solid #2e75b6;
              padding:14px 18px;border-radius:0 8px 8px 0;
              font-style:italic;line-height:1.6; }
.login-box { max-width:400px;margin:4rem auto;padding:2rem;
             border:1px solid #dde;border-radius:12px;
             background:#f8faff;text-align:center; }
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
        _ka = st.checkbox(
            "🔁 Oturumu açık tut (her 4 dk yenile)",
            value=st.session_state.get("keep_alive", False),
            key="keep_alive_cb",
            help="Giriş yaptıktan sonra da aktif kalır. Sayfa sessizce yenilenir, veri kaybı olmaz."
        )
        st.session_state["keep_alive"] = _ka
        if _ka:
            st.components.v1.html(
                """<script>
                if (!window._silKA) {
                    window._silKA = true;
                    setTimeout(function() { window.location.reload(); }, 240000);
                }
                </script>""",
                height=0,
            )
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
                   interpretation: str = ""):
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
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    # Auto-persist to disk after every analysis save
    _persist_user_data(_current_user, _projects(),
                       st.session_state.get(_user_key("current_project"), ""))

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
        "🔁 Oturumu açık tut",
        value=st.session_state.get("keep_alive", False),
        key="keep_alive_cb",
        help="İşaretli iken sayfa her 4 dakikada bir sessizce yenilenir; "
             "oturum kapanmaz, veriler kaydedildiğinden kayıp olmaz."
    )
    st.session_state["keep_alive"] = _keep_alive
    if _keep_alive:
        st.components.v1.html(
            """<script>
            if (!window._silKA) {
                window._silKA = true;
                setTimeout(function() { window.location.reload(); }, 240000);
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
                for uname, udata in cfg3["credentials"]["usernames"].items():
                    role = udata.get("role","user")
                    badge = "👑" if role=="admin" else "👤"
                    st.markdown(f"- {badge} **{uname}** ({udata.get('name','')})")
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
            _GPT_MODELS = ["gpt-4o","gpt-4o-mini","gpt-4-turbo","gpt-3.5-turbo"]
            if "gpt_model" not in st.session_state:
                st.session_state.gpt_model = _GPT_MODELS[0]
            st.session_state.gpt_model = st.selectbox(
                "Model", _GPT_MODELS,
                index=_GPT_MODELS.index(st.session_state.gpt_model)
                      if st.session_state.gpt_model in _GPT_MODELS else 0,
                key="gpt_model_select")
        else:
            st.caption("platform.openai.com → API Keys")

    else:  # Gemini
        _gmk = st.text_input("Gemini API key", value=st.session_state.gemini_api_key,
                              type="password", key="gemini_key_input", placeholder="AIza...")
        st.session_state.gemini_api_key = _gmk
        if _gmk:
            st.caption("✅ Gemini key active")
            _GEM_MODELS = ["gemini-2.0-flash","gemini-1.5-pro","gemini-1.5-flash"]
            if "gemini_model" not in st.session_state:
                st.session_state.gemini_model = _GEM_MODELS[0]
            st.session_state.gemini_model = st.selectbox(
                "Model", _GEM_MODELS,
                index=_GEM_MODELS.index(st.session_state.gemini_model)
                      if st.session_state.gemini_model in _GEM_MODELS else 0,
                key="gemini_model_select")
        else:
            st.caption("aistudio.google.com → Get API key")

    # ── Plot Style ───────────────────────────────────────────
    with st.expander("🎨 Plot Style"):
        _PALETTES = {
            "Navy Classic":  ['#1f4e79','#c55a11','#375623','#7030a0','#843c0c',
                              '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'],
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
**CV:** `Potential`(V), `Current`(A) — one file per scan rate
**EIS:** `Freq`(Hz), `Zreal`(Ω), `Zimag`(Ω)
**pH:** `pH`, `Ep_V`, `ip_A`
**Calibration:** `Concentration`, `ip_1`, `ip_2`, …
**Std addition:** `Addition`(µM), `ip`(µA)
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

def _ai_interpret_btn(module_key: str, interp_fn, fn_kwargs: dict,
                      result_key: str = "interpretation"):
    """Renders the 🤖 Interpret button and shows/saves commentary."""
    _provider = st.session_state.get("ai_provider", "Claude (Anthropic)")

    # Resolve active key and model
    if _provider == "ChatGPT (OpenAI)":
        _active_key   = st.session_state.get("openai_api_key", "")
        _active_model = st.session_state.get("gpt_model", "gpt-4o")
        _provider_label = "ChatGPT"
    elif _provider == "Gemini (Google)":
        _active_key   = st.session_state.get("gemini_api_key", "")
        _active_model = st.session_state.get("gemini_model", "gemini-2.0-flash")
        _provider_label = "Gemini"
    else:
        _active_key   = st.session_state.get("api_key", "")
        _active_model = st.session_state.get("ai_model", "claude-opus-5-5")
        _provider_label = "Claude"

    if not _active_key:
        st.info(f"🤖 Add your {_provider_label} API key in the sidebar to enable AI interpretation.")
        return

    if st.button(f"🤖 Generate AI Interpretation ({_provider_label})",
                 use_container_width=True, key=f"ai_{module_key}"):
        with st.spinner(f"Consulting the electrochemistry expert via {_provider_label}…"):
            try:
                if _provider == "ChatGPT (OpenAI)":
                    text = _call_openai(_active_key, _active_model, interp_fn, fn_kwargs)
                elif _provider == "Gemini (Google)":
                    text = _call_gemini(_active_key, _active_model, interp_fn, fn_kwargs)
                else:
                    text = interp_fn(api_key=_active_key, model=_active_model, **fn_kwargs)
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
# AI PROVIDER BRIDGES  (OpenAI · Gemini)
# ═══════════════════════════════════════════════════════════
def _call_openai(api_key: str, model: str, interp_fn, fn_kwargs: dict) -> str:
    """Bridge: build the same user prompt via interp_fn dummy, then call OpenAI."""
    try:
        import openai as _oai
    except ImportError:
        return ("⚠️ 'openai' package not installed.\n"
                "Add `openai>=1.0` to requirements.txt and redeploy.")
    # Reuse the prompt by calling _build_prompt if available, else call with dummy key
    # We call interp_fn with a placeholder key to get the prompt; we override the call
    from ai_interpreter import _SYSTEM
    # Build the user message by temporarily capturing what interp_fn would send
    # Simpler: directly call interp_fn but patch _call to return the prompt
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
        return result  # interp_fn short-circuited (no data)
    user_msg = _captured_prompt.get("msg", "Interpret the electrochemical data provided.")
    client = _oai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, max_tokens=1024,
        messages=[{"role":"system","content":_SYSTEM},
                  {"role":"user","content":user_msg}]
    )
    return resp.choices[0].message.content.strip()


def _call_gemini(api_key: str, model: str, interp_fn, fn_kwargs: dict) -> str:
    """Bridge: build prompt via interp_fn, then call Google Gemini."""
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
    _model = _genai.GenerativeModel(
        model_name=model,
        system_instruction=_SYSTEM
    )
    resp = _model.generate_content(user_msg)
    return resp.text.strip()


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

    # Row 2 — 3 modules
    _cols2 = st.columns(3, gap="medium")
    for _i, (_icon, _title, _desc, _nav, _key) in enumerate(_MODULES[3:]):
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


# ═══════════════════════════════════════════════════════════
# 📈  CV
# ═══════════════════════════════════════════════════════════
elif page == "📈  CV — Scan Rate Study":
    st.title("📈 CV — Scan Rate Study")
    st.caption(f"Project: **{_cur().get('name','')}**")

    left,right = st.columns([1,1])
    with left:
        st.markdown('<div class="sec">Parameters</div>', unsafe_allow_html=True)
        conc_mM    = st.number_input("Probe concentration (mM)", value=1.0)
        diff_coeff = st.number_input("D (cm²/s)", value=7.6e-6, format="%.2e")
        n_el       = st.number_input("n (electrons)", value=1, min_value=1, max_value=6)
    with right:
        st.markdown('<div class="sec">Input mode</div>', unsafe_allow_html=True)
        cv_mode = _input_mode("cv_mode")

    disk_map = {}
    tmpdir   = tempfile.mkdtemp()

    if cv_mode == "📁 Upload CSV":
        cv_ups = st.file_uploader("CV files (one per scan rate)",
                                   type="csv", accept_multiple_files=True)
        rate_map = {}
        if cv_ups:
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
                    "colors", ['#1f4e79','#c55a11','#375623','#7030a0',
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

    if disk_map and st.button("▶ Run CV Analysis",type="primary",use_container_width=True):
        with st.spinner("Analysing…"):
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
                _save_analysis("cv", metrics, out)
                st.markdown('<div class="sec">Plots</div>', unsafe_allow_html=True)
                _show_imgs(out); _dl_btn(out,"cv_analysis.zip")
                _ai_interpret_btn("cv", interpret_cv,
                                   {"results": res,
                                    "electrode_name": _cur().get("name","")})
            except ValueError as e:
                st.error(str(e))
                st.markdown("""
**CSV Formatı Kontrol Listesi:**
- Birinci sütun: `Potential` (V)
- İkinci sütun: `Current` (A)
- Tarama hızı bilgisi dosya adında (örn. `cv_50mVs.csv`, `50.csv`)
- Anodik pik tarama yönünde belirgin olmalı
""")
            except Exception as e:
                st.error(str(e))
                import traceback; st.code(traceback.format_exc())
    elif _cur().get("analyses",{}).get("cv"):
        a = _cur()["analyses"]["cv"]
        st.info(f"Previous result from {a.get('timestamp','')} stored in this project.")
        _ai_interpret_btn("cv", interpret_cv,
                           {"results": {"ecsa":{},"mechanism":{},"k0":{}},
                            "electrode_name": _cur().get("name","")})


# ═══════════════════════════════════════════════════════════
# 🔵  EIS
# ═══════════════════════════════════════════════════════════
elif page == "🔵  EIS — Impedance":
    st.title("🔵 EIS — Impedance Spectroscopy")
    st.caption(f"Project: **{_cur().get('name','')}**")

    left,right = st.columns([1,1])
    with left:
        st.markdown('<div class="sec">Parameters</div>', unsafe_allow_html=True)
        sign_z = st.selectbox("Z_imag sign",[-1,1],
                               format_func=lambda x:"−1 (most potentiostats)" if x==-1 else "+1")
    with right:
        st.markdown('<div class="sec">Input mode</div>', unsafe_allow_html=True)
        eis_mode = _input_mode("eis_mode")

    tmpdir=tempfile.mkdtemp(); file_dict={}

    if eis_mode == "📁 Upload CSV":
        eis_ups = st.file_uploader("EIS files",type="csv",accept_multiple_files=True)
        if eis_ups:
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
                "colors", ['#1f4e79','#c55a11','#375623','#7030a0',
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

    if file_dict and st.button("▶ Run EIS Analysis",type="primary",use_container_width=True):
        with st.spinner("Fitting…"):
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
                _save_analysis("eis", _eis_metrics, out, param_df=param_df)
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
        ph_n_el = st.number_input("n (electrons)",value=2,min_value=1,max_value=6)
        pk_type = st.selectbox("Peak type",["anodic","cathodic"])
        analyte_ph = st.text_input("Analyte (for report)",value="Analyte")
    with right:
        st.markdown('<div class="sec">Input mode</div>',unsafe_allow_html=True)
        ph_mode = _input_mode("ph_mode")

    tmpdir=tempfile.mkdtemp(); ph_path=None

    if ph_mode == "📁 Upload CSV":
        ph_up=st.file_uploader("pH table CSV",type="csv")
        if ph_up:
            st.dataframe(pd.read_csv(ph_up).head(10),use_container_width=True,hide_index=True)
            ph_up.seek(0); ph_path=os.path.join(tmpdir,"ph.csv")
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
        "colors", ['#1f4e79','#c55a11'])
    st.markdown('<div class="sec">🎨 Plot Colors</div>', unsafe_allow_html=True)
    _phc1, _phc2 = st.columns(2)
    _ph_data_col = _phc1.color_picker(
        "Data points", _ph_base[0], key="ph_col_data")
    _ph_fit_col  = _phc2.color_picker(
        "Fit line",    _ph_base[1] if len(_ph_base)>1 else '#c55a11',
        key="ph_col_fit")
    st.session_state["ph_custom_colors"] = [_ph_data_col, _ph_fit_col]

    if ph_path and st.button("▶ Run pH Analysis",type="primary",use_container_width=True):
        with st.spinner("Analysing…"):
            out=os.path.join(tmpdir,"ph_out"); os.makedirs(out,exist_ok=True)
            try:
                _ph_ps = dict(st.session_state.get("plot_style") or {})
                _ph_ps["colors"] = st.session_state.get(
                    "ph_custom_colors", _ph_ps.get("colors", ['#1f4e79','#c55a11']))
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
                _save_analysis("ph",metrics,out)
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
    with st.expander("📦 Toplu Validasyon Şablonu (.xlsx) — tek dosyada tüm bölümler",
                     expanded=False):
        col_dl, col_up2 = st.columns([1, 1])
        with col_dl:
            st.markdown("**1 · Şablonu İndir**")
            _tpl_path = os.path.join(os.path.dirname(__file__),
                                     "SilverstoneLab_Validation_Template.xlsx")
            if os.path.exists(_tpl_path):
                with open(_tpl_path, "rb") as _f:
                    st.download_button(
                        "⬇️ Şablonu İndir (.xlsx)",
                        data=_f.read(),
                        file_name="SilverstoneLab_Validation_Template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument"
                              ".spreadsheetml.sheet",
                        use_container_width=True,
                    )
                st.caption("README sekmesini okuyun · Yeşil satırlar örnek, sarı satırlara kendi verilerinizi girin · Sekme isimlerini değiştirmeyin.")
            else:
                st.warning("Şablon dosyası bulunamadı. Yönetici ile iletişime geçin.")

        with col_up2:
            st.markdown("**2 · Doldurulmuş Şablonu Yükle**")
            bulk_up = st.file_uploader("Doldurulmuş şablonu yükle (.xlsx)",
                                       type=["xlsx"], key="bulk_xlsx_up")
            if bulk_up:
                try:
                    import openpyxl as _opxl
                    _wb = _opxl.load_workbook(bulk_up, data_only=True)
                    _sheets = [s.title for s in _wb.worksheets]
                    st.success(f"✓ Yüklendi — sekmeler: {', '.join(_sheets)}")

                    def _xl_sheet_to_df(wb, name):
                        """Return a sheet as a list-of-lists (row 0 = headers)."""
                        if name not in [s.title for s in wb.worksheets]:
                            return None
                        ws = wb[name]
                        rows = list(ws.iter_rows(values_only=True))
                        return rows  # rows[0] = header

                    # ── Calibration ─────────────────────────────────────────
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
                            st.caption(f"  Kalibrasyon: {len(_bulk_cal)} konsantrasyon seviyesi aktarıldı.")

                    # ── Repeatability ────────────────────────────────────────
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
                            st.caption(f"  Tekrarlılık: {len(_bulk_rep)} seviye aktarıldı.")

                    # ── Inter_Precision ──────────────────────────────────────
                    _ipr_rows = _xl_sheet_to_df(_wb, "Inter_Precision")
                    if _ipr_rows and len(_ipr_rows) > 1:
                        _bulk_ipr = {}  # {gun: {seviye: [ip_vals]}}
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
                            st.caption(f"  Orta Kesinlik: {len(_bulk_ipr)} gün aktarıldı.")

                    # ── Accuracy ─────────────────────────────────────────────
                    _acc_rows = _xl_sheet_to_df(_wb, "Accuracy")
                    if _acc_rows and len(_acc_rows) > 1:
                        _bulk_acc = {}  # {qc_lbl: (nominal_c, [ip_vals])}
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
                            st.caption(f"  Accuracy: {len(_bulk_acc)} QC seviyesi aktarıldı.")

                    # ── Std_Addition ─────────────────────────────────────────
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
                            st.caption(f"  Standart Ekleme: {len(_bulk_sa)} nokta aktarıldı.")

                    # Kalibrasyon verisini bulk'tan al
                    if "bulk_cal" in st.session_state:
                        st.info("✅ Şablon verisi yüklendi. Aşağıdan analizi başlatın."
                                " ICH Q2 bölümlerine geçtiğinizde veriler otomatik uygulanır.")

                except Exception as _e:
                    st.error(f"Şablon okunamadı: {_e}")

    # Kalibrasyon verisi: bulk yükleme öncelikli
    if "bulk_cal" in st.session_state and st.session_state["bulk_cal"]:
        st.info("📦 Bulk şablon'dan kalibrasyon verisi kullanılıyor. "
                "Manuel giriş yapmak isterseniz aşağıya veri ekleyebilirsiniz.")

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
                                     value=0.0,min_value=0.0,format="%.4f")
    with right:
        st.markdown('<div class="sec">Input mode</div>',unsafe_allow_html=True)
        cal_mode = _input_mode("cal_mode")

    cal_data={}

    if cal_mode == "📁 Upload CSV":
        cal_up=st.file_uploader("Calibration CSV",type="csv")
        if cal_up:
            df_prev=pd.read_csv(cal_up)
            st.dataframe(df_prev,use_container_width=True,hide_index=True)
            cal_up.seek(0); df=pd.read_csv(cal_up); cc=df.columns[0]
            for _,row in df.iterrows():
                try:
                    conc_val = float(row[cc])
                except (ValueError, TypeError):
                    continue   # başlık / metadata satırı — atla
                vals=[]
                for c in df.columns[1:]:
                    try:
                        if pd.notna(row[c]):
                            vals.append(float(row[c]))
                    except (ValueError, TypeError):
                        pass   # sayısal olmayan hücre — atla
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
        "colors", ['#1f4e79','#c55a11'])
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
    with st.expander("🔬 ICH Q2(R2) Kesinlik Validasyonu (İsteğe Bağlı)", expanded=False):
        st.markdown("""
**ICH Q2(R2) Kabul Kriterleri:**
| Kesinlik Tipi | Minimum Şart | %RSD Limiti |
|---|---|---|
| **Tekrar Edilebilirlik** (Gün İçi) | ≥3 tekrar, ≥3 konsantrasyon | **≤ %2** |
| **Ara Kesinlik** (Günler Arası) | ≥3 gün, her günde ≥3 tekrar | **≤ %5** |
        """)

        ich_tab1, ich_tab2 = st.tabs(["📅 Gün İçi (Repeatability)", "📆 Günler Arası (Intermediate Precision)"])

        # --- Gün İçi -------------------------------------------------
        with ich_tab1:
            st.info("Aynı günde, farklı konsantrasyon seviyelerinde yapılan tekrar ölçümler. "
                    "ICH Q2(R2): en az 3 seviye (düşük/orta/yüksek), her seviyede ≥3 tekrar.")

            ich_intra_mode = st.radio("Veri girişi", ["📋 Tablo (önerilen)", "📁 CSV yükle"],
                                       horizontal=True, key="ich_intra_mode")

            intra_rows = []  # (etiket, list[float])

            if ich_intra_mode == "📁 CSV yükle":
                st.caption("Beklenen sütunlar: **Seviye**, **ip(uA)** — her satır bir tekrar.")
                ich_intra_up = st.file_uploader("Repeatability CSV", type="csv", key="ich_intra_up")
                if ich_intra_up:
                    try:
                        df_intra = pd.read_csv(ich_intra_up)
                        df_intra.columns = [c.strip() for c in df_intra.columns]
                        lbl_col = next((c for c in df_intra.columns if any(k in c.lower()
                                        for k in ('seviye','level','label','grup'))), df_intra.columns[0])
                        ip_col  = next((c for c in df_intra.columns if any(k in c.lower()
                                        for k in ('ip','current','i(','akım'))), None)
                        if ip_col is None:
                            st.error("CSV'de 'ip' sütunu bulunamadı.")
                        else:
                            for lbl, grp in df_intra.groupby(lbl_col, sort=False):
                                vals = grp[ip_col].dropna().astype(float).tolist()
                                if vals:
                                    intra_rows.append((str(lbl), vals))
                            st.success(f"✓ {len(intra_rows)} seviye yüklendi")
                            st.dataframe(df_intra, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"CSV okunamadı: {e}")
            else:
                # Tablo modu — her satır bir konsantrasyon seviyesi, sütunlar tekrarlar
                st.caption("Her satır = bir konsantrasyon seviyesi. Boş hücreler otomatik atlanır.")
                default_intra = pd.DataFrame({
                    'Seviye (Etiket)': ['Düşük', 'Orta', 'Yüksek'],
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
                    lbl = str(row['Seviye (Etiket)']).strip() if pd.notna(row['Seviye (Etiket)']) else ''
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
            st.info("Aynı konsantrasyon seviyesinde farklı günlerde yapılan ölçümler. "
                    "Her gün en az 3 tekrar önerilir.")

            ich_inter_mode = st.radio("Veri girişi", ["📋 Tablo (önerilen)", "📁 CSV yükle"],
                                       horizontal=True, key="ich_inter_mode")

            inter_rows = []  # (etiket, list[float])

            if ich_inter_mode == "📁 CSV yükle":
                st.caption("Beklenen sütunlar: **Gun**, **ip(uA)** — her satır bir tekrar.")
                ich_inter_up = st.file_uploader("Intermediate Precision CSV", type="csv", key="ich_inter_up")
                if ich_inter_up:
                    try:
                        df_inter = pd.read_csv(ich_inter_up)
                        df_inter.columns = [c.strip() for c in df_inter.columns]
                        day_col = next((c for c in df_inter.columns if any(k in c.lower()
                                        for k in ('gun','day','gün','date'))), df_inter.columns[0])
                        ip_col  = next((c for c in df_inter.columns if any(k in c.lower()
                                        for k in ('ip','current','i(','akım'))), None)
                        if ip_col is None:
                            st.error("CSV'de 'ip' sütunu bulunamadı.")
                        else:
                            for day, grp in df_inter.groupby(day_col, sort=False):
                                vals = grp[ip_col].dropna().astype(float).tolist()
                                if vals:
                                    inter_rows.append((str(day), vals))
                            st.success(f"✓ {len(inter_rows)} gün yüklendi")
                            st.dataframe(df_inter, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"CSV okunamadı: {e}")
            else:
                # Tablo modu — her satır bir gün, sütunlar tekrarlar
                st.caption("Her satır = bir gün. Boş hücreler otomatik atlanır.")
                default_inter = pd.DataFrame({
                    'Gün (Etiket)': ['Gün 1', 'Gün 2', 'Gün 3'],
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
                    lbl = str(row['Gün (Etiket)']).strip() if pd.notna(row['Gün (Etiket)']) else ''
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
            st.info(f"📦 Bulk şablondan tekrarlılık verisi yüklendi: {[r[0] for r in intra_rows]}")
        if not inter_rows and "bulk_ipr" in st.session_state:
            # {gun: {seviye: [vals]}} → flatten to [(gun_sev_lbl, vals)]
            for _gun, _sevs in st.session_state["bulk_ipr"].items():
                for _sev, _vals in _sevs.items():
                    inter_rows.append((f"{_gun} – {_sev}", _vals))
            if inter_rows:
                st.info(f"📦 Bulk şablondan orta kesinlik verisi yüklendi: {len(inter_rows)} grup")

        if st.button("▶ ICH Q2(R2) Validasyon Hesapla", type="secondary", use_container_width=True,
                     key="btn_ich_q2"):
            # intra_rows ve inter_rows her modda zaten (etiket, list[float]) formatında
            intraday_data_ich = {lbl: vs for lbl, vs in intra_rows if len(vs) >= 2}

            # Parse inter-day
            interday_data_ich = {lbl: vs for lbl, vs in inter_rows if len(vs) >= 2}

            if not intraday_data_ich:
                st.error("En az 1 gün içi veri grubu girilmeli (etiket + virgülle ayrılmış değerler).")
            else:
                with st.spinner("ICH Q2(R2) hesaplanıyor…"):
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
                        c1.metric(f"{intra_icon} Gün İçi Maks. %RSD",
                                  f"{intra_rsd_max:.2f}%",
                                  delta=f"Limit: 2.0% — {'Geçti' if intra_ok else 'BAŞARISIZ'}",
                                  delta_color="normal" if intra_ok else "inverse")
                        if inter_rsd_tot is not None:
                            inter_icon = "✅" if inter_ok else "❌"
                            c2.metric(f"{inter_icon} Günler Arası Toplam %RSD",
                                      f"{inter_rsd_tot:.2f}%",
                                      delta=f"Limit: 5.0% — {'Geçti' if inter_ok else 'BAŞARISIZ'}",
                                      delta_color="normal" if inter_ok else "inverse")

                        # ── Detay tablosu ──
                        st.markdown("**Gün İçi Sonuçları:**")
                        rows_disp = []
                        for lbl, r in ich_res['intraday'].items():
                            rows_disp.append({
                                'Grup': lbl, 'n': r['n'],
                                'Ortalama (μA)': f"{r['mean']:.4f}",
                                'SD (μA)': f"{r['std']:.4f}",
                                '%RSD': f"{r['rsd']:.2f}%",
                                'ICH Limit': '≤2%',
                                'Durum': '✅ Geçti' if r['pass'] else '❌ Başarısız'
                            })
                        st.dataframe(pd.DataFrame(rows_disp),
                                     use_container_width=True, hide_index=True)

                        if ich_res['interday']:
                            st.markdown("**Günler Arası Sonuçları:**")
                            rows_inter_disp = []
                            for day, r in ich_res['interday'].items():
                                if day.startswith('_'):
                                    continue
                                rows_inter_disp.append({
                                    'Gün': day, 'n': r['n'],
                                    'Ortalama (μA)': f"{r['mean']:.4f}",
                                    'SD (μA)': f"{r['std']:.4f}",
                                    '%RSD (gün içi)': f"{r['rsd']:.2f}%",
                                })
                            st.dataframe(pd.DataFrame(rows_inter_disp),
                                         use_container_width=True, hide_index=True)
                            if inter_rsd_tot is not None:
                                inter_icon = "✅" if inter_ok else "❌"
                                st.markdown(f"**Günler arası toplam %RSD:** {inter_rsd_tot:.2f}%  "
                                            f"(ICH limit ≤5%)  {inter_icon}")

                        # ── Grafik ──
                        _show_imgs(ich_out)
                        _dl_btn(ich_out, "ich_q2_validation.zip")

                    except Exception as e:
                        st.error(f"ICH Q2 hesaplama hatası: {e}")
                        import traceback; st.code(traceback.format_exc())

    # ── QC Noktaları + Accuracy ──────────────────────────────
    st.divider()
    with st.expander("🎯 QC Noktaları & Accuracy / Geri Kazanım (İsteğe Bağlı)", expanded=False):

        # ── QC Önerisi ──────────────────────────────────────────
        st.markdown("### 💡 QC Noktası Önerisi")
        st.caption("Girdiğin kalibrasyon verisi üzerinden otomatik hesaplanır. "
                   "Önce kalibrasyon verisini gir, sonra butona bas.")

        if st.button("🔢 QC Noktalarını Hesapla (LLOQ · LQC · MQC · HQC)",
                     key="btn_qc_suggest", type="secondary"):
            if not cal_data:
                st.warning("Önce kalibrasyon verisini gir.")
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
                    {"QC Seviyesi": "LLOQ", "Nominal C": f"{lloq:.4g} {conc_unit}",
                     "Beklenen ip (µA)": _ip(lloq), "Açıklama": "Alt tayin sınırı (LOQ)"},
                    {"QC Seviyesi": "LQC",  "Nominal C": f"{lqc:.4g} {conc_unit}",
                     "Beklenen ip (µA)": _ip(lqc),  "Açıklama": "Düşük QC ≈ 3× LLOQ"},
                    {"QC Seviyesi": "MQC",  "Nominal C": f"{mqc:.4g} {conc_unit}",
                     "Beklenen ip (µA)": _ip(mqc),  "Açıklama": "Orta QC (geometrik orta)"},
                    {"QC Seviyesi": "HQC",  "Nominal C": f"{hqc:.4g} {conc_unit}",
                     "Beklenen ip (µA)": _ip(hqc),  "Açıklama": "Yüksek QC ≈ %80 üst sınır"},
                ])
                st.dataframe(qc_df, use_container_width=True, hide_index=True)
                st.caption(f"Kalibrasyon: slope={_slope:.4f} µA/{conc_unit} | "
                           f"LOD={_lod:.4g} | LOQ={_loq:.4g} {conc_unit} | "
                           f"r={_r:.5f} | Aralık: {_cmin:.4g}–{_cmax:.4g} {conc_unit}")
                # Session state'e kaydet — Accuracy tablosunda kullanmak için
                st.session_state["qc_points"] = {
                    "slope": _slope, "intercept": _intercept,
                    "lloq": lloq, "lqc": lqc, "mqc": mqc, "hqc": hqc,
                    "unit": conc_unit
                }

        st.divider()

        # ── Accuracy / Geri Kazanım Tablosu ─────────────────────
        st.markdown("### 📐 Accuracy (Geri Kazanım)")
        st.caption("Her satır = bir QC seviyesi. Her sütun = bir tekrar ölçümü (ip, µA). "
                   "Nominal konsantrasyon ile ölçülen ip'den geri kazanım hesaplanır.")

        acc_mode = st.radio("Veri girişi", ["📋 Tablo (önerilen)", "📁 CSV yükle"],
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
            st.info(f"📦 Bulk şablondan accuracy verisi yüklendi: {[d[0] for d in acc_tbl_data]}")

        if acc_mode == "📁 CSV yükle":
            st.caption(
                "Beklenen sütunlar: **QC Seviyesi** (LLOQ/LQC/MQC/HQC), "
                f"**Nominal C ({conc_unit})**, **ip(uA)** — her satır bir tekrar. "
                "Üretilen `4_accuracy_recovery.csv` doğrudan yüklenebilir."
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
                        st.error("Nominal konsantrasyon sütunu bulunamadı.")
                    elif ip_col_up is None and not ip_run_cols:
                        st.error("ip ölçüm sütunu bulunamadı.")
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
                        st.success(f"✓ {len(acc_tbl_data)} QC seviyesi yüklendi: "
                                   f"{[d[0] for d in acc_tbl_data]}")
                except Exception as e:
                    st.error(f"CSV okunamadı: {e}")
        else:
            default_acc = pd.DataFrame({
                'QC Seviyesi':    ['LLOQ', 'LQC', 'MQC', 'HQC'],
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
                lbl = str(row['QC Seviyesi']).strip() if pd.notna(row['QC Seviyesi']) else ''
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

        if st.button("▶ Accuracy Hesapla", key="btn_acc", type="secondary"):
            _qc_now = st.session_state.get("qc_points", {})
            if not _qc_now and not cal_data:
                st.warning("Önce 'QC Noktalarını Hesapla' butonuna bas veya kalibrasyon verisini gir.")
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
                        'QC Seviyesi': lbl,
                        f'Nominal C ({conc_unit})': nom_c,
                        'n': len(vals),
                        'ip ort. (µA)': round(ip_mean, 4),
                        'ip SD (µA)': round(ip_std, 4),
                        'ip %RSD': round(ip_rsd, 2),
                        f'Hesaplanan C ({conc_unit})': round(calc_c, 4),
                        'Geri Kazanım (%)': round(rec_pct, 2),
                        'ICH Kabul (98–102%)': "✅" if 98 <= rec_pct <= 102 else "⚠️"
                    })

                if acc_rows:
                    acc_res_df = pd.DataFrame(acc_rows)
                    st.dataframe(acc_res_df, use_container_width=True, hide_index=True)
                    all_pass = all(98 <= r['Geri Kazanım (%)'] <= 102 for r in acc_rows
                                   if not np.isnan(r['Geri Kazanım (%)']))
                    if all_pass:
                        st.success("✅ Tüm QC seviyeleri ICH Q2(R2) accuracy kriterini karşılıyor (98–102%)")
                    else:
                        st.warning("⚠️ Bazı QC seviyeleri 98–102% aralığının dışında.")
                    # Kaydet
                    _save_analysis("cal_accuracy",
                                   {r['QC Seviyesi']: r['Geri Kazanım (%)'] for r in acc_rows},
                                   None)
                else:
                    st.warning("Tabloya en az 1 satır dolu veri gir.")

    if cal_data and st.button("▶ Run Calibration Analysis",type="primary",use_container_width=True):
        with st.spinner("Computing…"):
            tmpdir=tempfile.mkdtemp()
            out=os.path.join(tmpdir,"cal_out"); os.makedirs(out)
            try:
                _cal_ps = dict(st.session_state.get("plot_style") or {})
                _cal_ps["colors"] = st.session_state.get(
                    "cal_custom_colors", _cal_ps.get("colors", ['#1f4e79','#c55a11']))
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
                _save_analysis("cal",metrics,out)
                st.markdown('<div class="sec">Plots</div>',unsafe_allow_html=True)
                _show_imgs(out); _dl_btn(out,"calibration.zip")
                _ai_interpret_btn("cal",interpret_calibration,
                                   {"results":res,"analyte":analyte,"unit":conc_unit})
            except Exception as e:
                st.error(str(e))
                import traceback; st.code(traceback.format_exc())
    elif _cur().get("analyses",{}).get("cal"):
        a=_cur()["analyses"]["cal"]
        st.info(f"Previous calibration from {a.get('timestamp','')} stored.")
        _ai_interpret_btn("cal",interpret_calibration,{"results":{},"analyte":analyte,"unit":conc_unit})


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
        sa_up=st.file_uploader("Standard addition CSV",type="csv")
        if sa_up:
            df_sa=pd.read_csv(sa_up)
            st.dataframe(df_sa,use_container_width=True,hide_index=True)
            sa_up.seek(0); df_sa2=pd.read_csv(sa_up)
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

    if additions and ip_vals and st.button("▶ Run Standard Addition",type="primary",use_container_width=True):
        with st.spinner("Computing…"):
            tmpdir=tempfile.mkdtemp()
            out=os.path.join(tmpdir,"sa_out"); os.makedirs(out)
            try:
                res=standard_addition(additions_uM=additions,ip_uA=ip_vals,
                                       analyte_name=sa_analyte,output_dir=out,
                                       plot_style=st.session_state.get("plot_style"))
                st.success("✅ Done — results saved to project")
                metrics={
                    f"C_sample ({sa_cu})": f"{res.get('C_sample_uM',0):.4f}",
                    "R²":   f"{res.get('R2',0):.5f}",
                    "Slope":f"{res.get('slope',0):.4f} µA/{sa_cu}",
                }
                m1,m2,m3=st.columns(3)
                m1.metric(f"C_sample ({sa_cu})",metrics[f"C_sample ({sa_cu})"])
                m2.metric("R²",metrics["R²"])
                m3.metric("Slope",metrics["Slope"])
                _save_analysis("sa",metrics,out)
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
                   "cal":"📊 Calibration","sa":"➕ Std Addition"}
    st.markdown("**Analyses ready for report:**")
    _missing_interp = []
    for k, a in analyses.items():
        interp = a.get("interpretation","")
        if interp:
            st.markdown(f"- 🤖 {_mod_labels.get(k,k)} — {a.get('timestamp','')} ✅ *AI yorumu var*")
        else:
            st.markdown(f"- 📊 {_mod_labels.get(k,k)} — {a.get('timestamp','')} ⚪ *AI yorumu yok*")
            _missing_interp.append(k)

    st.divider()

    # ── AI Toplu Yorum Üretimi ─────────────────────────────
    _report_ak = (st.session_state.get("api_key","") or
                  st.session_state.get("openai_api_key","") or
                  st.session_state.get("gemini_api_key",""))
    _report_provider = st.session_state.get("ai_provider", "Claude (Anthropic)")

    st.markdown('<div class="sec">🤖 AI Modül Yorumları</div>', unsafe_allow_html=True)

    if not _report_ak:
        st.info("Sidebar'dan bir AI API anahtarı girin — tüm modüllerin yorumlarını buradan tek seferde üretebilirsiniz.")
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
        }

        def _run_interp(key, fn, kwargs):
            """AI yorumu üret (provider'a göre yönlendir)."""
            if _report_provider == "ChatGPT (OpenAI)":
                return _call_openai(
                    st.session_state.get("openai_api_key",""),
                    st.session_state.get("gpt_model","gpt-4o"),
                    fn, kwargs)
            elif _report_provider == "Gemini (Google)":
                return _call_gemini(
                    st.session_state.get("gemini_api_key",""),
                    st.session_state.get("gemini_model","gemini-2.0-flash"),
                    fn, kwargs)
            else:
                return fn(api_key=st.session_state.get("api_key",""),
                          model=st.session_state.get("ai_model","claude-opus-5-5"),
                          **kwargs)

        # Mevcut yorumlar (göster)
        for k, a in analyses.items():
            interp = a.get("interpretation","")
            if interp:
                with st.expander(f"✅ {_mod_labels.get(k,k)} — kayıtlı yorum"):
                    st.markdown(f'<div class="interp-box">{interp.replace(chr(10),"<br>")}</div>',
                                unsafe_allow_html=True)

        # Eksik yorumlar için "üret" butonları
        if _missing_interp:
            st.info(f"Yorum eksik modüller: {', '.join(_mod_labels.get(k,k) for k in _missing_interp)}")

            _col_all, _col_sel = st.columns(2)
            if _col_all.button("🤖 Tümü İçin AI Yorum Üret", use_container_width=True,
                                key="ai_all_modules"):
                _progress = st.progress(0, text="Başlıyor…")
                _total = len(_missing_interp)
                for _i, _mk in enumerate(_missing_interp):
                    _progress.progress((_i) / _total,
                                       text=f"{_mod_labels.get(_mk,_mk)} yorumlanıyor…")
                    try:
                        _fn, _kw_fn = _INTERP_MAP[_mk]
                        _kw = _kw_fn(analyses[_mk])
                        _text = _run_interp(_mk, _fn, _kw)
                        analyses[_mk]["interpretation"] = _text
                        st.session_state[f"interp_{_mk}"] = _text
                    except Exception as _e:
                        st.warning(f"{_mod_labels.get(_mk,_mk)}: {_e}")
                _progress.progress(1.0, text="Tamamlandı ✅")
                _persist_user_data(_current_user, _projects(),
                                   st.session_state.get(_user_key("current_project"), ""))
                st.rerun()

            # Bireysel seçim
            _sel_missing = _col_sel.multiselect(
                "Sadece bunlar için üret",
                options=_missing_interp,
                format_func=lambda x: _mod_labels.get(x,x),
                key="ai_sel_modules"
            )
            if _sel_missing and st.button("🤖 Seçilenleri Üret", use_container_width=True,
                                           key="ai_sel_run"):
                _progress2 = st.progress(0, text="Başlıyor…")
                for _i, _mk in enumerate(_sel_missing):
                    _progress2.progress(_i / len(_sel_missing),
                                        text=f"{_mod_labels.get(_mk,_mk)} yorumlanıyor…")
                    try:
                        _fn, _kw_fn = _INTERP_MAP[_mk]
                        _kw = _kw_fn(analyses[_mk])
                        _text = _run_interp(_mk, _fn, _kw)
                        analyses[_mk]["interpretation"] = _text
                        st.session_state[f"interp_{_mk}"] = _text
                    except Exception as _e:
                        st.warning(f"{_mod_labels.get(_mk,_mk)}: {_e}")
                _progress2.progress(1.0, text="Tamamlandı ✅")
                _persist_user_data(_current_user, _projects(),
                                   st.session_state.get(_user_key("current_project"), ""))
                st.rerun()
        else:
            st.success("✅ Tüm modüllerin AI yorumu mevcut.")

    st.divider()

    # ── Toplu / Executive Summary ──────────────────────────
    ak = st.session_state.get("api_key","")
    st.markdown('<div class="sec">📋 Toplu AI Raporu (Executive Summary)</div>',
                unsafe_allow_html=True)
    exec_sum = proj.get("executive_summary","")
    if exec_sum:
        st.markdown(f'<div class="interp-box">{exec_sum.replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True)
        if st.button("🔄 Yeniden üret"):
            exec_sum = ""
    if not exec_sum:
        if ak:
            _has_all = not _missing_interp   # True ise tüm modüllerin yorumu var
            _btn_lbl = ("🤖 Toplu Rapor Üret (tüm modüller dahil)"
                        if _has_all else
                        "🤖 Executive Summary Üret (mevcut verilerle)")
            if st.button(_btn_lbl, use_container_width=True, key="exec_sum_btn"):
                with st.spinner("Toplu rapor yazılıyor…"):
                    try:
                        text = interpret_full_project(ak, proj,
                                                      model=st.session_state.get(
                                                          "ai_model","claude-opus-5-5"))
                        proj["executive_summary"] = text
                        _persist_user_data(_current_user, _projects(),
                                           st.session_state.get(
                                               _user_key("current_project"), ""))
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            if not _has_all:
                st.caption("💡 Önce tüm modüllerin AI yorumunu üretirseniz toplu rapor daha kapsamlı olur.")
        else:
            proj_sum = st.text_area("Kendi özetinizi yazın (opsiyonel)",
                                     height=120, key="manual_exec_sum")
            if proj_sum.strip():
                proj["executive_summary"] = proj_sum.strip()

    st.divider()
    st.markdown('<div class="sec">Generate Report</div>', unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    gen_pdf  = col1.checkbox("PDF (.pdf)",  value=True)
    gen_docx = col2.checkbox("Word (.docx)",value=True)

    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        if not gen_pdf and not gen_docx:
            st.warning("Select at least one format.")
        else:
            with st.spinner("Building report…"):
                tmpdir = tempfile.mkdtemp()
                try:
                    paths = generate_reports(proj, tmpdir)
                    st.success("✅ Report ready!")
                    if gen_pdf and "pdf" in paths:
                        with open(paths["pdf"],"rb") as f:
                            st.download_button("⬇ Download PDF",f.read(),
                                               Path(paths["pdf"]).name,
                                               "application/pdf",
                                               use_container_width=True)
                    elif "pdf_error" in paths:
                        st.error(f"PDF error: {paths['pdf_error']}")

                    if gen_docx and "docx" in paths:
                        with open(paths["docx"],"rb") as f:
                            st.download_button(
                                "⬇ Download Word (.docx)",f.read(),
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
    st.caption("Reports embed all plots, metric tables, and AI interpretations "
               "in a publication-ready layout. ©SilverstoneLab watermark is "
               "included on every page.")
