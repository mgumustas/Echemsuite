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
from analytical_module  import run_analytical_performance, standard_addition
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

    # ── Password change ──────────────────────────────────────
    with st.expander("🔑 Change my password"):
        if _AUTH_AVAILABLE:
            try:
                _authenticator.reset_password(_current_username if hasattr(
                    _authenticator,'_current_username') else _current_user,
                    location="sidebar")
                st.success("Password updated!")
            except Exception:
                # newer API
                try:
                    if _authenticator.reset_password(_current_user, location="sidebar"):
                        _save_config(_cfg)
                        st.success("Password updated!")
                except Exception as e:
                    st.caption(str(e))
        else:
            st.caption("Auth not available in dev mode.")

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

    # ── Claude API key ───────────────────────────────────────
    st.markdown("### 🤖 Claude AI")
    api_key_input = st.text_input("API key (console.anthropic.com)",
                                   value=st.session_state.api_key,
                                   type="password", key="api_key_input",
                                   placeholder="sk-ant-...")
    st.session_state.api_key = api_key_input
    if api_key_input:
        st.caption("✅ API key saved for this session")
    else:
        st.caption("Enter key to enable AI interpretations")
    _MODEL_OPTIONS = [
        "claude-opus-4-0",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20241022",
    ]
    if "ai_model" not in st.session_state:
        st.session_state.ai_model = _MODEL_OPTIONS[0]
    st.session_state.ai_model = st.selectbox(
        "Model", _MODEL_OPTIONS,
        index=_MODEL_OPTIONS.index(st.session_state.ai_model)
              if st.session_state.ai_model in _MODEL_OPTIONS else 0,
        key="ai_model_select"
    )

    st.divider()

    # ── Navigation ───────────────────────────────────────────
    page = st.radio(f"navigate_{_current_user}", label_visibility="collapsed", options=[
        "🏠  Home",
        "📈  CV — Scan Rate Study",
        "🔵  EIS — Impedance",
        "🧪  pH — Working pH",
        "📊  Calibration & LOD/LOQ",
        "➕  Standard Addition",
        "📄  Report",
    ])

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
    ak = st.session_state.api_key
    if not ak:
        st.info("🤖 Add your Claude API key in the sidebar to enable AI interpretation.")
        return
    if st.button("🤖 Generate AI Interpretation", use_container_width=True,
                 key=f"ai_{module_key}"):
        with st.spinner("Consulting the electrochemistry expert…"):
            try:
                _model = st.session_state.get("ai_model", "claude-opus-4-0")
                text = interp_fn(api_key=ak, model=_model, **fn_kwargs)
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
    st.title(f"⚗️ {proj.get('name','My Study')}")
    st.caption(f"Created {proj.get('created','')}  |  SilverstoneLab Electrochemical Analysis Suite v3.0")
    if proj.get("notes"):
        st.info(proj["notes"])

    done = proj.get("analyses", {})
    if done:
        st.markdown("### ✅ Analyses in this project")
        labels = {"cv":"📈 CV","eis":"🔵 EIS","ph":"🧪 pH",
                  "cal":"📊 Calibration","sa":"➕ Std Addition"}
        cols = st.columns(min(len(done),5))
        for i,(k,a) in enumerate(done.items()):
            cols[i%5].metric(labels.get(k,k), a.get("timestamp","done"))
        st.markdown("→ Go to **📄 Report** to generate your PDF + Word report.")
    else:
        st.markdown("""
### Quick start
1. Select a module in the left sidebar
2. Upload a CSV **or** type your numbers directly
3. Click **▶ Run** — results, plots and AI interpretation appear instantly
4. When all analyses are done, go to **📄 Report** for a one-click PDF + Word export
""")
    st.divider()
    c1,c2 = st.columns(2)
    c1.markdown("""
| Module | Key outputs |
|--------|-------------|
| 📈 CV  | ECSA · b-value · k° · E½ |
| 🔵 EIS | Rct · CPE · roughness |
| 🧪 pH  | Ep/pH slope · Nernst · optimal pH |
| 📊 Cal | LOD · LOQ · linear range · r |
| ➕ SA  | C_sample · 95 % PI |
""")
    c2.markdown("""
**New in v3.0**
- 📁 **Project folders** — keep studies separate, switch anytime
- 🤖 **AI interpretation** — Claude expert commentary per module
- 📄 **Reports** — publication-ready PDF + Word with plots embedded
- ✏️ **Manual entry** — type values directly, no CSV needed
""")


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
                res = run_cv_analysis(disk_map,conc_mM,diff_coeff,int(n_el),out)
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
            for i,f in enumerate(eis_ups):
                lbl = st.text_input(f"`{f.name}`",
                                     value=f.name.replace('.csv','').replace('_',' '),
                                     key=f"eis_l{i}")
                p = os.path.join(tmpdir,f"eis_{i}.csv")
                Path(p).write_bytes(f.getvalue()); file_dict[lbl]=p
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
                _eis_result=run_eis_analysis(file_dict,out,sign_zimag=sign_z)
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

    if ph_path and st.button("▶ Run pH Analysis",type="primary",use_container_width=True):
        with st.spinner("Analysing…"):
            out=os.path.join(tmpdir,"ph_out"); os.makedirs(out,exist_ok=True)
            try:
                res=run_ph_analysis(ph_path,int(ph_n_el),pk_type,out)
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

    left,right = st.columns([1,1])
    with left:
        st.markdown('<div class="sec">Parameters</div>',unsafe_allow_html=True)
        analyte   = st.text_input("Analyte name",value="Analyte")
        conc_unit = st.text_input("Concentration unit",value="μM")
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
                vals=[float(row[c]) for c in df.columns[1:] if pd.notna(row[c])]
                if vals: cal_data[float(row[cc])]=vals
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

    if cal_data and st.button("▶ Run Calibration Analysis",type="primary",use_container_width=True):
        with st.spinner("Computing…"):
            tmpdir=tempfile.mkdtemp()
            out=os.path.join(tmpdir,"cal_out"); os.makedirs(out)
            try:
                res=run_analytical_performance(
                    calibration_source=cal_data,analyte_name=analyte,
                    electrode_area_cm2=float(area_cal) if area_cal>0 else None,
                    unit=conc_unit,output_dir=out)
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
        sa_cu     =st.text_input("Concentration unit",value="μM")
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
                                       analyte_name=sa_analyte,output_dir=out)
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

    # summary of what's ready
    labels = {"cv":"📈 CV","eis":"🔵 EIS","ph":"🧪 pH",
              "cal":"📊 Calibration","sa":"➕ Std Addition"}
    st.markdown("**Analyses ready for report:**")
    for k,a in analyses.items():
        interp = a.get("interpretation","")
        icon   = "🤖" if interp else "📊"
        st.markdown(f"- {icon} {labels.get(k,k)} — {a.get('timestamp','')}"
                    + (" *(with AI interpretation)*" if interp else ""))

    st.divider()

    # Executive summary generation
    ak = st.session_state.api_key
    st.markdown('<div class="sec">Executive Summary</div>', unsafe_allow_html=True)
    exec_sum = proj.get("executive_summary","")
    if exec_sum:
        st.markdown(f'<div class="interp-box">{exec_sum.replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True)
        if st.button("🔄 Regenerate executive summary"):
            exec_sum = ""
    if not exec_sum:
        if ak:
            if st.button("🤖 Generate executive summary with Claude AI",
                          use_container_width=True):
                with st.spinner("Generating executive summary…"):
                    try:
                        text = interpret_full_project(ak, proj)
                        proj["executive_summary"] = text
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            proj_sum = st.text_area("Write your own executive summary (optional)",
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
