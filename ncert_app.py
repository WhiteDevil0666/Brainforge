# ================================================================
# BrainForge — NCERT Chat (Class 8, 9, 10 | Maths & Science)
# FIXES APPLIED:
#   1. Sidebar always visible — fixed CSS width + left offset
#   2. Chat input pinned to bottom, left edge respects sidebar
#   3. Resend OTP — "to" as plain string, "from" plain email
#   4. Verbose OTP error logging for debugging
# ================================================================

import os, re, json, hashlib, datetime, random, string, resend
import streamlit as st
from groq import Groq
from supabase import create_client, Client

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

GROQ_MODEL         = "llama-3.3-70b-versatile"
MAX_HISTORY        = 6
DAILY_LIMIT        = 15
TOP_K              = 6
OTP_EXPIRY_MINUTES = 10

CLASSES       = ["Class 8", "Class 9", "Class 10"]
SUBJECTS      = ["Both", "Mathematics", "Science"]
SUBJECT_ICONS = {"Mathematics": "📐", "Science": "🔬", "Both": "📚"}
CLASS_ICONS   = {"Class 8": "8️⃣", "Class 9": "9️⃣", "Class 10": "🔟"}
CLASS_COLORS  = {"Class 8": "#7c3aed", "Class 9": "#0284c7", "Class 10": "#059669"}

CHAPTER_INDEX = {
    "Class 8": {
        "Mathematics": {
            "hemh101": "Chapter 1 — Rational Numbers",
            "hemh102": "Chapter 2 — Linear Equations in One Variable",
            "hemh103": "Chapter 3 — Understanding Quadrilaterals",
            "hemh104": "Chapter 4 — Data Handling",
            "hemh105": "Chapter 5 — Squares and Square Roots",
            "hemh106": "Chapter 6 — Cubes and Cube Roots",
            "hemh107": "Chapter 7 — Comparing Quantities",
            "hemh108": "Chapter 8 — Algebraic Expressions and Identities",
            "hemh109": "Chapter 9 — Mensuration",
            "hemh110": "Chapter 10 — Exponents and Powers",
            "hemh111": "Chapter 11 — Direct and Inverse Proportions",
            "hemh112": "Chapter 12 — Factorisation",
            "hemh113": "Chapter 13 — Introduction to Graphs",
        },
        "Science": {
            "hesc101": "Chapter 1 — Crop Production and Management",
            "hesc102": "Chapter 2 — Microorganisms: Friend and Foe",
            "hesc103": "Chapter 3 — Synthetic Fibres and Plastics",
            "hesc104": "Chapter 4 — Materials: Metals and Non-Metals",
            "hesc105": "Chapter 5 — Coal and Petroleum",
            "hesc106": "Chapter 6 — Combustion and Flame",
            "hesc107": "Chapter 7 — Conservation of Plants and Animals",
            "hesc108": "Chapter 8 — Cell Structure and Functions",
            "hesc109": "Chapter 9 — Reproduction in Animals",
            "hesc110": "Chapter 10 — Reaching the Age of Adolescence",
            "hesc111": "Chapter 11 — Force and Pressure",
            "hesc112": "Chapter 12 — Friction",
            "hesc113": "Chapter 13 — Sound",
            "hesc1ps": "Chapter 14 — Chemical Effects of Electric Current",
        },
    },
    "Class 9": {
        "Mathematics": {
            "iemh101": "Chapter 1 — Number Systems",
            "iemh102": "Chapter 2 — Polynomials",
            "iemh103": "Chapter 3 — Coordinate Geometry",
            "iemh104": "Chapter 4 — Linear Equations in Two Variables",
            "iemh105": "Chapter 5 — Introduction to Euclid's Geometry",
            "iemh106": "Chapter 6 — Lines and Angles",
            "iemh107": "Chapter 7 — Triangles",
            "iemh108": "Chapter 8 — Quadrilaterals",
            "iemh109": "Chapter 9 — Circles",
            "iemh110": "Chapter 10 — Heron's Formula",
            "iemh111": "Chapter 11 — Surface Areas and Volumes",
            "iemh112": "Chapter 12 — Statistics",
        },
        "Science": {
            "iesc101": "Chapter 1 — Matter in Our Surroundings",
            "iesc102": "Chapter 2 — Is Matter Around Us Pure?",
            "iesc103": "Chapter 3 — Atoms and Molecules",
            "iesc104": "Chapter 4 — Structure of the Atom",
            "iesc105": "Chapter 5 — The Fundamental Unit of Life",
            "iesc106": "Chapter 6 — Tissues",
            "iesc107": "Chapter 7 — Motion",
            "iesc108": "Chapter 8 — Force and Laws of Motion",
            "iesc109": "Chapter 9 — Gravitation",
            "iesc110": "Chapter 10 — Work and Energy",
            "iesc111": "Chapter 11 — Sound",
            "iesc112": "Chapter 12 — Improvement in Food Resources",
        },
    },
    "Class 10": {
        "Mathematics": {
            "jemh101": "Chapter 1 — Real Numbers",
            "jemh102": "Chapter 2 — Polynomials",
            "jemh103": "Chapter 3 — Pair of Linear Equations in Two Variables",
            "jemh104": "Chapter 4 — Quadratic Equations",
            "jemh105": "Chapter 5 — Arithmetic Progressions",
            "jemh106": "Chapter 6 — Triangles",
            "jemh107": "Chapter 7 — Coordinate Geometry",
            "jemh108": "Chapter 8 — Introduction to Trigonometry",
            "jemh109": "Chapter 9 — Some Applications of Trigonometry",
            "jemh110": "Chapter 10 — Circles",
            "jemh111": "Chapter 11 — Areas Related to Circles",
            "jemh112": "Chapter 12 — Surface Areas and Volumes",
            "jemh113": "Chapter 13 — Statistics",
            "jemh114": "Chapter 14 — Probability",
        },
        "Science": {
            "jesc101": "Chapter 1 — Chemical Reactions and Equations",
            "jesc102": "Chapter 2 — Acids, Bases and Salts",
            "jesc103": "Chapter 3 — Metals and Non-Metals",
            "jesc104": "Chapter 4 — Carbon and Its Compounds",
            "jesc105": "Chapter 5 — Life Processes",
            "jesc106": "Chapter 6 — Control and Coordination",
            "jesc107": "Chapter 7 — How do Organisms Reproduce?",
            "jesc108": "Chapter 8 — Heredity",
            "jesc109": "Chapter 9 — Light — Reflection and Refraction",
            "jesc110": "Chapter 10 — The Human Eye and the Colourful World",
            "jesc111": "Chapter 11 — Electricity",
            "jesc112": "Chapter 12 — Magnetic Effects of Electric Current",
            "jesc113": "Chapter 13 — Our Environment",
        },
    },
}

SUGGESTIONS = {
    "Class 8": {
        "Both":        ["What is photosynthesis?","Rational numbers?","Cell structure?","What is friction?","Metals vs non-metals?","Linear equations?"],
        "Mathematics": ["Rational numbers?","Linear equations?","Factorisation?","Area of trapezium?","Algebraic expressions?","Comparing quantities?"],
        "Science":     ["Photosynthesis?","Cell structure?","Friction?","How does sound travel?","Microorganisms?","Force and pressure?"],
    },
    "Class 9": {
        "Both":        ["Irrational numbers?","Newton's laws?","Polynomials?","Gravitation?","Tissues?","Coordinate geometry?"],
        "Mathematics": ["Irrational numbers?","Heron's formula?","Coordinate geometry?","Polynomials?","Euclid's geometry?","Lines and angles?"],
        "Science":     ["Newton's three laws?","Gravitation?","Atoms vs molecules?","Tissues?","Sound production?","Work and energy?"],
    },
    "Class 10": {
        "Both":        ["Real numbers?","Chemical reactions?","Trigonometry?","Acids vs bases?","Probability?","Light reflection?"],
        "Mathematics": ["Real numbers?","Quadratic equations?","Trigonometry?","Arithmetic progressions?","Surface area of cone?","Probability?"],
        "Science":     ["Chemical reactions?","Acids vs bases?","Heredity?","Human eye?","Electricity?","Life processes?"],
    },
}

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BrainForge — NCERT AI Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
# CSS  — FIX 1 (sidebar) + FIX 2 (chat input pinned correctly)
# ════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

:root {
  --bg:       #06070f;
  --surface:  #0d1117;
  --border:   rgba(255,255,255,0.07);
  --accent:   #7c3aed;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --success:  #10b981;
  --warn:     #f59e0b;
  --danger:   #ef4444;
  --radius:   14px;
  --font:     'Space Grotesk', sans-serif;
  --sidebar-w: 260px;   /* must match Streamlit's actual sidebar width */
}

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}

/* Hide only what's safe — do NOT hide the sidebar collapse button */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── SIDEBAR: style only, never override width/display/position ── */
/* Streamlit controls sidebar open/close via JS — we must not fight it */
section[data-testid="stSidebar"] {
  background: #080b14 !important;
  border-right: 1px solid rgba(124,58,237,0.18) !important;
}
/* Sidebar content text */
section[data-testid="stSidebar"] > div {
  background: #080b14 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
  color: var(--text) !important;
  font-family: var(--font) !important;
}
section[data-testid="stSidebar"] .stButton > button {
  background: rgba(124,58,237,0.14) !important;
  border: 1px solid rgba(124,58,237,0.28) !important;
  color: #a78bfa !important;
}

/* ── Sidebar collapse/expand toggle button — ALWAYS visible ── */
button[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  background: rgba(124,58,237,0.2) !important;
  border: 1px solid rgba(124,58,237,0.4) !important;
  border-radius: 8px !important;
  color: #a78bfa !important;
}

/* ── Main content ── */
.block-container {
  padding: 1rem 1.5rem 120px 1.5rem !important;
  max-width: 100% !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important;
  color: #fff !important; border: none !important;
  border-radius: 10px !important;
  font-family: var(--font) !important;
  font-weight: 700 !important; font-size: 0.85rem !important;
  padding: 0.5rem 1rem !important;
  transition: all 0.15s !important; width: 100% !important;
}
.stButton > button:hover { background: #6d28d9 !important; transform: translateY(-1px) !important; }
.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.05) !important;
  color: var(--text) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input,
.stTextArea textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(124,58,237,0.18) !important;
}

/* ── Hide Streamlit chat avatars ── */
div[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
  gap: 0 !important;
}
div[data-testid="stChatMessage"] > div:first-child { display: none !important; }
div[data-testid="stChatMessageContent"] { padding: 0 !important; }

/* ── Chat bubbles ── */
.msg-wrap { display: flex; flex-direction: column; margin-bottom: 14px; gap: 3px; }
.msg-label { font-size: 0.65rem; font-weight: 700; color: var(--muted);
             text-transform: uppercase; letter-spacing: 0.06em; padding: 0 4px; }
.msg-label-right { text-align: right; }
.msg-user {
  align-self: flex-end;
  background: linear-gradient(135deg, rgba(124,58,237,0.22), rgba(124,58,237,0.12));
  border: 1px solid rgba(124,58,237,0.3);
  border-radius: 16px 16px 4px 16px;
  padding: 11px 15px; max-width: 82%;
  font-size: 0.88rem; line-height: 1.55; color: #e2e8f0;
}
.msg-bot {
  align-self: flex-start;
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 4px 16px 16px 16px;
  padding: 14px 18px; max-width: 94%;
  font-size: 0.87rem; line-height: 1.65; color: #e2e8f0;
}
.msg-bot strong { color: #a78bfa; }
.msg-bot h2,.msg-bot h3,.msg-bot h4 {
  color: #f1f5f9 !important; font-family: var(--font) !important; margin: 10px 0 5px !important;
}

/* ── Chat input: pinned to bottom, width follows Streamlit's main column ── */
/* Do NOT set left — that breaks the sidebar layout engine on Streamlit Cloud */
div[data-testid="stChatInput"] {
  position: fixed !important;
  bottom: 0 !important;
  right: 0 !important;
  z-index: 999 !important;
  background: linear-gradient(to top, #06070f 70%, transparent) !important;
  padding: 14px 32px 18px 32px !important;
  width: auto !important;
}
div[data-testid="stChatInput"] textarea {
  background: #0d1117 !important;
  border: 1px solid rgba(124,58,237,0.45) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  font-size: 0.88rem !important; padding: 12px 16px !important;
}
div[data-testid="stChatInput"] textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important; border-radius: 10px !important;
  padding: 4px !important; gap: 3px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important; color: var(--muted) !important;
  font-family: var(--font) !important; font-weight: 600 !important; font-size: 0.78rem !important;
}
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: #fff !important; }

/* ── Radio ── */
.stRadio > div { gap: 4px !important; }
.stRadio > div > label {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important; border-radius: 8px !important;
  padding: 6px 12px !important; font-size: 0.83rem !important;
  color: var(--text) !important; text-transform: none !important; letter-spacing: 0 !important;
}
.stRadio > div > label:has(input:checked) {
  border-color: var(--accent) !important;
  background: rgba(124,58,237,0.12) !important; color: #a78bfa !important;
}

/* ── Labels ── */
label, div[data-testid="stWidgetLabel"] p {
  color: var(--muted) !important; font-size: 0.73rem !important;
  font-weight: 700 !important; text-transform: uppercase !important;
  letter-spacing: 0.05em !important; font-family: var(--font) !important;
}

/* ── Expanders ── */
details {
  background: var(--surface) !important; border-radius: 10px !important;
  border: 1px solid var(--border) !important;
}
details summary { color: var(--text) !important; font-family: var(--font) !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; color: var(--text) !important;
}

/* ── Utility ── */
.bf-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 18px; margin-bottom: 10px;
  transition: border-color 0.2s;
}
.bf-card:hover { border-color: rgba(124,58,237,0.32); }

.pill {
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.24);
  border-radius: 99px; padding: 3px 10px;
  font-size: 0.68rem; font-weight: 700; color: #a78bfa;
  margin-right: 4px; margin-bottom: 4px;
}
.pill.green { background:rgba(16,185,129,0.1);  border-color:rgba(16,185,129,0.24); color:#6ee7b7; }
.pill.blue  { background:rgba(6,182,212,0.1);   border-color:rgba(6,182,212,0.24);  color:#67e8f9; }
.pill.warn  { background:rgba(245,158,11,0.1);  border-color:rgba(245,158,11,0.24); color:#fcd34d; }
.pill.red   { background:rgba(239,68,68,0.1);   border-color:rgba(239,68,68,0.24);  color:#fca5a5; }

.rate-bar-wrap { background:rgba(255,255,255,0.05); border-radius:99px; height:5px; margin:5px 0; overflow:hidden; }
.rate-bar { height:5px; border-radius:99px; transition:width 0.4s; }

.src-card {
  border-left: 3px solid var(--accent); background: rgba(255,255,255,0.02);
  border-radius: 0 10px 10px 0; padding: 9px 12px; margin-bottom: 7px; font-size: 0.78rem;
}
.note-card {
  background: rgba(6,182,212,0.04); border: 1px solid rgba(6,182,212,0.18);
  border-radius: 12px; padding: 13px 15px; margin-bottom: 10px;
}
.note-ts   { font-size: 0.66rem; color: var(--muted); margin-bottom: 4px; }
.note-ch   { font-size: 0.7rem; color: #67e8f9; font-weight: 700; margin-bottom: 5px; }
.note-body { font-size: 0.83rem; color: var(--text); line-height: 1.55; }

.quiz-opt {
  background: rgba(255,255,255,0.04); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
  font-size: 0.86rem; color: var(--text);
}
.quiz-opt.correct   { border-color: var(--success) !important; background: rgba(16,185,129,0.1) !important; color: #6ee7b7 !important; }
.quiz-opt.incorrect { border-color: var(--danger)  !important; background: rgba(239,68,68,0.1)  !important; color: #fca5a5 !important; }

/* ── Mobile ── */
@media (max-width: 768px) {
  .block-container { padding: 0.5rem 0.75rem 120px 0.75rem !important; }
  div[data-testid="stChatInput"] { padding: 14px 16px 18px !important; }
  .msg-user, .msg-bot { max-width: 96% !important; }
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.38); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# SECRETS & CLIENTS
# ════════════════════════════════════════════════════════════════

def _secret(k):
    try:    return st.secrets[k]
    except: return os.getenv(k, "")

GROQ_API_KEY = _secret("GROQ_API_KEY")
SUPABASE_URL = _secret("SUPABASE_URL")
SUPABASE_KEY = _secret("SUPABASE_KEY")

missing = [k for k,v in {"GROQ_API_KEY":GROQ_API_KEY,"SUPABASE_URL":SUPABASE_URL,"SUPABASE_KEY":SUPABASE_KEY}.items() if not v]
if missing:
    st.error(f"Missing secrets: {', '.join(missing)}")
    st.code("\n".join(f'{k} = "..."' for k in missing), language="toml")
    st.stop()

@st.cache_resource
def get_groq():  return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def get_sb(): return create_client(SUPABASE_URL, SUPABASE_KEY)

groq_client = get_groq()
sb: Client  = get_sb()

# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════

def _init():
    for k, v in {
        "auth_stage": "input", "auth_identifier": "",
        "auth_user_id": None, "auth_otp_time": None,
        "messages": [], "selected_chapter": None, "chapter_mode": False,
        "selected_class": "Class 8", "selected_subject": "Both",
        "quiz_state": None, "quiz_answered": False,
        "quiz_last_pick": "", "usage": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ════════════════════════════════════════════════════════════════

def _norm(s): return s.strip().lower()

def _validate(val):
    v = _norm(val)
    if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v): return True, "email"
    if 10 <= len(re.sub(r"\D","",v)) <= 13:          return True, "phone"
    return False, ""

def _otp(): return "".join(random.choices(string.digits, k=6))

def get_or_create_user(identifier):
    v = _norm(identifier)
    try:
        r = sb.table("users").select("id").eq("identifier", v).execute()
        if r.data: return r.data[0]["id"]
        r2 = sb.table("users").insert({"identifier": v}).execute()
        return r2.data[0]["id"]
    except Exception as e:
        st.error(f"DB error: {e}"); return None

def store_otp(identifier, otp):
    try:
        sb.table("otps").insert({"identifier": _norm(identifier), "otp": otp}).execute()
    except Exception as e:
        st.error(f"OTP store error: {e}")

def verify_otp(identifier, otp):
    try:
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat()
        r = (sb.table("otps").select("id,otp,used")
               .eq("identifier", _norm(identifier)).eq("used", False)
               .gte("created_at", cutoff).order("created_at", desc=True).limit(1).execute())
        if not r.data: return False
        row = r.data[0]
        if row["otp"] == otp.strip():
            sb.table("otps").update({"used": True}).eq("id", row["id"]).execute()
            return True
        return False
    except Exception as e:
        st.error(f"OTP verify error: {e}")
        return False

# ════════════════════════════════════════════════════════════════
# EMAIL OTP — FIX: plain string "to" + plain "from" + verbose errors
# ════════════════════════════════════════════════════════════════

RESEND_API_KEY = _secret("RESEND_API_KEY")

if not RESEND_API_KEY:
    st.error("Missing RESEND_API_KEY in secrets")
    st.stop()

resend.api_key = RESEND_API_KEY

def send_otp_email(to_email: str, otp: str) -> bool:
    """
    Send OTP email via Resend.
    FIX 1: 'to' must be a plain string (not a list) for free-tier Resend accounts.
    FIX 2: 'from' must be a plain email without a display name when using onboarding@resend.dev.
    FIX 3: Print full error to server logs so it's visible in Streamlit Cloud.
    """
    try:
        result = resend.Emails.send({
            "from":    "onboarding@resend.dev",   # plain email, no display name
            "to":      to_email.strip(),           # plain string, NOT a list
            "subject": "Your BrainForge OTP 🔐",
            "html": f"""
                <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;
                            background:#06070f;color:#e2e8f0;padding:32px;border-radius:12px;">
                  <h2 style="color:#7c3aed;margin-bottom:8px;">🧠 BrainForge</h2>
                  <p style="font-size:15px;color:#94a3b8;">Your one-time login code:</p>
                  <div style="font-size:44px;font-weight:900;color:#7c3aed;
                       letter-spacing:12px;margin:20px 0;text-align:center;">{otp}</div>
                  <p style="color:#64748b;font-size:13px;">
                    Valid for {OTP_EXPIRY_MINUTES} minutes. Do not share this code with anyone.
                  </p>
                  <hr style="border:none;border-top:1px solid #1e293b;margin:24px 0;">
                  <p style="color:#475569;font-size:11px;">
                    If you didn't request this, you can safely ignore this email.
                  </p>
                </div>
            """,
        })
        # Log result to server console (visible in Streamlit Cloud → Logs)
        print(f"[Resend OK] to={to_email} | result={result}")
        return True
    except Exception as e:
        # Full error in both UI and server logs
        print(f"[Resend ERROR] to={to_email} | error={e}")
        st.error(f"❌ Email failed: {e}")
        return False

# ════════════════════════════════════════════════════════════════
# AUTH SCREEN
# ════════════════════════════════════════════════════════════════

def render_auth():
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid rgba(124,58,237,0.28);border-radius:20px;
             padding:36px 30px;text-align:center;margin-top:60px;">
          <div style="font-size:2.8rem;">🧠</div>
          <div style="font-size:1.35rem;font-weight:800;color:#e2e8f0;margin:10px 0 4px;">BrainForge</div>
          <div style="font-size:0.8rem;color:#64748b;margin-bottom:24px;">
            NCERT AI Tutor · Class 8, 9 &amp; 10<br>
            Sign in to get <strong style="color:#a78bfa;">15 free questions daily</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        # ── INPUT STAGE ──
        if st.session_state.auth_stage == "input":
            identifier = st.text_input(
                "📧 Email address",
                placeholder="you@example.com",
                key="auth_id_input",
            )
            if st.button("Send OTP →", type="primary", use_container_width=True):
                valid, kind = _validate(identifier)
                if not valid:
                    st.error("Enter a valid email address.", icon="❌")
                elif kind == "phone":
                    st.error("📱 SMS OTP not supported yet. Please use email.", icon="⚠️")
                else:
                    otp = _otp()
                    store_otp(identifier, otp)
                    success = send_otp_email(identifier.strip(), otp)
                    if success:
                        st.success("📩 OTP sent! Check your inbox (and spam folder).")
                        st.session_state.auth_identifier = identifier.strip()
                        st.session_state.auth_otp_time   = datetime.datetime.utcnow()
                        st.session_state.auth_stage      = "otp"
                        st.rerun()

        # ── OTP VERIFY STAGE ──
        elif st.session_state.auth_stage == "otp":
            ident = st.session_state.auth_identifier
            st.info(f"OTP sent to **{ident}** — check spam if not in inbox.")
            otp_in = st.text_input("Enter 6-digit OTP", max_chars=6,
                                   placeholder="······", key="otp_in")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Verify", type="primary", use_container_width=True):
                    if verify_otp(ident, otp_in):
                        uid = get_or_create_user(ident)
                        if uid:
                            st.session_state.auth_user_id = uid
                            st.session_state.auth_stage   = "done"
                            st.session_state.usage        = None
                            st.rerun()
                        else:
                            st.error("Account error. Try again.", icon="❌")
                    else:
                        st.error("Wrong or expired OTP.", icon="❌")
            with c2:
                if st.button("🔄 Resend OTP", use_container_width=True):
                    new_otp = _otp()
                    store_otp(ident, new_otp)
                    if send_otp_email(ident, new_otp):
                        st.success("📩 New OTP sent!")
                    st.session_state.auth_otp_time = datetime.datetime.utcnow()
            if st.button("← Change email", use_container_width=True):
                st.session_state.auth_stage = "input"
                st.rerun()

        # ── DEBUG expander (remove before going to production) ──
        with st.expander("🔧 Debug: Test email delivery"):
            test_addr = st.text_input("Send test OTP to:", key="dbg_email")
            if st.button("📨 Send Test", key="dbg_send"):
                if test_addr.strip():
                    ok = send_otp_email(test_addr.strip(), "999888")
                    if ok:
                        st.success(f"Sent! Check {test_addr}. Code: 999888")
                else:
                    st.warning("Enter an email address above.")


if st.session_state.auth_stage != "done":
    render_auth()
    st.stop()

# ════════════════════════════════════════════════════════════════
# RATE LIMITING
# ════════════════════════════════════════════════════════════════

def get_usage():
    try:
        uid   = st.session_state.auth_user_id
        today = datetime.date.today().isoformat()
        r = sb.table("rate_limits").select("count").eq("user_id", uid).eq("day", today).execute()
        return r.data[0]["count"] if r.data else 0
    except: return 0

def increment_usage():
    try:
        uid   = st.session_state.auth_user_id
        today = datetime.date.today().isoformat()
        r = sb.table("rate_limits").select("count").eq("user_id", uid).eq("day", today).execute()
        if r.data:
            new = r.data[0]["count"] + 1
            sb.table("rate_limits").update({"count": new}).eq("user_id", uid).eq("day", today).execute()
        else:
            new = 1
            sb.table("rate_limits").insert({"user_id": uid, "day": today, "count": 1}).execute()
        return new
    except: return 0

def rate_limit_check():
    if st.session_state.usage is None:
        st.session_state.usage = get_usage()
    if st.session_state.usage >= DAILY_LIMIT:
        st.error(f"🚫 You've used all **{DAILY_LIMIT} daily questions**. Come back tomorrow! 🌅", icon="🚫")
        return False
    return True

# ════════════════════════════════════════════════════════════════
# VECTOR SEARCH
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except: return None

def embed(text):
    m = get_embedder()
    return m.encode(text, normalize_embeddings=True).tolist() if m else None

def retrieve_chunks(query, class_label, subject, chapter_filter="", top_k=TOP_K):
    q_emb = embed(query)
    if not q_emb: return []
    try:
        r = sb.rpc("match_chunks", {
            "query_embedding": q_emb,
            "filter_class":    class_label,
            "filter_subject":  subject,
            "filter_chapter":  chapter_filter or "",
            "match_count":     top_k,
        }).execute()
        out = []
        for row in (r.data or []):
            sim = float(row.get("similarity", 0))
            if sim > 0.15:
                out.append({
                    "text":row["content"],"subject":row["subject"],
                    "chapter":row["chapter"],"page":row.get("page","?"),
                    "source":row.get("source",""),"class":row["class"],
                    "relevance":round(sim*100,1),
                })
        return sorted(out, key=lambda x: x["relevance"], reverse=True)
    except Exception as e:
        st.warning(f"Search error: {e}", icon="⚠️"); return []

# ════════════════════════════════════════════════════════════════
# NOTES
# ════════════════════════════════════════════════════════════════

def save_note(content, cls, subj, chapter=""):
    try:
        sb.table("notes").insert({
            "user_id":st.session_state.auth_user_id,
            "class":cls,"subject":subj,"chapter":chapter,"content":content,
        }).execute()
        return True
    except: return False

def get_notes(cls=None):
    try:
        q = (sb.table("notes").select("*")
               .eq("user_id", st.session_state.auth_user_id)
               .order("created_at", desc=True))
        if cls: q = q.eq("class", cls)
        return (q.limit(50).execute()).data or []
    except: return []

def delete_note(nid):
    try:
        sb.table("notes").delete().eq("id", nid).eq("user_id", st.session_state.auth_user_id).execute()
        return True
    except: return False

# ════════════════════════════════════════════════════════════════
# AI GENERATION
# ════════════════════════════════════════════════════════════════

def build_ctx(chunks):
    if not chunks: return ""
    out = "\n--- NCERT CONTENT ---\n"
    for c in chunks[:6]:
        out += f"\n[{c['class']}|{c['subject']}|{c['chapter']}|p.{c['page']}]\n{c['text']}\n"
    return out + "\n--- END ---\n"

def generate_answer(question, chunks, cls, subj, style, history, ch_ctx=None):
    age = {"Class 8":"13–14","Class 9":"14–15","Class 10":"15–16"}.get(cls,"13–16")
    style_instr = {
        "Simple":   "Simple language, short paragraphs.",
        "Detailed": "Comprehensive, cover all sub-topics.",
        "Bullets":  "Use numbered headings and bullet points.",
        "Examples": "Include 2–3 real Indian examples.",
    }.get(style,"Simple language.")
    ch_line = f"\nStudent studying: {ch_ctx}" if ch_ctx else ""
    system = f"""You are BrainForge — expert CBSE AI tutor for India.
Read NCERT snippets. Rewrite as clear, structured, student-friendly teaching.
Enrich with analogies, memory tricks, Indian examples.
STYLE: {style_instr}  Bold key terms. Add 💡 Quick Tip where helpful.
STUDENT: {cls} CBSE, age {age}{ch_line}
FORMAT: ## [Topic] / Explanation / **Key Points:** - ... / 💡 Quick Tip: ... / 📚 NCERT {cls} {subj}
End with: 💬 Want examples, a quiz, or deeper explanation?"""
    msgs = [{"role":"system","content":system}]
    for m in history[-(MAX_HISTORY*2):]:
        if m["role"] in ("user","assistant"):
            msgs.append({"role":m["role"],"content":m["content"]})
    msgs.append({"role":"user","content":f"Q: {question}\n{build_ctx(chunks)}\nAnswer for {cls} student."})
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=msgs, temperature=0.4, max_tokens=1400)
        return r.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        return ("⚠️ AI rate limit hit. Wait 30 s and retry."
                if "rate_limit" in err.lower() or "429" in err
                else f"⚠️ Error: {err}")

def generate_quiz(topic, cls, subj, chunks):
    prompt = f"""CBSE exam setter. Generate ONE MCQ on "{topic}" | {cls} | {subj}
{build_ctx(chunks[:4])}
Reply ONLY valid JSON (no markdown):
{{"question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"correct":"A","explanation":"..."}}"""
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3, max_tokens=400,
        )
        raw = re.sub(r"```json|```","",r.choices[0].message.content.strip()).strip()
        return json.loads(raw)
    except: return None

def generate_chapter_summary(ch_key, ch_title, cls, subj):
    chunks = retrieve_chunks(ch_title, cls, subj, ch_key, top_k=8)
    ctx = "\n".join(c["text"] for c in chunks[:6])
    prompt = f"""Expert {cls} CBSE teacher. Student opened: {ch_title} ({cls}·{subj})
NCERT: {ctx}
Write chapter overview:
## 📖 {ch_title}
### What You Will Learn [3–5 bullets]
### Key Concepts [4–6 concepts]
### Why This Chapter Matters [1–2 lines]
### Quick Preview [2–3 sentences]
End: "💬 Ask me anything about this chapter!" """
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.4, max_tokens=800,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not load chapter overview: {e}"

def is_followup(q, history):
    if not history: return False
    ql = q.lower().strip()
    if len(ql.split()) <= 5:
        for p in [
            r"^(what about|tell me more|explain more|can you|how about|give me|more about)",
            r"^(example|examples|illustrate|show me)",
            r"^(i (don't|do not) understand|unclear|confused|simpler)",
            r"^(test me|quiz me|ask me|mcq)",
        ]:
            if re.match(p, ql): return True
    return False

def process_question(question, ch_ctx=None, ch_filter="", ch_subj=None):
    if not rate_limit_check(): return None, []
    cls_   = st.session_state.selected_class
    subj_  = ch_subj or st.session_state.selected_subject
    style  = st.session_state.get("answer_depth","Simple")
    hist   = st.session_state.messages
    chunks = []
    if not is_followup(question, hist):
        with st.spinner("🔍 Searching NCERT…"):
            chunks = retrieve_chunks(question, cls_, subj_, ch_filter, TOP_K)
    with st.spinner("✍️ Writing answer…"):
        answer = generate_answer(question, chunks, cls_, subj_, style, hist, ch_ctx)
    st.session_state.usage = increment_usage()
    return answer, chunks

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    ident_short = st.session_state.auth_identifier
    if len(ident_short) > 22: ident_short = ident_short[:20] + "…"
    st.markdown(f"""
    <div style="padding:8px 0 14px;border-bottom:1px solid rgba(255,255,255,0.07);">
      <div style="font-size:1.15rem;font-weight:800;color:#e2e8f0;">🧠 BrainForge</div>
      <div style="font-size:0.68rem;color:#64748b;margin-top:2px;">NCERT Class 8 · 9 · 10</div>
      <div style="margin-top:7px;font-size:0.7rem;color:#a78bfa;font-weight:600;">👤 {ident_short}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎓 Class")
    sel_cls = st.radio("cls", CLASSES,
                       format_func=lambda x: f"{CLASS_ICONS[x]} {x}",
                       label_visibility="collapsed", key="cls_radio")
    if sel_cls != st.session_state.selected_class:
        st.session_state.update(selected_class=sel_cls, selected_chapter=None,
                                chapter_mode=False, messages=[], quiz_state=None)

    st.markdown("#### 📚 Subject")
    sel_subj = st.radio("subj", SUBJECTS,
                        format_func=lambda x: f"{SUBJECT_ICONS[x]} {x}",
                        label_visibility="collapsed", key="subj_radio")
    if sel_subj != st.session_state.selected_subject:
        st.session_state.update(selected_subject=sel_subj, selected_chapter=None,
                                chapter_mode=False, messages=[], quiz_state=None)

    st.markdown("---")

    if st.session_state.usage is None: st.session_state.usage = get_usage()
    used  = st.session_state.usage
    pct   = min(used / DAILY_LIMIT, 1.0)
    bar_c = ("linear-gradient(90deg,#7c3aed,#06b6d4)" if pct < 0.8
             else "linear-gradient(90deg,#f59e0b,#ef4444)")
    pill_c = "green" if pct < 0.6 else ("warn" if pct < 0.9 else "red")
    st.markdown(f"""
    <div style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <span style="font-size:0.7rem;color:#94a3b8;font-weight:600;">Daily Usage</span>
        <span class="pill {pill_c}">{used}/{DAILY_LIMIT}</span>
      </div>
      <div class="rate-bar-wrap">
        <div class="rate-bar" style="width:{int(pct*100)}%;background:{bar_c};"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    show_src     = st.toggle("📄 Show NCERT sources", value=False, key="show_src")
    answer_depth = st.selectbox("Answer style", ["Simple","Detailed","Bullets","Examples"],
                                key="answer_depth")

    msgs_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    if msgs_count:
        st.markdown(
            f'<div class="pill green">💬 {msgs_count} question{"s" if msgs_count!=1 else ""}</div>',
            unsafe_allow_html=True,
        )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.update(messages=[], selected_chapter=None,
                                chapter_mode=False, quiz_state=None)
        st.rerun()

    if st.button("🚪 Sign Out", use_container_width=True):
        for k in ["auth_stage","auth_identifier","auth_user_id","auth_otp_time",
                  "messages","usage","quiz_state","selected_chapter"]:
            st.session_state[k] = None
        st.session_state.auth_stage = "input"
        st.rerun()

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

cls   = st.session_state.selected_class
subj  = st.session_state.selected_subject
c_col = CLASS_COLORS.get(cls, "#7c3aed")

st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(124,58,237,0.09),rgba(6,182,212,0.04));
     border:1px solid {c_col}30;border-radius:14px;padding:12px 18px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <span style="font-size:1.5rem;">{CLASS_ICONS.get(cls,"🎓")}</span>
    <div>
      <div style="font-size:1rem;font-weight:800;color:#e2e8f0;">NCERT AI Tutor</div>
      <div style="font-size:0.72rem;color:#64748b;">{cls} · {subj} · NCERT-grounded answers</div>
    </div>
    <div style="margin-left:auto;">
      <span class="pill">{CLASS_ICONS.get(cls,"")} {cls}</span>
      <span class="pill blue">{SUBJECT_ICONS.get(subj,"")} {subj}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab_chat, tab_idx, tab_quiz, tab_notes = st.tabs(["💬 Chat","📖 Chapters","🎯 Quiz","🗒️ Notes"])

# ── Shared source renderer ───────────────────────────────────

def render_sources(chunks):
    if not chunks or not st.session_state.get("show_src"): return
    with st.expander(f"📄 {len(chunks)} NCERT passages"):
        for c in chunks:
            rel = c["relevance"]
            bc  = "#10b981" if rel >= 70 else "#f59e0b" if rel >= 45 else "#64748b"
            st.markdown(f"""
            <div class="src-card" style="border-left-color:{bc};">
              <div style="display:flex;justify-content:space-between;">
                <span style="color:#a78bfa;font-weight:700;font-size:0.72rem;">
                  {CLASS_ICONS.get(c.get('class',''),'')} {c.get('class','')} ·
                  {SUBJECT_ICONS.get(c['subject'],'')} {c['subject']} — {c['chapter'].upper()}
                </span>
                <span style="color:{bc};font-size:0.68rem;font-weight:700;">{rel}%</span>
              </div>
              <div style="color:#64748b;font-size:0.66rem;margin:3px 0;">p.{c['page']} · {c['source']}</div>
              <div style="color:#cbd5e1;font-size:0.76rem;line-height:1.5;">
                {c['text'][:230]}{'…' if len(c['text'])>230 else ''}
              </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════

with tab_chat:
    ch_ctx    = None
    ch_filter = ""
    ch_subj_  = subj

    if st.session_state.chapter_mode and st.session_state.selected_chapter:
        ch_key, ch_s = st.session_state.selected_chapter
        ch_title = CHAPTER_INDEX.get(cls,{}).get(ch_s,{}).get(ch_key, ch_key)
        ch_ctx, ch_filter, ch_subj_ = ch_title, ch_key, ch_s

        b1, b2 = st.columns([5, 1])
        with b1:
            st.markdown(f"""
            <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.22);
                 border-radius:10px;padding:8px 14px;margin-bottom:8px;">
              <div style="font-size:0.66rem;color:#a78bfa;font-weight:700;">📖 Chapter Mode</div>
              <div style="font-size:0.83rem;color:#e2e8f0;font-weight:600;">{ch_title}</div>
            </div>""", unsafe_allow_html=True)
        with b2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✕ Exit", key="exit_ch"):
                st.session_state.update(chapter_mode=False, selected_chapter=None, messages=[])
                st.rerun()

    if not st.session_state.messages:
        st.markdown("##### 💡 Try asking:")
        suggs = SUGGESTIONS.get(cls,{}).get(subj, SUGGESTIONS["Class 8"]["Both"])
        c1, c2 = st.columns(2)
        for i, s in enumerate(suggs):
            with (c1 if i % 2 == 0 else c2):
                if st.button(s, key=f"sg{i}"):
                    answer, chunks = process_question(s, ch_ctx, ch_filter, ch_subj_)
                    if answer:
                        st.session_state.messages += [
                            {"role":"user",      "content":s,      "chunks":[]},
                            {"role":"assistant", "content":answer, "chunks":chunks},
                        ]
                    st.rerun()
        st.markdown("")

    # Render chat history
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-wrap">
              <div class="msg-label msg-label-right">You</div>
              <div style="display:flex;justify-content:flex-end;">
                <div class="msg-user">{msg['content']}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="msg-label">🧠 BrainForge</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown(
                    '<div style="background:rgba(255,255,255,0.035);'
                    'border:1px solid rgba(255,255,255,0.08);'
                    'border-radius:4px 16px 16px 16px;padding:14px 18px;margin-bottom:4px;">',
                    unsafe_allow_html=True,
                )
                st.markdown(msg["content"])
                st.markdown('</div>', unsafe_allow_html=True)
            render_sources(msg.get("chunks", []))
            sv_col, _ = st.columns([1, 5])
            with sv_col:
                if st.button("🗒️ Save", key=f"sv_{idx}", help="Save as note"):
                    ok = save_note(msg["content"], cls, subj, ch_filter or "General")
                    st.toast("✅ Saved!" if ok else "❌ Could not save.")

    # Chat input — CSS pins it to bottom, left edge respects sidebar
    ph = (f"Ask about {ch_title}…" if ch_ctx else
          f"Ask anything from {cls} {subj}…" if subj != "Both" else
          f"Ask anything from {cls} Maths or Science…")

    question = st.chat_input(ph)
    if question:
        st.markdown(f"""
        <div class="msg-wrap">
          <div class="msg-label msg-label-right">You</div>
          <div style="display:flex;justify-content:flex-end;">
            <div class="msg-user">{question}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        answer, chunks = process_question(question, ch_ctx, ch_filter, ch_subj_)
        if answer:
            st.markdown('<div class="msg-label">🧠 BrainForge</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown(
                    '<div style="background:rgba(255,255,255,0.035);'
                    'border:1px solid rgba(255,255,255,0.08);'
                    'border-radius:4px 16px 16px 16px;padding:14px 18px;margin-bottom:4px;">',
                    unsafe_allow_html=True,
                )
                st.markdown(answer)
                st.markdown('</div>', unsafe_allow_html=True)
            render_sources(chunks)
            st.session_state.messages += [
                {"role":"user",      "content":question, "chunks":[]},
                {"role":"assistant", "content":answer,   "chunks":chunks},
            ]

# ════════════════════════════════════════════════════════════════
# TAB 2 — CHAPTERS
# ════════════════════════════════════════════════════════════════

with tab_idx:
    subjs_show = ["Mathematics","Science"] if subj == "Both" else [subj]

    if st.session_state.selected_chapter and not st.session_state.chapter_mode:
        ch_key, ch_s = st.session_state.selected_chapter
        ch_title = CHAPTER_INDEX.get(cls,{}).get(ch_s,{}).get(ch_key, ch_key)

        if st.button("← Back", key="back_idx"):
            st.session_state.selected_chapter = None; st.rerun()

        st.markdown(f"""
        <div class="bf-card" style="border-color:{c_col}40;margin-bottom:12px;">
          <div style="font-size:0.66rem;color:#a78bfa;font-weight:700;margin-bottom:3px;">
            {CLASS_ICONS.get(cls,'')} {cls} · {SUBJECT_ICONS.get(ch_s,'')} {ch_s}
          </div>
          <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">{ch_title}</div>
        </div>""", unsafe_allow_html=True)

        with st.spinner("📖 Loading overview…"):
            summary = generate_chapter_summary(ch_key, ch_title, cls, ch_s)
        st.markdown(summary)
        st.divider()
        if st.button("💬 Chat about this Chapter", use_container_width=True, type="primary"):
            st.session_state.update(
                chapter_mode=True, selected_subject=ch_s,
                messages=[{"role":"assistant",
                           "content":f"📖 Ready! Ask anything about **{ch_title}** ({cls}·{ch_s}).",
                           "chunks":[]}],
            )
            st.rerun()
    else:
        for s in subjs_show:
            chapters = CHAPTER_INDEX.get(cls,{}).get(s,{})
            if not chapters: continue
            st.markdown(f"#### {SUBJECT_ICONS.get(s,'')} {cls} — {s}")
            for ch_key, ch_title in chapters.items():
                parts  = ch_title.split(" — ", 1)
                ch_num = parts[0] if len(parts) > 1 else ""
                ch_nm  = parts[1] if len(parts) > 1 else ch_title
                a, b = st.columns([6, 1])
                with a:
                    st.markdown(f"""
                    <div class="bf-card">
                      <div style="font-size:0.66rem;color:#7c3aed;font-weight:700;">{ch_num}</div>
                      <div style="font-size:0.86rem;font-weight:600;color:#e2e8f0;">{ch_nm}</div>
                    </div>""", unsafe_allow_html=True)
                with b:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("Open", key=f"ch_{cls}_{s}_{ch_key}"):
                        st.session_state.selected_chapter = (ch_key, s); st.rerun()
            st.markdown("")

# ════════════════════════════════════════════════════════════════
# TAB 3 — QUIZ
# ════════════════════════════════════════════════════════════════

with tab_quiz:
    st.markdown("#### 🎯 Quick Quiz")
    st.caption("AI-generated MCQs from NCERT content")
    quiz_topic = st.text_input("Topic",
                               placeholder="e.g. Photosynthesis, Newton's Laws, Trigonometry…",
                               key="qtopic")
    g1, _ = st.columns([1, 2])
    with g1:
        gen_btn = st.button("⚡ Generate Question", use_container_width=True, type="primary")

    if gen_btn:
        if not quiz_topic.strip():
            st.warning("Enter a topic first.", icon="⚠️")
        elif rate_limit_check():
            with st.spinner("Generating MCQ from NCERT…"):
                chunks = retrieve_chunks(quiz_topic, cls, subj, "", top_k=4)
                q_data = generate_quiz(quiz_topic, cls, subj, chunks)
            if q_data:
                st.session_state.quiz_state    = q_data
                st.session_state.quiz_answered = False
                st.session_state.usage = increment_usage()
            else:
                st.error("Could not generate quiz. Try a more specific topic.", icon="❌")

    if st.session_state.quiz_state:
        q            = st.session_state.quiz_state
        answered     = st.session_state.quiz_answered
        correct_letter = q.get("correct","A").upper()
        st.markdown(f"""
        <div class="bf-card" style="border-color:rgba(124,58,237,0.38);margin-top:10px;">
          <div style="font-size:0.66rem;color:#a78bfa;font-weight:700;margin-bottom:5px;">
            {CLASS_ICONS.get(cls,'')} {cls} · {SUBJECT_ICONS.get(subj,'')} {subj}
          </div>
          <div style="font-size:0.93rem;font-weight:700;color:#e2e8f0;line-height:1.5;">{q['question']}</div>
        </div>""", unsafe_allow_html=True)

        for opt in q["options"]:
            letter = opt[0].upper()
            if not answered:
                if st.button(opt, key=f"opt_{letter}", use_container_width=True):
                    st.session_state.quiz_answered = True
                    st.session_state.quiz_last_pick = letter
                    st.rerun()
            else:
                chosen = st.session_state.get("quiz_last_pick","")
                sty = "correct" if letter == correct_letter else ("incorrect" if letter == chosen else "")
                ico = "✅" if letter == correct_letter else ("❌" if letter == chosen else "")
                st.markdown(f'<div class="quiz-opt {sty}">{ico} {opt}</div>', unsafe_allow_html=True)

        if answered:
            chosen = st.session_state.get("quiz_last_pick","")
            if chosen == correct_letter: st.success("🎉 Correct! Well done.")
            else: st.error(f"The correct answer is **{correct_letter}**.")
            st.markdown(f"""
            <div style="background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.18);
                 border-radius:10px;padding:11px 15px;margin-top:8px;font-size:0.82rem;color:#cbd5e1;">
              💡 <strong>Explanation:</strong> {q.get('explanation','')}
            </div>""", unsafe_allow_html=True)
            n1, n2 = st.columns(2)
            with n1:
                if st.button("🔄 New Question", use_container_width=True):
                    st.session_state.quiz_state    = None
                    st.session_state.quiz_answered = False
                    st.rerun()
            with n2:
                if st.button("💬 Discuss this topic", use_container_width=True):
                    st.rerun()

# ════════════════════════════════════════════════════════════════
# TAB 4 — NOTES
# ════════════════════════════════════════════════════════════════

with tab_notes:
    st.markdown("#### 🗒️ Saved Notes")
    with st.expander("✏️ Add a custom note"):
        note_txt = st.text_area("Your note",
                                placeholder="Write something to remember…",
                                height=90, key="new_note")
        if st.button("💾 Save Note", use_container_width=True):
            if note_txt.strip():
                ok = save_note(note_txt.strip(), cls, subj, "Manual")
                st.toast("✅ Saved!" if ok else "❌ Could not save.", icon="🗒️")
                if ok: st.rerun()
            else:
                st.warning("Note is empty.", icon="⚠️")
    st.markdown("---")

    notes = get_notes(cls)
    if not notes:
        st.markdown("""
        <div style="text-align:center;padding:36px 20px;color:#64748b;">
          <div style="font-size:2rem;">🗒️</div>
          <div style="font-weight:600;margin-top:6px;">No notes yet</div>
          <div style="font-size:0.78rem;margin-top:4px;">Tap "Save" under any chat answer</div>
        </div>""", unsafe_allow_html=True)
    else:
        for note in notes:
            ts      = note.get("created_at","")[:16].replace("T"," ")
            ch      = note.get("chapter","")
            txt     = note.get("content","")
            nid     = note.get("id")
            display = txt[:400] + ("…" if len(txt) > 400 else "")
            na, nb  = st.columns([8, 1])
            with na:
                st.markdown(f"""
                <div class="note-card">
                  <div class="note-ts">🕐 {ts}</div>
                  <div class="note-ch">
                    {CLASS_ICONS.get(note.get('class',''),'')} {note.get('class','')} ·
                    {SUBJECT_ICONS.get(note.get('subject',''),'')} {note.get('subject','')}
                    {'· 📖 '+ch if ch and ch not in ('General','Manual') else ''}
                  </div>
                  <div class="note-body">{display}</div>
                </div>""", unsafe_allow_html=True)
            with nb:
                if st.button("🗑️", key=f"del_{nid}", help="Delete"):
                    delete_note(nid); st.rerun()
