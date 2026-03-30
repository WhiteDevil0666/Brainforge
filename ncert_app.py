# ================================================================
# BrainForge — NCERT Chat (Class 8, 9, 10 | Maths & Science)
# v3.0 — NEW FEATURE: 📸 Image Doubt Solver
#   Upload a photo of any textbook question, handwritten problem,
#   or diagram → get an instant NCERT-grounded AI explanation.
#   Uses Groq llama-3.2-90b-vision-preview (multimodal).
#
# ALL v2.0 FEATURES RETAINED:
#   1. LaTeX / KaTeX math rendering
#   2. Chapter progress tracking
#   3. Wrong answer review in Quiz
#   4. Streak system (daily login)
#
# SUPABASE TABLES REQUIRED (run once in SQL editor):
# ----------------------------------------------------------------
# CREATE TABLE streaks (
#   id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
#   user_id uuid REFERENCES users(id) ON DELETE CASCADE,
#   current_streak int DEFAULT 1,
#   longest_streak int DEFAULT 1,
#   last_login_date date,
#   created_at timestamptz DEFAULT now()
# );
# CREATE UNIQUE INDEX idx_streaks_unique_user ON streaks(user_id);
#
# CREATE TABLE chapter_progress (
#   id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
#   user_id uuid REFERENCES users(id) ON DELETE CASCADE,
#   class text NOT NULL, subject text NOT NULL,
#   chapter_key text NOT NULL,
#   first_visited date, last_visited date,
#   visit_count int DEFAULT 1,
#   created_at timestamptz DEFAULT now()
# );
#
# CREATE TABLE quiz_attempts (
#   id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
#   user_id uuid REFERENCES users(id) ON DELETE CASCADE,
#   topic text, question text, options jsonb,
#   correct_answer text, user_answer text,
#   explanation text, is_correct boolean,
#   class text, subject text,
#   created_at timestamptz DEFAULT now()
# );
# ================================================================

import os, re, json, base64, datetime, random, string, resend
import streamlit as st
from groq import Groq
from supabase import create_client, Client
from PIL import Image
import io

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

GROQ_MODEL        = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
:root {
  --bg:#06070f; --surface:#0d1117; --border:rgba(255,255,255,0.07);
  --accent:#7c3aed; --text:#e2e8f0; --muted:#64748b;
  --success:#10b981; --warn:#f59e0b; --danger:#ef4444;
  --radius:14px; --font:'Space Grotesk',sans-serif;
}
html,body,.stApp { background:var(--bg)!important; color:var(--text)!important; font-family:var(--font)!important; }
#MainMenu,footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent!important; border-bottom:none!important; }
button[data-testid="collapsedControl"] {
  display:block!important; visibility:visible!important; opacity:1!important;
  background:rgba(124,58,237,0.2)!important; border-radius:8px!important; color:#a78bfa!important;
}
section[data-testid="stSidebar"] { background:#080b14!important; border-right:1px solid rgba(124,58,237,0.18)!important; }
section[data-testid="stSidebar"] * { color:var(--text)!important; }
.block-container { padding:1rem 1.5rem 110px 1.5rem!important; max-width:100%!important; }
.stButton>button { background:var(--accent)!important; color:#fff!important; border:none!important;
  border-radius:10px!important; font-weight:700!important; font-size:0.85rem!important; padding:0.5rem 1rem!important; }
.stButton>button:hover { background:#6d28d9!important; }
.stTextInput input,textarea { background:var(--surface)!important; border:1px solid var(--border)!important;
  border-radius:10px!important; color:var(--text)!important; }
textarea:focus,.stTextInput input:focus { border-color:var(--accent)!important; }
div[data-testid="stChatMessage"] { background:transparent!important; border:none!important; }
.msg-user { background:rgba(124,58,237,0.2); border-radius:16px; padding:10px 14px; max-width:80%; }
div[data-testid="stChatInput"] {
  position:sticky!important; bottom:0!important; z-index:10!important;
  background:linear-gradient(to top,#06070f 70%,transparent)!important; padding:14px 20px!important;
}
div[data-testid="stChatInput"] textarea {
  background:#0d1117!important; border:1px solid rgba(124,58,237,0.4)!important;
  border-radius:12px!important; color:var(--text)!important; font-size:0.9rem!important; padding:12px!important;
}
.stTabs [data-baseweb="tab-list"] { background:var(--surface)!important; border-radius:10px!important; padding:4px!important; }
.stTabs [data-baseweb="tab"] { color:var(--muted)!important; font-weight:600!important; }
.stTabs [aria-selected="true"] { background:var(--accent)!important; color:#fff!important; }
.stRadio label { background:rgba(255,255,255,0.03)!important; border:1px solid var(--border)!important;
  border-radius:8px!important; padding:6px 10px!important; }
.bf-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:14px; }
.note-card { background:rgba(6,182,212,0.05); border:1px solid rgba(6,182,212,0.2); border-radius:12px; padding:12px; }
.quiz-opt { background:rgba(255,255,255,0.04); border:1px solid var(--border); border-radius:10px; padding:10px; margin-bottom:6px; }
.quiz-opt.correct { border-color:var(--success)!important; background:rgba(16,185,129,0.1); }
.quiz-opt.incorrect { border-color:var(--danger)!important; background:rgba(239,68,68,0.1); }

/* IMAGE DOUBT */
.img-solution-box { background:rgba(255,255,255,0.025); border:1px solid rgba(124,58,237,0.25); border-radius:14px; padding:18px 20px; margin-top:14px; }
.img-type-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(124,58,237,0.12);
  border:1px solid rgba(124,58,237,0.28); border-radius:8px; padding:4px 12px;
  font-size:0.7rem; font-weight:700; color:#a78bfa; margin-bottom:12px; }
.img-preview-wrap { border:1px solid rgba(255,255,255,0.08); border-radius:12px; overflow:hidden; margin-bottom:12px; }

/* STREAK */
.streak-badge { display:inline-flex; align-items:center; gap:6px;
  background:linear-gradient(135deg,rgba(251,146,60,0.15),rgba(239,68,68,0.1));
  border:1px solid rgba(251,146,60,0.3); border-radius:10px; padding:6px 12px; margin:6px 0; }
.streak-num { font-size:1.2rem; font-weight:800; color:#fb923c; }
.streak-label { font-size:0.68rem; color:#94a3b8; font-weight:600; }

/* PROGRESS */
.ch-visited { display:inline-block; background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3);
  border-radius:6px; padding:2px 7px; font-size:0.62rem; color:#10b981; font-weight:700; margin-left:6px; }
.ch-new { display:inline-block; background:rgba(124,58,237,0.1); border:1px solid rgba(124,58,237,0.25);
  border-radius:6px; padding:2px 7px; font-size:0.62rem; color:#a78bfa; font-weight:600; }
.prog-bar-wrap { height:5px; background:rgba(255,255,255,0.07); border-radius:99px; overflow:hidden; margin-top:4px; }
.prog-bar { height:100%; border-radius:99px; transition:width 0.3s; }

/* WRONG ANSWER */
.wrong-card { background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.2);
  border-radius:12px; padding:13px 16px; margin-bottom:10px; }
.wrong-card .wc-topic { font-size:0.66rem; color:#f87171; font-weight:700; margin-bottom:4px; }
.wrong-card .wc-q { font-size:0.86rem; color:#e2e8f0; font-weight:600; margin-bottom:6px; }
.wrong-card .wc-ans { font-size:0.78rem; margin-top:4px; }
.wrong-card .wc-exp { font-size:0.76rem; color:#94a3b8; margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.06); }

@media (max-width:768px) { .block-container { padding:0.5rem 0.75rem 110px 0.75rem!important; } }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:rgba(124,58,237,0.4); border-radius:99px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# SECRETS & CLIENTS
# ════════════════════════════════════════════════════════════════

def _secret(k):
    try:    return st.secrets[k]
    except: return os.getenv(k, "")

GROQ_API_KEY   = _secret("GROQ_API_KEY")
SUPABASE_URL   = _secret("SUPABASE_URL")
SUPABASE_KEY   = _secret("SUPABASE_KEY")
RESEND_API_KEY = _secret("RESEND_API_KEY")

missing = [k for k,v in {"GROQ_API_KEY":GROQ_API_KEY,"SUPABASE_URL":SUPABASE_URL,
                          "SUPABASE_KEY":SUPABASE_KEY,"RESEND_API_KEY":RESEND_API_KEY}.items() if not v]
if missing:
    st.error(f"Missing secrets: {', '.join(missing)}")
    st.code("\n".join(f'{k} = "..."' for k in missing), language="toml")
    st.stop()

@st.cache_resource
def get_groq():  return Groq(api_key=GROQ_API_KEY)
@st.cache_resource
def get_sb():    return create_client(SUPABASE_URL, SUPABASE_KEY)

groq_client    = get_groq()
sb: Client     = get_sb()
resend.api_key = RESEND_API_KEY

# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════

def _init():
    for k, v in {
        "auth_stage":"input","auth_identifier":"","auth_user_id":None,"auth_otp_time":None,
        "messages":[],"selected_chapter":None,"chapter_mode":False,
        "selected_class":"Class 8","selected_subject":"Both",
        "quiz_state":None,"quiz_answered":False,"quiz_last_pick":"","quiz_topic_last":"",
        "usage":None,"streak":None,"chapter_progress":None,"streak_initialized":False,
        "img_solution":None,"img_solution_history":[],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ════════════════════════════════════════════════════════════════
# AUTH
# ════════════════════════════════════════════════════════════════

def _norm(s):     return s.strip().lower()
def _otp():       return "".join(random.choices(string.digits, k=6))

def _validate(val):
    v = _norm(val)
    if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v): return True, "email"
    if 10 <= len(re.sub(r"\D","",v)) <= 13:          return True, "phone"
    return False, ""

def get_or_create_user(identifier):
    v = _norm(identifier)
    try:
        r = sb.table("users").select("id").eq("identifier", v).execute()
        if r.data: return r.data[0]["id"]
        return sb.table("users").insert({"identifier": v}).execute().data[0]["id"]
    except Exception as e:
        st.error(f"DB error: {e}"); return None

def store_otp(identifier, otp):
    try: sb.table("otps").insert({"identifier":_norm(identifier),"otp":otp}).execute()
    except Exception as e: st.error(f"OTP store error: {e}")

def verify_otp(identifier, otp):
    try:
        cutoff = (datetime.datetime.utcnow()-datetime.timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat()
        r = (sb.table("otps").select("id,otp,used").eq("identifier",_norm(identifier))
               .eq("used",False).gte("created_at",cutoff).order("created_at",desc=True).limit(1).execute())
        if not r.data: return False
        row = r.data[0]
        if row["otp"] == otp.strip():
            sb.table("otps").update({"used":True}).eq("id",row["id"]).execute(); return True
        return False
    except Exception as e:
        st.error(f"OTP verify error: {e}"); return False

def send_otp_email(to_email, otp):
    try:
        resend.Emails.send({
            "from":"onboarding@resend.dev","to":to_email.strip(),
            "subject":"Your BrainForge OTP 🔐",
            "html":f'<div style="font-family:Arial;background:#06070f;color:#e2e8f0;padding:32px;border-radius:12px;max-width:480px;margin:auto"><h2 style="color:#7c3aed">🧠 BrainForge</h2><p style="color:#94a3b8">Your one-time login code:</p><div style="font-size:44px;font-weight:900;color:#7c3aed;letter-spacing:12px;margin:20px 0;text-align:center">{otp}</div><p style="color:#64748b;font-size:13px">Valid for {OTP_EXPIRY_MINUTES} minutes.</p></div>',
        }); return True
    except Exception as e:
        print(f"[Resend ERROR] {e}"); st.error(f"❌ Email failed: {e}"); return False

def render_auth():
    _, mid, _ = st.columns([1,2,1])
    with mid:
        st.markdown("""<div style="background:#0d1117;border:1px solid rgba(124,58,237,0.28);border-radius:20px;
             padding:36px 30px;text-align:center;margin-top:60px;">
          <div style="font-size:2.8rem">🧠</div>
          <div style="font-size:1.35rem;font-weight:800;color:#e2e8f0;margin:10px 0 4px">BrainForge</div>
          <div style="font-size:0.8rem;color:#64748b;margin-bottom:24px">NCERT AI Tutor · Class 8, 9 &amp; 10<br>
            Sign in to get <strong style="color:#a78bfa">15 free questions daily</strong></div></div>""",
            unsafe_allow_html=True)
        st.markdown("")
        if st.session_state.auth_stage == "input":
            identifier = st.text_input("📧 Email address", placeholder="you@example.com", key="auth_id_input")
            if st.button("Send OTP →", type="primary", use_container_width=True):
                valid, kind = _validate(identifier)
                if not valid: st.error("Enter a valid email address.", icon="❌")
                elif kind == "phone": st.error("📱 SMS OTP not supported yet.", icon="⚠️")
                else:
                    otp = _otp(); store_otp(identifier, otp)
                    if send_otp_email(identifier.strip(), otp):
                        st.success("📩 OTP sent!")
                        st.session_state.auth_identifier = identifier.strip()
                        st.session_state.auth_otp_time   = datetime.datetime.utcnow()
                        st.session_state.auth_stage      = "otp"; st.rerun()
        elif st.session_state.auth_stage == "otp":
            ident = st.session_state.auth_identifier
            st.info(f"OTP sent to **{ident}**")
            otp_in = st.text_input("Enter 6-digit OTP", max_chars=6, placeholder="······", key="otp_in")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Verify", type="primary", use_container_width=True):
                    if verify_otp(ident, otp_in):
                        uid = get_or_create_user(ident)
                        if uid:
                            st.session_state.auth_user_id=uid; st.session_state.auth_stage="done"
                            st.session_state.usage=None; st.rerun()
                        else: st.error("Account error.", icon="❌")
                    else: st.error("Wrong or expired OTP.", icon="❌")
            with c2:
                if st.button("🔄 Resend OTP", use_container_width=True):
                    new_otp=_otp(); store_otp(ident,new_otp)
                    if send_otp_email(ident,new_otp): st.success("📩 New OTP sent!")
            if st.button("← Change email", use_container_width=True):
                st.session_state.auth_stage="input"; st.rerun()

if st.session_state.auth_stage != "done":
    render_auth(); st.stop()

# ════════════════════════════════════════════════════════════════
# RATE LIMITING
# ════════════════════════════════════════════════════════════════

def get_usage():
    try:
        uid=st.session_state.auth_user_id; today=datetime.date.today().isoformat()
        r=sb.table("rate_limits").select("count").eq("user_id",uid).eq("day",today).execute()
        return r.data[0]["count"] if r.data else 0
    except: return 0

def increment_usage():
    try:
        uid=st.session_state.auth_user_id; today=datetime.date.today().isoformat()
        r=sb.table("rate_limits").select("count").eq("user_id",uid).eq("day",today).execute()
        if r.data:
            new=r.data[0]["count"]+1
            sb.table("rate_limits").update({"count":new}).eq("user_id",uid).eq("day",today).execute()
        else:
            new=1; sb.table("rate_limits").insert({"user_id":uid,"day":today,"count":1}).execute()
        return new
    except: return 0

def rate_limit_check():
    if st.session_state.usage is None: st.session_state.usage=get_usage()
    if st.session_state.usage >= DAILY_LIMIT:
        st.error(f"🚫 Daily limit of **{DAILY_LIMIT} questions** reached. Come back tomorrow! 🌅"); return False
    return True

# ════════════════════════════════════════════════════════════════
# STREAK
# ════════════════════════════════════════════════════════════════

def update_streak():
    try:
        uid=st.session_state.auth_user_id
        today=datetime.date.today().isoformat()
        yesterday=(datetime.date.today()-datetime.timedelta(days=1)).isoformat()
        r=sb.table("streaks").select("*").eq("user_id",uid).execute()
        if r.data:
            row=r.data[0]; last=row.get("last_login_date","")
            streak=row.get("current_streak",0); longest=row.get("longest_streak",0)
            if last==today: return streak
            elif last==yesterday: streak+=1
            else: streak=1
            longest=max(longest,streak)
            sb.table("streaks").update({"current_streak":streak,"longest_streak":longest,"last_login_date":today}).eq("user_id",uid).execute()
            return streak
        else:
            sb.table("streaks").insert({"user_id":uid,"current_streak":1,"longest_streak":1,"last_login_date":today}).execute()
            return 1
    except Exception as e:
        print(f"[Streak error] {e}"); return 0

if not st.session_state.streak_initialized and st.session_state.auth_user_id:
    st.session_state.streak=update_streak(); st.session_state.streak_initialized=True

# ════════════════════════════════════════════════════════════════
# CHAPTER PROGRESS
# ════════════════════════════════════════════════════════════════

def mark_chapter_visited(ch_key, cls, subj):
    try:
        uid=st.session_state.auth_user_id; today=datetime.date.today().isoformat()
        r=sb.table("chapter_progress").select("id,visit_count").eq("user_id",uid).eq("chapter_key",ch_key).execute()
        if r.data:
            sb.table("chapter_progress").update({"last_visited":today,"visit_count":r.data[0]["visit_count"]+1}).eq("id",r.data[0]["id"]).execute()
        else:
            sb.table("chapter_progress").insert({"user_id":uid,"class":cls,"subject":subj,"chapter_key":ch_key,"first_visited":today,"last_visited":today,"visit_count":1}).execute()
        st.session_state.chapter_progress=None
    except Exception as e: print(f"[Progress error] {e}")

def get_chapter_progress():
    if st.session_state.chapter_progress is not None: return st.session_state.chapter_progress
    try:
        r=sb.table("chapter_progress").select("chapter_key,visit_count").eq("user_id",st.session_state.auth_user_id).execute()
        prog={row["chapter_key"]:row["visit_count"] for row in (r.data or [])}
        st.session_state.chapter_progress=prog; return prog
    except: return {}

def get_class_progress_pct(cls):
    prog=get_chapter_progress()
    total=sum(len(v) for v in CHAPTER_INDEX.get(cls,{}).values())
    if total==0: return 0.0
    done=sum(1 for k in prog if any(k in CHAPTER_INDEX[cls][s] for s in CHAPTER_INDEX.get(cls,{})))
    return min(done/total,1.0)

# ════════════════════════════════════════════════════════════════
# QUIZ TRACKING
# ════════════════════════════════════════════════════════════════

def save_quiz_attempt(topic, q_data, user_answer, is_correct):
    try:
        sb.table("quiz_attempts").insert({
            "user_id":st.session_state.auth_user_id,"topic":topic,
            "question":q_data["question"],"options":json.dumps(q_data["options"]),
            "correct_answer":q_data["correct"],"user_answer":user_answer,
            "explanation":q_data.get("explanation",""),"is_correct":is_correct,
            "class":st.session_state.selected_class,"subject":st.session_state.selected_subject,
        }).execute()
    except Exception as e: print(f"[Quiz attempt error] {e}")

def get_wrong_answers(limit=20):
    try:
        r=(sb.table("quiz_attempts")
            .select("topic,question,options,correct_answer,user_answer,explanation,class,subject,created_at")
            .eq("user_id",st.session_state.auth_user_id).eq("is_correct",False)
            .order("created_at",desc=True).limit(limit).execute())
        return r.data or []
    except: return []

def get_quiz_stats():
    try:
        r=sb.table("quiz_attempts").select("is_correct").eq("user_id",st.session_state.auth_user_id).execute()
        data=r.data or []; return len(data), sum(1 for d in data if d["is_correct"])
    except: return 0,0

# ════════════════════════════════════════════════════════════════
# LaTeX RENDERER
# ════════════════════════════════════════════════════════════════

def render_answer_with_math(text):
    parts=re.split(r'(\$\$[\s\S]+?\$\$)',text)
    for part in parts:
        if part.startswith('$$') and part.endswith('$$') and len(part)>4:
            try: st.latex(part[2:-2].strip())
            except: st.markdown(f"```\n{part[2:-2].strip()}\n```")
        elif part.strip(): st.markdown(part)

# ════════════════════════════════════════════════════════════════
# ✨ IMAGE DOUBT SOLVER
# ════════════════════════════════════════════════════════════════

def prepare_image_for_api(uploaded_file):
    """Resize + compress image, return (base64_str, mime_type)."""
    img=Image.open(uploaded_file)
    if img.mode in ("RGBA","P"): img=img.convert("RGB")
    if img.width>1600:
        ratio=1600/img.width; img=img.resize((1600,int(img.height*ratio)),Image.LANCZOS)
    buf=io.BytesIO(); img.save(buf,format="JPEG",quality=85); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8"), "image/jpeg"

def detect_question_type(image_b64, mime):
    """Quick vision call to classify the image content."""
    try:
        r=groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:{mime};base64,{image_b64}"}},
                {"type":"text","text":(
                    "Classify this image in 3-4 words. Reply ONLY with one of: "
                    "'Maths Problem','Science Diagram','Handwritten Question',"
                    "'Textbook Exercise','Chemistry Equation','Physics Numericals',"
                    "'Biology Diagram','Graph or Chart','Mixed Content'. No extra text."
                )},
            ]}],
            max_tokens=20, temperature=0.1,
        )
        return r.choices[0].message.content.strip()
    except: return "Question"

def solve_image_doubt(image_b64, mime, cls, subj, extra_hint=""):
    """
    Core vision solve call.
    Sends the image + structured prompt to GROQ_VISION_MODEL.
    Returns a full step-by-step solution string.
    """
    age={"Class 8":"13-14","Class 9":"14-15","Class 10":"15-16"}.get(cls,"13-16")
    hint_line=f"\nStudent note: {extra_hint}" if extra_hint.strip() else ""

    prompt=(
        f"You are BrainForge — expert CBSE tutor for {cls} students (age {age}).\n"
        f"The student has uploaded a photo of a {subj} question/diagram.{hint_line}\n\n"
        "YOUR JOB:\n"
        "1. Read the question / diagram carefully from the image.\n"
        "2. Identify the exact topic (e.g. Quadratic Equations, Photosynthesis).\n"
        "3. Solve it step-by-step using NCERT methods.\n"
        "4. Use LaTeX for ALL math: display math $$...$$, inline math $...$\n"
        "5. Keep language simple and student-friendly.\n\n"
        "FORMAT:\n"
        "## 🔍 [Topic Name]\n"
        "**What the question asks:** [1-line summary]\n\n"
        "### Step-by-Step Solution\n"
        "[numbered steps with full working]\n\n"
        "### ✅ Final Answer\n"
        "[clearly stated]\n\n"
        f"💡 **Quick Tip:** [memory trick or exam tip]\n"
        f"📚 *NCERT {cls} — {subj}*"
    )

    try:
        r=groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:{mime};base64,{image_b64}"}},
                {"type":"text","text":prompt},
            ]}],
            max_tokens=1600, temperature=0.3,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        err=str(e)
        if "rate_limit" in err.lower() or "429" in err:
            return "⚠️ Vision model rate limit hit. Please wait 30 seconds and try again."
        if "model_not_active" in err.lower() or "model" in err.lower():
            return (
                "⚠️ Vision model unavailable on your Groq plan.\n\n"
                "**Fix:** Go to [Groq Console](https://console.groq.com/) → Models "
                "and confirm `llama-3.2-90b-vision-preview` is active. "
                "You can also try swapping to `llama-3.2-11b-vision-preview` in the config."
            )
        return f"⚠️ Error: {err}"

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
    m=get_embedder()
    return m.encode(text,normalize_embeddings=True).tolist() if m else None

def retrieve_chunks(query,class_label,subject,chapter_filter="",top_k=TOP_K):
    q_emb=embed(query)
    if not q_emb: return []
    try:
        r=sb.rpc("match_chunks",{"query_embedding":q_emb,"filter_class":class_label,
            "filter_subject":subject,"filter_chapter":chapter_filter or "","match_count":top_k}).execute()
        out=[]
        for row in (r.data or []):
            sim=float(row.get("similarity",0))
            if sim>0.15:
                out.append({"text":row["content"],"subject":row["subject"],"chapter":row["chapter"],
                    "page":row.get("page","?"),"source":row.get("source",""),
                    "class":row["class"],"relevance":round(sim*100,1)})
        return sorted(out,key=lambda x:x["relevance"],reverse=True)
    except Exception as e:
        st.warning(f"Search error: {e}",icon="⚠️"); return []

# ════════════════════════════════════════════════════════════════
# NOTES
# ════════════════════════════════════════════════════════════════

def save_note(content,cls,subj,chapter=""):
    try:
        sb.table("notes").insert({"user_id":st.session_state.auth_user_id,
            "class":cls,"subject":subj,"chapter":chapter,"content":content}).execute(); return True
    except: return False

def get_notes(cls=None):
    try:
        q=sb.table("notes").select("*").eq("user_id",st.session_state.auth_user_id).order("created_at",desc=True)
        if cls: q=q.eq("class",cls)
        return (q.limit(50).execute()).data or []
    except: return []

def delete_note(nid):
    try:
        sb.table("notes").delete().eq("id",nid).eq("user_id",st.session_state.auth_user_id).execute(); return True
    except: return False

# ════════════════════════════════════════════════════════════════
# AI GENERATION (text)
# ════════════════════════════════════════════════════════════════

def build_ctx(chunks):
    if not chunks: return ""
    out="\n--- NCERT CONTENT ---\n"
    for c in chunks[:6]:
        out+=f"\n[{c['class']}|{c['subject']}|{c['chapter']}|p.{c['page']}]\n{c['text']}\n"
    return out+"\n--- END ---\n"

def generate_answer(question,chunks,cls,subj,style,history,ch_ctx=None):
    age={"Class 8":"13-14","Class 9":"14-15","Class 10":"15-16"}.get(cls,"13-16")
    style_instr={"Simple":"Simple language, short paragraphs.","Detailed":"Comprehensive, cover all sub-topics.",
        "Bullets":"Use numbered headings and bullet points.","Examples":"Include 2-3 real Indian examples."}.get(style,"Simple language.")
    ch_line=f"\nStudent studying: {ch_ctx}" if ch_ctx else ""
    system=(f"You are BrainForge — expert CBSE AI tutor for India.\n"
        f"MATH: Always use LaTeX. Display math: $$...$$ Inline: $...$\n"
        f"STYLE: {style_instr} Bold key terms. Add 💡 Quick Tip.\n"
        f"STUDENT: {cls} CBSE, age {age}{ch_line}\n"
        f"FORMAT: ## [Topic] / Explanation / **Key Points:** / 💡 Quick Tip / 📚 NCERT {cls} {subj}\n"
        f"End with: 💬 Want examples, a quiz, or deeper explanation?")
    msgs=[{"role":"system","content":system}]
    for m in history[-(MAX_HISTORY*2):]:
        if m["role"] in ("user","assistant"): msgs.append({"role":m["role"],"content":m["content"]})
    msgs.append({"role":"user","content":f"Q: {question}\n{build_ctx(chunks)}\nAnswer for {cls} student."})
    try:
        r=groq_client.chat.completions.create(model=GROQ_MODEL,messages=msgs,temperature=0.4,max_tokens=1400)
        return r.choices[0].message.content.strip()
    except Exception as e:
        err=str(e)
        return "⚠️ AI rate limit hit. Wait 30 s." if "rate_limit" in err.lower() or "429" in err else f"⚠️ Error: {err}"

def generate_quiz(topic,cls,subj,chunks):
    prompt=(f'CBSE exam setter. Generate ONE MCQ on "{topic}" | {cls} | {subj}\n{build_ctx(chunks[:4])}\n'
        f'Reply ONLY valid JSON (no markdown):\n{{"question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"correct":"A","explanation":"..."}}')
    try:
        r=groq_client.chat.completions.create(model=GROQ_MODEL,messages=[{"role":"user","content":prompt}],temperature=0.3,max_tokens=400)
        return json.loads(re.sub(r"```json|```","",r.choices[0].message.content.strip()).strip())
    except: return None

def generate_chapter_summary(ch_key,ch_title,cls,subj):
    chunks=retrieve_chunks(ch_title,cls,subj,ch_key,top_k=8)
    ctx="\n".join(c["text"] for c in chunks[:6])
    prompt=(f"Expert {cls} CBSE teacher. Student opened: {ch_title} ({cls}·{subj})\nNCERT: {ctx}\n"
        "Write chapter overview with: What You Will Learn / Key Concepts / Why It Matters / Quick Preview\n"
        'End: "💬 Ask me anything about this chapter!"')
    try:
        r=groq_client.chat.completions.create(model=GROQ_MODEL,messages=[{"role":"user","content":prompt}],temperature=0.4,max_tokens=800)
        return r.choices[0].message.content.strip()
    except Exception as e: return f"Could not load overview: {e}"

def is_followup(q,history):
    if not history: return False
    ql=q.lower().strip()
    if len(ql.split())<=5:
        for p in [r"^(what about|tell me more|explain more|can you|how about|give me|more about)",
                  r"^(example|examples|illustrate|show me)",
                  r"^(i (don't|do not) understand|unclear|confused|simpler)",
                  r"^(test me|quiz me|ask me|mcq)"]:
            if re.match(p,ql): return True
    return False

def process_question(question,ch_ctx=None,ch_filter="",ch_subj=None):
    if not rate_limit_check(): return None,[]
    cls_=st.session_state.selected_class; subj_=ch_subj or st.session_state.selected_subject
    style=st.session_state.get("answer_depth","Simple"); hist=st.session_state.messages; chunks=[]
    if not is_followup(question,hist):
        with st.spinner("🔍 Searching NCERT…"): chunks=retrieve_chunks(question,cls_,subj_,ch_filter,TOP_K)
    with st.spinner("✍️ Writing answer…"): answer=generate_answer(question,chunks,cls_,subj_,style,hist,ch_ctx)
    st.session_state.usage=increment_usage()
    return answer,chunks

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    cls=st.session_state.selected_class; subj=st.session_state.selected_subject
    ident_short=st.session_state.auth_identifier
    if len(ident_short)>22: ident_short=ident_short[:20]+"…"
    streak_val=st.session_state.streak or 0
    flame="🔥" if streak_val>=2 else "✨"

    st.markdown(f"""
    <div style="padding:8px 0 14px;border-bottom:1px solid rgba(255,255,255,0.07)">
      <div style="font-size:1.15rem;font-weight:800;color:#e2e8f0">🧠 BrainForge</div>
      <div style="font-size:0.68rem;color:#64748b;margin-top:2px">NCERT Class 8 · 9 · 10</div>
      <div style="margin-top:7px;font-size:0.7rem;color:#a78bfa;font-weight:600">👤 {ident_short}</div>
      <div class="streak-badge" style="margin-top:10px">
        <span style="font-size:1.1rem">{flame}</span>
        <span class="streak-num">{streak_val}</span>
        <span class="streak-label">day streak</span>
      </div>
    </div>""",unsafe_allow_html=True)

    st.markdown("#### 🎓 Class")
    sel_cls=st.radio("cls",CLASSES,format_func=lambda x:f"{CLASS_ICONS[x]} {x}",label_visibility="collapsed",key="cls_radio")
    if sel_cls!=st.session_state.selected_class:
        st.session_state.update(selected_class=sel_cls,selected_chapter=None,chapter_mode=False,
                                messages=[],quiz_state=None,chapter_progress=None)

    prog_pct=get_class_progress_pct(sel_cls); prog_pct_int=int(prog_pct*100)
    prog_color="#10b981" if prog_pct>=0.7 else "#7c3aed" if prog_pct>=0.3 else "#06b6d4"
    st.markdown(f"""
    <div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:#64748b;margin-bottom:3px">
        <span>Chapter progress</span><span style="color:{prog_color};font-weight:700">{prog_pct_int}%</span>
      </div>
      <div class="prog-bar-wrap"><div class="prog-bar" style="width:{prog_pct_int}%;background:{prog_color}"></div></div>
    </div>""",unsafe_allow_html=True)

    st.markdown("#### 📚 Subject")
    sel_subj=st.radio("subj",SUBJECTS,format_func=lambda x:f"{SUBJECT_ICONS[x]} {x}",label_visibility="collapsed",key="subj_radio")
    if sel_subj!=st.session_state.selected_subject:
        st.session_state.update(selected_subject=sel_subj,selected_chapter=None,chapter_mode=False,messages=[],quiz_state=None)

    total_q,correct_q=get_quiz_stats(); acc=int((correct_q/total_q)*100) if total_q else 0
    if total_q:
        st.markdown(f"""
        <div style="margin:8px 0;padding:8px 10px;background:rgba(16,185,129,0.06);
             border:1px solid rgba(16,185,129,0.18);border-radius:9px">
          <div style="font-size:0.65rem;color:#64748b;font-weight:600;margin-bottom:3px">🎯 Quiz Accuracy</div>
          <div style="font-size:1.1rem;font-weight:800;color:#10b981">{acc}%
            <span style="font-size:0.68rem;color:#64748b">({correct_q}/{total_q})</span></div>
        </div>""",unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.usage is None: st.session_state.usage=get_usage()
    used=st.session_state.usage; pct=min(used/DAILY_LIMIT,1.0)
    bar_c="linear-gradient(90deg,#7c3aed,#06b6d4)" if pct<0.8 else "linear-gradient(90deg,#f59e0b,#ef4444)"
    st.markdown(f"""
    <div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span style="font-size:0.7rem;color:#94a3b8;font-weight:600">Daily Usage</span>
        <span style="font-size:0.7rem;color:#a78bfa;font-weight:700">{used}/{DAILY_LIMIT}</span>
      </div>
      <div class="prog-bar-wrap"><div class="prog-bar" style="width:{int(pct*100)}%;background:{bar_c}"></div></div>
    </div>""",unsafe_allow_html=True)

    st.markdown("---")
    show_src=st.toggle("📄 Show NCERT sources",value=False,key="show_src")
    answer_depth=st.selectbox("Answer style",["Simple","Detailed","Bullets","Examples"],key="answer_depth")

    if st.button("🗑️ Clear Chat",use_container_width=True):
        st.session_state.update(messages=[],selected_chapter=None,chapter_mode=False,quiz_state=None); st.rerun()
    if st.button("🚪 Sign Out",use_container_width=True):
        for k in ["auth_stage","auth_identifier","auth_user_id","auth_otp_time","messages","usage",
                  "quiz_state","selected_chapter","streak","streak_initialized","chapter_progress",
                  "img_solution","img_solution_history"]:
            st.session_state[k]=None
        st.session_state.auth_stage="input"; st.rerun()

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

cls=st.session_state.selected_class; subj=st.session_state.selected_subject
c_col=CLASS_COLORS.get(cls,"#7c3aed")

st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(124,58,237,0.09),rgba(6,182,212,0.04));
     border:1px solid {c_col}30;border-radius:14px;padding:12px 18px;margin-bottom:12px">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <span style="font-size:1.5rem">{CLASS_ICONS.get(cls,"🎓")}</span>
    <div>
      <div style="font-size:1rem;font-weight:800;color:#e2e8f0">NCERT AI Tutor</div>
      <div style="font-size:0.72rem;color:#64748b">{cls} · {subj} · NCERT-grounded answers</div>
    </div>
    <div style="margin-left:auto;display:flex;gap:6px;flex-wrap:wrap">
      <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);
            border-radius:6px;padding:3px 9px;font-size:0.68rem;font-weight:700;color:#a78bfa">
        {CLASS_ICONS.get(cls,"")} {cls}</span>
      <span style="background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.25);
            border-radius:6px;padding:3px 9px;font-size:0.68rem;font-weight:700;color:#67e8f9">
        {SUBJECT_ICONS.get(subj,"")} {subj}</span>
    </div>
  </div>
</div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab_chat,tab_img,tab_idx,tab_quiz,tab_notes=st.tabs(["💬 Chat","📸 Image Doubt","📖 Chapters","🎯 Quiz","🗒️ Notes"])

def render_sources(chunks):
    if not chunks or not st.session_state.get("show_src"): return
    with st.expander(f"📄 {len(chunks)} NCERT passages"):
        for c in chunks:
            rel=c["relevance"]; bc="#10b981" if rel>=70 else "#f59e0b" if rel>=45 else "#64748b"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border-left:3px solid {bc};
                 border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px">
              <div style="display:flex;justify-content:space-between">
                <span style="color:#a78bfa;font-weight:700;font-size:0.72rem">
                  {CLASS_ICONS.get(c.get('class',''),'')} {c.get('class','')} ·
                  {SUBJECT_ICONS.get(c['subject'],'')} {c['subject']} — {c['chapter'].upper()}</span>
                <span style="color:{bc};font-size:0.68rem;font-weight:700">{rel}%</span>
              </div>
              <div style="color:#64748b;font-size:0.66rem;margin:3px 0">p.{c['page']}</div>
              <div style="color:#cbd5e1;font-size:0.76rem;line-height:1.5">
                {c['text'][:230]}{'…' if len(c['text'])>230 else ''}</div>
            </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════

with tab_chat:
    ch_ctx=None; ch_filter=""; ch_subj_=subj
    if st.session_state.chapter_mode and st.session_state.selected_chapter:
        ch_key,ch_s=st.session_state.selected_chapter
        ch_title=CHAPTER_INDEX.get(cls,{}).get(ch_s,{}).get(ch_key,ch_key)
        ch_ctx,ch_filter,ch_subj_=ch_title,ch_key,ch_s
        b1,b2=st.columns([5,1])
        with b1:
            st.markdown(f"""<div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.22);
                 border-radius:10px;padding:8px 14px;margin-bottom:8px">
              <div style="font-size:0.66rem;color:#a78bfa;font-weight:700">📖 Chapter Mode</div>
              <div style="font-size:0.83rem;color:#e2e8f0;font-weight:600">{ch_title}</div>
            </div>""",unsafe_allow_html=True)
        with b2:
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("✕ Exit",key="exit_ch"):
                st.session_state.update(chapter_mode=False,selected_chapter=None,messages=[]); st.rerun()

    if not st.session_state.messages:
        st.markdown("##### 💡 Try asking:")
        suggs=SUGGESTIONS.get(cls,{}).get(subj,SUGGESTIONS["Class 8"]["Both"])
        c1,c2=st.columns(2)
        for i,s in enumerate(suggs):
            with (c1 if i%2==0 else c2):
                if st.button(s,key=f"sg{i}"):
                    answer,chunks=process_question(s,ch_ctx,ch_filter,ch_subj_)
                    if answer:
                        st.session_state.messages+=[{"role":"user","content":s,"chunks":[]},
                                                    {"role":"assistant","content":answer,"chunks":chunks}]
                    st.rerun()
        st.markdown("")

    for idx,msg in enumerate(st.session_state.messages):
        if msg["role"]=="user":
            st.markdown(f'<div style="display:flex;justify-content:flex-end;margin-bottom:6px"><div class="msg-user">{msg["content"]}</div></div>',unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.68rem;color:#a78bfa;font-weight:700;margin-bottom:4px">🧠 BrainForge</div>',unsafe_allow_html=True)
            with st.container():
                st.markdown('<div style="background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.08);border-radius:4px 16px 16px 16px;padding:14px 18px;margin-bottom:4px">',unsafe_allow_html=True)
                render_answer_with_math(msg["content"])
                st.markdown('</div>',unsafe_allow_html=True)
            render_sources(msg.get("chunks",[]))
            sv_col,_=st.columns([1,5])
            with sv_col:
                if st.button("🗒️ Save",key=f"sv_{idx}"):
                    ok=save_note(msg["content"],cls,subj,ch_filter or "General")
                    st.toast("✅ Saved!" if ok else "❌ Could not save.")

    ph=(f"Ask about {ch_title}…" if ch_ctx else f"Ask anything from {cls} {subj}…" if subj!="Both" else f"Ask anything from {cls} Maths or Science…")
    question=st.chat_input(ph)
    if question:
        st.markdown(f'<div style="display:flex;justify-content:flex-end;margin-bottom:6px"><div class="msg-user">{question}</div></div>',unsafe_allow_html=True)
        answer,chunks=process_question(question,ch_ctx,ch_filter,ch_subj_)
        if answer:
            st.markdown('<div style="font-size:0.68rem;color:#a78bfa;font-weight:700;margin-bottom:4px">🧠 BrainForge</div>',unsafe_allow_html=True)
            with st.container():
                st.markdown('<div style="background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.08);border-radius:4px 16px 16px 16px;padding:14px 18px;margin-bottom:4px">',unsafe_allow_html=True)
                render_answer_with_math(answer)
                st.markdown('</div>',unsafe_allow_html=True)
            render_sources(chunks)
            st.session_state.messages+=[{"role":"user","content":question,"chunks":[]},
                                        {"role":"assistant","content":answer,"chunks":chunks}]

# ════════════════════════════════════════════════════════════════
# ✨ TAB 2 — IMAGE DOUBT SOLVER
# ════════════════════════════════════════════════════════════════

with tab_img:
    st.markdown("#### 📸 Image Doubt Solver")
    st.markdown('<div style="font-size:0.78rem;color:#64748b;margin-bottom:16px">Photograph any textbook question, handwritten problem, or diagram — get an instant step-by-step NCERT solution.</div>',unsafe_allow_html=True)

    uploaded=st.file_uploader("Upload image",type=["jpg","jpeg","png","webp"],
                               key="img_uploader",label_visibility="collapsed")
    st.markdown('<div style="text-align:center;color:#475569;font-size:0.72rem;margin:-8px 0 14px">📷 JPG · PNG · WEBP &nbsp;|&nbsp; Works great for: equations, diagrams, MCQs, fill-in-the-blanks</div>',unsafe_allow_html=True)

    if uploaded:
        img_preview=Image.open(uploaded)
        col_prev,col_info=st.columns([3,2])
        with col_prev:
            st.markdown('<div class="img-preview-wrap">',unsafe_allow_html=True)
            st.image(img_preview,use_column_width=True)
            st.markdown('</div>',unsafe_allow_html=True)
        with col_info:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid var(--border);
                 border-radius:12px;padding:14px">
              <div style="font-size:0.7rem;color:#64748b;font-weight:600;margin-bottom:10px">📎 Image Info</div>
              <div style="font-size:0.78rem;color:#cbd5e1;line-height:2">
                <b>File:</b> {uploaded.name}<br>
                <b>Size:</b> {uploaded.size//1024} KB<br>
                <b>Dimensions:</b> {img_preview.width}×{img_preview.height}px<br>
                <b>Solving as:</b><br>
                <span style="color:#a78bfa;font-weight:700">{CLASS_ICONS.get(cls,'')} {cls} · {SUBJECT_ICONS.get(subj,'')} {subj}</span>
              </div>
            </div>""",unsafe_allow_html=True)

        st.markdown("")
        hint=st.text_input("💬 Add a hint (optional)",
            placeholder="e.g. 'Focus on part b only', 'This is from Chapter 4', 'Explain simply'…",
            key="img_hint")

        solve_col,_=st.columns([1,2])
        with solve_col:
            solve_btn=st.button("🔍 Solve this Doubt",type="primary",use_container_width=True,key="solve_img")

        if solve_btn:
            if not rate_limit_check(): st.stop()
            with st.spinner("📸 Reading image…"):
                uploaded.seek(0); img_b64,mime=prepare_image_for_api(uploaded)
            with st.spinner("🔍 Identifying question type…"):
                q_type=detect_question_type(img_b64,mime)
            with st.spinner(f"✍️ Solving {q_type}… (takes ~10 seconds)"):
                solution=solve_image_doubt(img_b64,mime,cls,subj,hint)
            st.session_state.usage=increment_usage()
            ts=datetime.datetime.now().strftime("%H:%M")
            st.session_state.img_solution_history.append(
                {"image_b64":img_b64,"mime":mime,"q_type":q_type,"solution":solution,"ts":ts})
            st.session_state.img_solution=solution
            st.rerun()

    # ── Solution display ──
    if st.session_state.img_solution_history:
        latest=st.session_state.img_solution_history[-1]
        st.markdown("---")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
          <span class="img-type-badge">🏷️ {latest['q_type']}</span>
          <span style="font-size:0.65rem;color:#64748b">Solved at {latest['ts']} ·
            {CLASS_ICONS.get(cls,'')} {cls} · {SUBJECT_ICONS.get(subj,'')} {subj}</span>
        </div>""",unsafe_allow_html=True)

        st.markdown('<div class="img-solution-box">',unsafe_allow_html=True)
        render_answer_with_math(latest["solution"])
        st.markdown('</div>',unsafe_allow_html=True)

        a1,a2,a3=st.columns(3)
        with a1:
            if st.button("🗒️ Save to Notes",use_container_width=True,key="save_img_note"):
                ok=save_note(f"[Image Doubt — {latest['q_type']}]\n\n{latest['solution']}",cls,subj,"Image Doubt")
                st.toast("✅ Saved!" if ok else "❌ Could not save.")
        with a2:
            if st.button("💬 Continue in Chat",use_container_width=True,key="img_to_chat"):
                st.session_state.messages.append({"role":"assistant",
                    "content":f"📸 **From Image Doubt Solver:**\n\n{latest['solution']}","chunks":[]})
                st.toast("✅ Copied to Chat tab!")
        with a3:
            if st.button("🔄 Upload New Image",use_container_width=True,key="clear_img"):
                st.session_state.img_solution=None; st.session_state.img_solution_history=[]; st.rerun()

        if len(st.session_state.img_solution_history)>1:
            with st.expander(f"🕐 Session history ({len(st.session_state.img_solution_history)} solved)"):
                for item in reversed(st.session_state.img_solution_history[:-1]):
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.02);border:1px solid var(--border);
                         border-radius:10px;padding:10px 14px;margin-bottom:8px">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                        <span style="font-size:0.68rem;color:#a78bfa;font-weight:700">🏷️ {item['q_type']}</span>
                        <span style="font-size:0.62rem;color:#475569">{item['ts']}</span>
                      </div>
                      <div style="font-size:0.76rem;color:#94a3b8">{item['solution'][:180]}…</div>
                    </div>""",unsafe_allow_html=True)
    elif not uploaded:
        # Empty state capability cards
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px">
          <div style="background:rgba(124,58,237,0.05);border:1px solid rgba(124,58,237,0.18);border-radius:12px;padding:14px">
            <div style="font-size:1.1rem;margin-bottom:6px">📐</div>
            <div style="font-size:0.78rem;font-weight:700;color:#e2e8f0;margin-bottom:4px">Maths Problems</div>
            <div style="font-size:0.7rem;color:#64748b">Algebra, geometry, trigonometry — full step-by-step working</div>
          </div>
          <div style="background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.18);border-radius:12px;padding:14px">
            <div style="font-size:1.1rem;margin-bottom:6px">🔬</div>
            <div style="font-size:0.78rem;font-weight:700;color:#e2e8f0;margin-bottom:4px">Science Diagrams</div>
            <div style="font-size:0.7rem;color:#64748b">Labelled diagrams, chemical equations, physics numericals</div>
          </div>
          <div style="background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.18);border-radius:12px;padding:14px">
            <div style="font-size:1.1rem;margin-bottom:6px">✍️</div>
            <div style="font-size:0.78rem;font-weight:700;color:#e2e8f0;margin-bottom:4px">Handwritten Questions</div>
            <div style="font-size:0.7rem;color:#64748b">Works with clearly written homework and classwork doubts</div>
          </div>
          <div style="background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.18);border-radius:12px;padding:14px">
            <div style="font-size:1.1rem;margin-bottom:6px">📖</div>
            <div style="font-size:0.78rem;font-weight:700;color:#e2e8f0;margin-bottom:4px">Textbook Exercises</div>
            <div style="font-size:0.7rem;color:#64748b">Snap any NCERT exercise question for instant answers</div>
          </div>
        </div>
        <div style="margin-top:14px;padding:10px 14px;background:rgba(255,255,255,0.02);
             border:1px solid rgba(255,255,255,0.06);border-radius:10px;font-size:0.72rem;color:#64748b">
          💡 <strong style="color:#94a3b8">Best results:</strong>
          Good lighting · Flat surface · Question fully visible · Avoid shadows
        </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — CHAPTERS
# ════════════════════════════════════════════════════════════════

with tab_idx:
    subjs_show=["Mathematics","Science"] if subj=="Both" else [subj]
    progress=get_chapter_progress()

    if st.session_state.selected_chapter and not st.session_state.chapter_mode:
        ch_key,ch_s=st.session_state.selected_chapter
        ch_title=CHAPTER_INDEX.get(cls,{}).get(ch_s,{}).get(ch_key,ch_key)
        mark_chapter_visited(ch_key,cls,ch_s)
        if st.button("← Back",key="back_idx"):
            st.session_state.selected_chapter=None; st.rerun()
        visits=progress.get(ch_key,0); visit_color="#10b981" if visits else "#a78bfa"
        visit_label=f"✅ Visited {visits}×" if visits else "🆕 First visit!"
        st.markdown(f"""
        <div class="bf-card" style="border-color:{c_col}40;margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <div style="font-size:0.66rem;color:#a78bfa;font-weight:700;margin-bottom:3px">
                {CLASS_ICONS.get(cls,'')} {cls} · {SUBJECT_ICONS.get(ch_s,'')} {ch_s}</div>
              <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0">{ch_title}</div>
            </div>
            <span style="background:rgba(16,185,129,0.1);border:1px solid {visit_color}40;border-radius:8px;
                  padding:4px 10px;font-size:0.66rem;color:{visit_color};font-weight:700">{visit_label}</span>
          </div>
        </div>""",unsafe_allow_html=True)
        with st.spinner("📖 Loading overview…"):
            summary=generate_chapter_summary(ch_key,ch_title,cls,ch_s)
        render_answer_with_math(summary)
        st.divider()
        if st.button("💬 Chat about this Chapter",use_container_width=True,type="primary"):
            st.session_state.update(chapter_mode=True,selected_subject=ch_s,
                messages=[{"role":"assistant","content":f"📖 Ready! Ask anything about **{ch_title}** ({cls}·{ch_s}).","chunks":[]}])
            st.rerun()
    else:
        done_count=sum(1 for s in CHAPTER_INDEX.get(cls,{}) for k in CHAPTER_INDEX[cls][s] if k in progress)
        total_count=sum(len(v) for v in CHAPTER_INDEX.get(cls,{}).values())
        st.markdown(f"""
        <div style="margin-bottom:14px;padding:10px 14px;background:rgba(124,58,237,0.05);
             border:1px solid rgba(124,58,237,0.18);border-radius:10px">
          <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#94a3b8;margin-bottom:5px">
            <span>📊 {cls} Progress</span>
            <span style="color:#a78bfa;font-weight:700">{done_count}/{total_count} chapters visited</span>
          </div>
          <div class="prog-bar-wrap">
            <div class="prog-bar" style="width:{int((done_count/total_count)*100) if total_count else 0}%;background:linear-gradient(90deg,#7c3aed,#06b6d4)"></div>
          </div>
        </div>""",unsafe_allow_html=True)
        for s in subjs_show:
            chapters=CHAPTER_INDEX.get(cls,{}).get(s,{})
            if not chapters: continue
            st.markdown(f"#### {SUBJECT_ICONS.get(s,'')} {cls} — {s}")
            for ch_key,ch_title in chapters.items():
                parts=ch_title.split(" — ",1); ch_num=parts[0] if len(parts)>1 else ""; ch_nm=parts[1] if len(parts)>1 else ch_title
                visits=progress.get(ch_key,0)
                badge=(f'<span class="ch-visited">✅ {visits}× visited</span>' if visits else '<span class="ch-new">New</span>')
                a,b=st.columns([6,1])
                with a:
                    st.markdown(f"""<div class="bf-card">
                      <div style="font-size:0.66rem;color:#7c3aed;font-weight:700">{ch_num} {badge}</div>
                      <div style="font-size:0.86rem;font-weight:600;color:#e2e8f0">{ch_nm}</div>
                    </div>""",unsafe_allow_html=True)
                with b:
                    st.markdown("<br><br>",unsafe_allow_html=True)
                    if st.button("Open",key=f"ch_{cls}_{s}_{ch_key}"):
                        st.session_state.selected_chapter=(ch_key,s); st.rerun()
            st.markdown("")

# ════════════════════════════════════════════════════════════════
# TAB 4 — QUIZ
# ════════════════════════════════════════════════════════════════

with tab_quiz:
    quiz_tab,review_tab=st.tabs(["⚡ New Question","📋 Review Mistakes"])
    with quiz_tab:
        st.markdown("#### 🎯 Quick Quiz")
        quiz_topic=st.text_input("Topic",placeholder="e.g. Photosynthesis, Newton's Laws, Trigonometry…",key="qtopic")
        g1,_=st.columns([1,2])
        with g1: gen_btn=st.button("⚡ Generate Question",use_container_width=True,type="primary")
        if gen_btn:
            if not quiz_topic.strip(): st.warning("Enter a topic first.",icon="⚠️")
            elif rate_limit_check():
                with st.spinner("Generating MCQ from NCERT…"):
                    chunks=retrieve_chunks(quiz_topic,cls,subj,"",top_k=4)
                    q_data=generate_quiz(quiz_topic,cls,subj,chunks)
                if q_data:
                    st.session_state.quiz_state=q_data; st.session_state.quiz_answered=False
                    st.session_state.quiz_topic_last=quiz_topic.strip()
                    st.session_state.usage=increment_usage()
                else: st.error("Could not generate quiz. Try a more specific topic.",icon="❌")
        if st.session_state.quiz_state:
            q=st.session_state.quiz_state; answered=st.session_state.quiz_answered
            correct_letter=q.get("correct","A").upper()
            st.markdown(f"""<div class="bf-card" style="border-color:rgba(124,58,237,0.38);margin-top:10px">
              <div style="font-size:0.66rem;color:#a78bfa;font-weight:700;margin-bottom:5px">
                {CLASS_ICONS.get(cls,'')} {cls} · {SUBJECT_ICONS.get(subj,'')} {subj}</div>
              <div style="font-size:0.93rem;font-weight:700;color:#e2e8f0;line-height:1.5">{q['question']}</div>
            </div>""",unsafe_allow_html=True)
            for opt in q["options"]:
                letter=opt[0].upper()
                if not answered:
                    if st.button(opt,key=f"opt_{letter}",use_container_width=True):
                        is_correct=(letter==correct_letter)
                        st.session_state.quiz_answered=True; st.session_state.quiz_last_pick=letter
                        save_quiz_attempt(st.session_state.quiz_topic_last,q,letter,is_correct); st.rerun()
                else:
                    chosen=st.session_state.get("quiz_last_pick","")
                    sty="correct" if letter==correct_letter else ("incorrect" if letter==chosen else "")
                    ico="✅" if letter==correct_letter else ("❌" if letter==chosen else "")
                    st.markdown(f'<div class="quiz-opt {sty}">{ico} {opt}</div>',unsafe_allow_html=True)
            if answered:
                chosen=st.session_state.get("quiz_last_pick","")
                if chosen==correct_letter: st.success("🎉 Correct!")
                else: st.error(f"Correct answer: **{correct_letter}**")
                st.markdown(f"""<div style="background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.18);
                     border-radius:10px;padding:11px 15px;margin-top:8px;font-size:0.82rem;color:#cbd5e1">
                  💡 <strong>Explanation:</strong> {q.get('explanation','')}</div>""",unsafe_allow_html=True)
                n1,n2=st.columns(2)
                with n1:
                    if st.button("🔄 New Question",use_container_width=True):
                        st.session_state.quiz_state=None; st.session_state.quiz_answered=False; st.rerun()
                with n2:
                    if st.button("💬 Ask in Chat",use_container_width=True): st.rerun()

    with review_tab:
        total_q,correct_q=get_quiz_stats(); wrong_q=total_q-correct_q
        acc=int((correct_q/total_q)*100) if total_q else 0
        if total_q:
            r1,r2,r3=st.columns(3)
            for col,val,label,color in [(r1,total_q,"Attempted","#a78bfa"),(r2,correct_q,"Correct","#10b981"),(r3,wrong_q,"Wrong","#f87171")]:
                with col:
                    st.markdown(f"""<div style="text-align:center;padding:12px;background:rgba(255,255,255,0.03);
                         border:1px solid rgba(255,255,255,0.07);border-radius:10px">
                      <div style="font-size:1.4rem;font-weight:800;color:{color}">{val}</div>
                      <div style="font-size:0.65rem;color:#64748b">{label}</div></div>""",unsafe_allow_html=True)
            st.markdown("")
        wrong_list=get_wrong_answers()
        if not wrong_list:
            st.markdown("""<div style="text-align:center;padding:40px 20px;color:#64748b">
              <div style="font-size:2rem">🎉</div>
              <div style="font-weight:600;margin-top:8px">No wrong answers yet!</div></div>""",unsafe_allow_html=True)
        else:
            st.markdown(f"#### 📋 {len(wrong_list)} Wrong Answers to Review")
            for wa in wrong_list:
                try: opts=json.loads(wa.get("options","[]"))
                except: opts=[]
                correct_text=next((o for o in opts if o.startswith(wa.get("correct_answer",""))),wa.get("correct_answer",""))
                picked_text=next((o for o in opts if o.startswith(wa.get("user_answer",""))),wa.get("user_answer",""))
                st.markdown(f"""<div class="wrong-card">
                  <div class="wc-topic">📌 {wa.get('topic','?')} · {CLASS_ICONS.get(wa.get('class',''),'')} {wa.get('class','')} · {wa.get('created_at','')[:10]}</div>
                  <div class="wc-q">{wa.get('question','')}</div>
                  <div class="wc-ans">❌ <span style="color:#f87171">You answered:</span> {picked_text}<br>✅ <span style="color:#10b981">Correct:</span> {correct_text}</div>
                  <div class="wc-exp">💡 {wa.get('explanation','')}</div>
                </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 5 — NOTES
# ════════════════════════════════════════════════════════════════

with tab_notes:
    st.markdown("#### 🗒️ Saved Notes")
    with st.expander("✏️ Add a custom note"):
        note_txt=st.text_area("Your note",placeholder="Write something to remember…",height=90,key="new_note")
        if st.button("💾 Save Note",use_container_width=True):
            if note_txt.strip():
                ok=save_note(note_txt.strip(),cls,subj,"Manual")
                st.toast("✅ Saved!" if ok else "❌ Could not save.",icon="🗒️")
                if ok: st.rerun()
            else: st.warning("Note is empty.",icon="⚠️")
    st.markdown("---")
    notes=get_notes(cls)
    if not notes:
        st.markdown("""<div style="text-align:center;padding:36px 20px;color:#64748b">
          <div style="font-size:2rem">🗒️</div>
          <div style="font-weight:600;margin-top:6px">No notes yet</div>
          <div style="font-size:0.78rem;margin-top:4px">Tap "Save" under any answer</div></div>""",unsafe_allow_html=True)
    else:
        for note in notes:
            ts=note.get("created_at","")[:16].replace("T"," "); ch=note.get("chapter","")
            txt=note.get("content",""); nid=note.get("id")
            display=txt[:400]+("…" if len(txt)>400 else "")
            na,nb=st.columns([8,1])
            with na:
                ch_label=f"· 📸 Image Doubt" if ch=="Image Doubt" else (f"· 📖 {ch}" if ch and ch not in ("General","Manual") else "")
                st.markdown(f"""<div class="note-card">
                  <div style="font-size:0.65rem;color:#64748b;margin-bottom:3px">🕐 {ts}</div>
                  <div style="font-size:0.66rem;color:#a78bfa;font-weight:700;margin-bottom:5px">
                    {CLASS_ICONS.get(note.get('class',''),'')} {note.get('class','')} ·
                    {SUBJECT_ICONS.get(note.get('subject',''),'')} {note.get('subject','')} {ch_label}</div>
                  <div style="font-size:0.82rem;color:#cbd5e1;line-height:1.6">{display}</div>
                </div>""",unsafe_allow_html=True)
            with nb:
                if st.button("🗑️",key=f"del_{nid}",help="Delete"):
                    delete_note(nid); st.rerun()
