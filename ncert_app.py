# ================================================================
# BrainForge — NCERT Chat (Class 8, 9, 10 | Maths & Science)
# Upgraded: pgvector on Supabase, rate limiting, quiz mode,
#           save notes, mobile-first UI, Streamlit Cloud ready
# ================================================================
# requirements.txt:
#   streamlit
#   groq
#   supabase
#   sentence-transformers
#   pymupdf
#   requests
# ================================================================

import os
import re
import uuid
import json
import hashlib
import datetime
import streamlit as st
from groq import Groq

# ── Supabase ──────────────────────────────────────────────────
from supabase import create_client, Client

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

GROQ_MODEL      = "llama-3.3-70b-versatile"
MAX_HISTORY     = 6
DAILY_LIMIT     = 15          # requests per IP per day
TOP_K           = 6           # chunks to retrieve

CLASSES  = ["Class 8", "Class 9", "Class 10"]
SUBJECTS = ["Both", "Mathematics", "Science"]

SUBJECT_ICONS = {"Mathematics": "📐", "Science": "🔬", "Both": "📚"}
CLASS_ICONS   = {"Class 8": "8️⃣",  "Class 9": "9️⃣",  "Class 10": "🔟"}
CLASS_COLORS  = {"Class 8": "#7c3aed", "Class 9": "#0284c7", "Class 10": "#059669"}

# ── Chapter Index ─────────────────────────────────────────────
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
        "Both":        ["What is photosynthesis?", "What are rational numbers?", "Explain cell structure", "What is friction?", "Metals vs non-metals?", "Linear equations?"],
        "Mathematics": ["What are rational numbers?", "Explain linear equations", "What is factorisation?", "Area of a trapezium?", "Algebraic expressions?", "Comparing quantities?"],
        "Science":     ["What is photosynthesis?", "Explain cell structure", "What is friction?", "How does sound travel?", "What are microorganisms?", "Force and pressure?"],
    },
    "Class 9": {
        "Both":        ["What are irrational numbers?", "Newton's laws?", "What is a polynomial?", "How does gravitation work?", "What are tissues?", "Coordinate geometry?"],
        "Mathematics": ["What are irrational numbers?", "Heron's formula?", "Coordinate geometry?", "What are polynomials?", "Euclid's geometry?", "Lines and angles?"],
        "Science":     ["Newton's three laws?", "What is gravitation?", "Atoms vs molecules?", "What are tissues?", "How is sound produced?", "Work and energy?"],
    },
    "Class 10": {
        "Both":        ["What are real numbers?", "Chemical reactions?", "What is trigonometry?", "Acids vs bases?", "What is probability?", "Light reflection?"],
        "Mathematics": ["What are real numbers?", "Quadratic equations?", "What is trigonometry?", "Arithmetic progressions?", "Surface area of cone?", "What is probability?"],
        "Science":     ["Chemical reactions?", "Acids vs bases?", "What is heredity?", "How does the human eye work?", "What is electricity?", "Life processes?"],
    },
}

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS  (mobile-first)
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BrainForge — NCERT AI Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",   # mobile: sidebar hidden by default
)

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
:root {
  --bg:        #06070f;
  --surface:   #0d1117;
  --border:    rgba(255,255,255,0.07);
  --accent:    #7c3aed;
  --accent2:   #06b6d4;
  --text:      #e2e8f0;
  --muted:     #64748b;
  --success:   #10b981;
  --warn:      #f59e0b;
  --danger:    #ef4444;
  --radius:    14px;
  --font:      'Space Grotesk', sans-serif;
  --mono:      'JetBrains Mono', monospace;
}

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 1rem 5rem 1rem !important; max-width: 860px !important; margin: auto; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; font-family: var(--font) !important; }

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: var(--font) !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  padding: 0.55rem 1rem !important;
  transition: all 0.18s ease !important;
  width: 100% !important;
}
.stButton > button:hover {
  background: #6d28d9 !important;
  transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
}
.stButton > button[kind="secondary"]:hover {
  background: rgba(255,255,255,0.1) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
  background: rgba(255,255,255,0.03) !important;
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  padding: 14px 16px !important;
  margin-bottom: 8px !important;
}
div[data-testid="stChatMessage"] * { color: var(--text) !important; font-family: var(--font) !important; }

/* ── Chat input ── */
div[data-testid="stChatInput"] textarea {
  background: var(--surface) !important;
  border: 1px solid rgba(124,58,237,0.4) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}

/* ── Expanders ── */
details summary { color: var(--text) !important; font-family: var(--font) !important; }
details { background: var(--surface) !important; border-radius: 10px !important; border: 1px solid var(--border) !important; }

/* ── Labels ── */
label, .stRadio label, div[data-testid="stWidgetLabel"] p {
  color: var(--muted) !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  font-family: var(--font) !important;
}

/* ── Radio ── */
.stRadio > div { gap: 4px !important; }
.stRadio > div > label {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 6px 12px !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  font-size: 0.85rem !important;
  color: var(--text) !important;
}
.stRadio > div > label:has(input:checked) {
  border-color: var(--accent) !important;
  background: rgba(124,58,237,0.12) !important;
  color: #a78bfa !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 4px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important;
  color: var(--muted) !important;
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: #fff !important;
}

/* ── Progress bars ── */
.stProgress > div > div { background: var(--accent) !important; border-radius: 99px !important; }
.stProgress > div { background: rgba(255,255,255,0.06) !important; border-radius: 99px !important; }

/* ── Cards / custom ── */
.bf-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: 10px;
  transition: border-color 0.2s;
}
.bf-card:hover { border-color: rgba(124,58,237,0.35); }

.pill {
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(124,58,237,0.12);
  border: 1px solid rgba(124,58,237,0.25);
  border-radius: 99px;
  padding: 3px 10px;
  font-size: 0.72rem; font-weight: 700; color: #a78bfa;
  margin-right: 4px; margin-bottom: 4px;
}
.pill.green  { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.25); color: #6ee7b7; }
.pill.blue   { background: rgba(6,182,212,0.1);  border-color: rgba(6,182,212,0.25);  color: #67e8f9; }
.pill.warn   { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.25); color: #fcd34d; }
.pill.red    { background: rgba(239,68,68,0.1);  border-color: rgba(239,68,68,0.25);  color: #fca5a5; }

.rate-bar-wrap { background: rgba(255,255,255,0.05); border-radius: 99px; height: 6px; margin: 6px 0; overflow: hidden; }
.rate-bar      { height: 6px; border-radius: 99px; background: linear-gradient(90deg, #7c3aed, #06b6d4); transition: width 0.4s ease; }

/* ── Quiz options ── */
.quiz-opt {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 6px;
  cursor: pointer;
  font-size: 0.88rem;
  transition: all 0.15s;
}
.quiz-opt:hover      { border-color: rgba(124,58,237,0.4); background: rgba(124,58,237,0.07); }
.quiz-opt.correct    { border-color: var(--success) !important; background: rgba(16,185,129,0.1) !important; }
.quiz-opt.incorrect  { border-color: var(--danger) !important;  background: rgba(239,68,68,0.1) !important;  }

/* ── Mobile breakpoints ── */
@media (max-width: 640px) {
  .block-container { padding: 0.5rem 0.5rem 5rem 0.5rem !important; }
  h1 { font-size: 1.3rem !important; }
  .stTabs [data-baseweb="tab"] { font-size: 0.75rem !important; padding: 6px 8px !important; }
}

/* ── Source cards ── */
.src-card {
  border-left: 3px solid var(--accent);
  background: rgba(255,255,255,0.02);
  border-radius: 0 10px 10px 0;
  padding: 10px 14px;
  margin-bottom: 8px;
  font-size: 0.8rem;
}

/* ── Note cards ── */
.note-card {
  background: rgba(6,182,212,0.05);
  border: 1px solid rgba(6,182,212,0.2);
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 10px;
  position: relative;
}
.note-ts { font-size: 0.68rem; color: var(--muted); margin-bottom: 4px; }
.note-ch { font-size: 0.72rem; color: #67e8f9; font-weight: 700; margin-bottom: 6px; }
.note-body { font-size: 0.85rem; color: var(--text); line-height: 1.55; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SECRETS / CLIENTS
# ════════════════════════════════════════════════════════════════

def _get_secret(key: str, fallback_env: str = None) -> str:
    """Gracefully fetch from st.secrets or env."""
    try:
        return st.secrets[key]
    except Exception:
        pass
    return os.getenv(fallback_env or key, "")


GROQ_API_KEY    = _get_secret("GROQ_API_KEY")
SUPABASE_URL    = _get_secret("SUPABASE_URL")
SUPABASE_KEY    = _get_secret("SUPABASE_KEY")   # anon/service key


@st.cache_resource
def get_groq():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


@st.cache_resource
def get_supabase() -> Client | None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


groq_client = get_groq()
sb: Client  = get_supabase()

# ════════════════════════════════════════════════════════════════
# GUARD — missing secrets
# ════════════════════════════════════════════════════════════════

missing = []
if not GROQ_API_KEY:  missing.append("GROQ_API_KEY")
if not SUPABASE_URL:  missing.append("SUPABASE_URL")
if not SUPABASE_KEY:  missing.append("SUPABASE_KEY")

if missing:
    st.markdown("""
    <div style="max-width:520px;margin:80px auto;padding:32px;
         background:#0d1117;border:1px solid #7c3aed44;border-radius:18px;text-align:center;">
      <div style="font-size:2.5rem;margin-bottom:12px;">🔑</div>
      <h2 style="font-family:'Space Grotesk',sans-serif;color:#e2e8f0;margin:0 0 8px;">
        Missing Secrets
      </h2>
      <p style="color:#64748b;font-size:0.88rem;margin:0 0 18px;">
        Add these in <strong>Streamlit Cloud → Settings → Secrets</strong>:
      </p>
    """, unsafe_allow_html=True)
    st.code("\n".join(f'{k} = "your_{k.lower()}_here"' for k in missing), language="toml")
    st.stop()

# ════════════════════════════════════════════════════════════════
# RATE LIMITING  (Supabase table: rate_limits)
# ════════════════════════════════════════════════════════════════
# Table schema (run once in Supabase SQL editor):
#
#   create table rate_limits (
#     id         bigserial primary key,
#     user_hash  text not null,
#     day        date not null default current_date,
#     count      int  not null default 0,
#     unique (user_hash, day)
#   );

def _user_hash() -> str:
    """Deterministic per-session fingerprint (not truly unique — good enough)."""
    raw = st.context.headers.get("X-Forwarded-For", "anon") + str(st.runtime.exists())
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def get_usage() -> int:
    """How many requests has this user made today?"""
    try:
        h   = _user_hash()
        today = datetime.date.today().isoformat()
        res = sb.table("rate_limits").select("count").eq("user_hash", h).eq("day", today).execute()
        if res.data:
            return res.data[0]["count"]
    except Exception:
        pass
    return 0


def increment_usage() -> int:
    """Upsert +1. Returns new count."""
    try:
        h     = _user_hash()
        today = datetime.date.today().isoformat()
        res   = sb.table("rate_limits").select("count").eq("user_hash", h).eq("day", today).execute()
        if res.data:
            new_count = res.data[0]["count"] + 1
            sb.table("rate_limits").update({"count": new_count}).eq("user_hash", h).eq("day", today).execute()
        else:
            new_count = 1
            sb.table("rate_limits").insert({"user_hash": h, "day": today, "count": 1}).execute()
        return new_count
    except Exception:
        return 0   # fail open — don't block on DB errors

# ════════════════════════════════════════════════════════════════
# VECTOR SEARCH  (Supabase pgvector)
# ════════════════════════════════════════════════════════════════
# Table schema (run once in Supabase SQL editor):
#
#   create extension if not exists vector;
#
#   create table ncert_chunks (
#     id        bigserial primary key,
#     content   text,
#     embedding vector(384),
#     class     text,
#     subject   text,
#     chapter   text,
#     source    text,
#     page      int
#   );
#
#   create index on ncert_chunks using ivfflat (embedding vector_cosine_ops)
#     with (lists = 100);
#
#   -- RPC function for similarity search:
#   create or replace function match_chunks(
#     query_embedding vector(384),
#     filter_class    text,
#     filter_subject  text,
#     filter_chapter  text,
#     match_count     int
#   )
#   returns table (
#     id int8, content text, class text, subject text,
#     chapter text, source text, page int, similarity float
#   )
#   language sql stable as $$
#     select id, content, class, subject, chapter, source, page,
#            1 - (embedding <=> query_embedding) as similarity
#     from   ncert_chunks
#     where  class   = filter_class
#       and  (filter_subject = 'Both' or subject = filter_subject)
#       and  (filter_chapter = ''    or chapter  = filter_chapter)
#     order  by embedding <=> query_embedding
#     limit  match_count;
#   $$;


@st.cache_resource(show_spinner=False)
def get_embedder():
    """Load sentence-transformer once; cached across reruns."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def embed(text: str) -> list[float] | None:
    model = get_embedder()
    if model is None:
        return None
    return model.encode(text, normalize_embeddings=True).tolist()


def retrieve_chunks(query: str, class_label: str, subject: str,
                    chapter_filter: str = "", top_k: int = TOP_K) -> list:
    q_emb = embed(query)
    if q_emb is None or sb is None:
        return []
    try:
        res = sb.rpc("match_chunks", {
            "query_embedding": q_emb,
            "filter_class":    class_label,
            "filter_subject":  subject,
            "filter_chapter":  chapter_filter or "",
            "match_count":     top_k,
        }).execute()
        chunks = []
        for row in (res.data or []):
            sim = float(row.get("similarity", 0))
            if sim > 0.15:
                chunks.append({
                    "text":      row["content"],
                    "subject":   row["subject"],
                    "chapter":   row["chapter"],
                    "page":      row.get("page", "?"),
                    "source":    row.get("source", ""),
                    "class":     row["class"],
                    "relevance": round(sim * 100, 1),
                })
        return sorted(chunks, key=lambda x: x["relevance"], reverse=True)
    except Exception as e:
        st.warning(f"Vector search error: {e}", icon="⚠️")
        return []

# ════════════════════════════════════════════════════════════════
# NOTES  (Supabase table: notes)
# ════════════════════════════════════════════════════════════════
# Table schema:
#   create table notes (
#     id         bigserial primary key,
#     user_hash  text not null,
#     class      text,
#     subject    text,
#     chapter    text,
#     content    text,
#     created_at timestamptz default now()
#   );

def save_note(content: str, class_label: str, subject: str, chapter: str = "") -> bool:
    try:
        sb.table("notes").insert({
            "user_hash": _user_hash(),
            "class":     class_label,
            "subject":   subject,
            "chapter":   chapter,
            "content":   content,
        }).execute()
        return True
    except Exception:
        return False


def get_notes(class_label: str = None) -> list:
    try:
        q = sb.table("notes").select("*").eq("user_hash", _user_hash()).order("created_at", desc=True)
        if class_label:
            q = q.eq("class", class_label)
        return (q.limit(50).execute()).data or []
    except Exception:
        return []


def delete_note(note_id: int) -> bool:
    try:
        sb.table("notes").delete().eq("id", note_id).eq("user_hash", _user_hash()).execute()
        return True
    except Exception:
        return False

# ════════════════════════════════════════════════════════════════
# GROQ — ANSWER GENERATION
# ════════════════════════════════════════════════════════════════

def build_context(chunks: list) -> str:
    if not chunks:
        return ""
    out = "\n--- NCERT CONTENT ---\n"
    for c in chunks[:6]:
        out += f"\n[{c['class']} | {c['subject']} | {c['chapter']} | p.{c['page']}]\n{c['text']}\n"
    return out + "\n--- END ---\n"


def generate_answer(question: str, chunks: list, class_label: str, subject: str,
                    style: str, chat_history: list, chapter_context: str = None) -> str:

    age_map = {"Class 8": "13–14", "Class 9": "14–15", "Class 10": "15–16"}
    style_map = {
        "Simple":   "Use simple language, short paragraphs. Avoid jargon.",
        "Detailed": "Give a comprehensive explanation covering all sub-topics in depth.",
        "Bullets":  "Use numbered headings and bullet points throughout.",
        "Examples": "Include 2–3 relatable real-life Indian examples after each concept.",
    }
    style_instr = next((v for k, v in style_map.items() if k in style),
                       "Use simple, clear language.")
    ch_ctx = f"\nStudent is studying: {chapter_context}" if chapter_context else ""

    system = f"""You are BrainForge — expert AI tutor for CBSE India students.

PROCESS:
1. Read the NCERT content snippets provided.
2. Rewrite as a clear, structured teaching response — never copy-paste.
3. Enrich with analogies, memory tricks, real-world Indian connections.
4. For Class 10: add board exam tips and key formulas in a box.

STYLE: {style_instr}
- **Bold** key terms
- Add 💡 Quick Tip or 🔑 Key Point where helpful
- Use Indian context (cricket, food, daily life) where natural

STUDENT: {class_label} CBSE, age {age_map.get(class_label, '13–16')}
{ch_ctx}

FORMAT:
## [Topic]
[Explanation]

**Key Points:**
- ...

💡 Quick Tip: [memory trick or exam tip]

📚 Source: NCERT {class_label} {subject}
💬 Want examples, a quiz, or a deeper explanation?"""

    messages = [{"role": "system", "content": system}]
    for m in chat_history[-(MAX_HISTORY * 2):]:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({
        "role": "user",
        "content": f"Question: {question}\n{build_context(chunks)}\nAnswer for a {class_label} student.",
    })

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            temperature=0.4, max_tokens=1400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "rate_limit" in err.lower() or "429" in err:
            return "⚠️ AI rate limit hit. Please wait 30 seconds and try again."
        return f"⚠️ Answer generation failed: {err}"


def generate_quiz(topic: str, class_label: str, subject: str, chunks: list) -> dict | None:
    """Generate a 4-option MCQ as JSON."""
    ctx = build_context(chunks[:4])
    prompt = f"""You are a CBSE exam question setter.
Generate ONE multiple-choice question about: "{topic}"
Class: {class_label} | Subject: {subject}
{ctx}

Respond ONLY with valid JSON — no markdown, no extra text:
{{
  "question": "...",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "correct": "A",
  "explanation": "Brief explanation why correct answer is right."
}}
"""
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return None


def generate_chapter_summary(ch_key: str, ch_title: str,
                              class_label: str, subject: str) -> str:
    chunks  = retrieve_chunks(ch_title, class_label, subject, ch_key, top_k=8)
    ctx     = "\n".join(c["text"] for c in chunks[:6])
    prompt  = f"""You are BrainForge — expert {class_label} CBSE teacher.
Student opened: {ch_title} ({class_label} · {subject})

NCERT content:
{ctx}

Write a CHAPTER OVERVIEW:
## 📖 {ch_title}
### What You Will Learn
[3–5 bullet points]
### Key Concepts
[4–6 concepts with 1-line explanations]
### Why This Chapter Matters
[1–2 lines on real-life relevance or board exam importance]
### Quick Preview
[2–3 engaging sentences]
End with: "💬 Ask me anything about this chapter!"
"""
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate summary: {e}"


def is_followup(question: str, history: list) -> bool:
    if not history:
        return False
    patterns = [
        r"^(what about|tell me more|explain more|can you|how about|give me|more about)",
        r"^(example|examples|illustrate|show me)",
        r"^(i (don't|do not) understand|unclear|confused|simpler)",
        r"^(test me|quiz me|ask me|mcq)",
    ]
    q = question.lower().strip()
    if len(q.split()) <= 5:
        for p in patterns:
            if re.match(p, q):
                return True
    return False

# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "messages":         [],
        "selected_chapter": None,
        "chapter_mode":     False,
        "selected_class":   "Class 8",
        "selected_subject": "Both",
        "quiz_state":       None,    # dict with current quiz
        "quiz_answered":    False,
        "usage":            None,    # cached usage count
        "active_tab":       0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px;border-bottom:1px solid rgba(255,255,255,0.06);">
      <div style="font-size:1.5rem;font-weight:800;color:#e2e8f0;">🧠 BrainForge</div>
      <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">NCERT Class 8 · 9 · 10</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎓 Class")
    sel_class = st.radio("class", CLASSES,
        format_func=lambda x: f"{CLASS_ICONS[x]} {x}",
        label_visibility="collapsed",
        key="class_radio",
    )
    if sel_class != st.session_state.selected_class:
        st.session_state.selected_class   = sel_class
        st.session_state.selected_chapter = None
        st.session_state.chapter_mode     = False
        st.session_state.messages         = []
        st.session_state.quiz_state       = None

    st.markdown("#### 📚 Subject")
    sel_subj = st.radio("subj", SUBJECTS,
        format_func=lambda x: f"{SUBJECT_ICONS[x]} {x}",
        label_visibility="collapsed",
        key="subject_radio",
    )
    if sel_subj != st.session_state.selected_subject:
        st.session_state.selected_subject = sel_subj
        st.session_state.selected_chapter = None
        st.session_state.chapter_mode     = False
        st.session_state.messages         = []
        st.session_state.quiz_state       = None

    st.markdown("---")

    # Usage meter
    if st.session_state.usage is None:
        st.session_state.usage = get_usage()
    used = st.session_state.usage
    pct  = min(used / DAILY_LIMIT, 1.0)
    color_cls = "green" if pct < 0.6 else ("warn" if pct < 0.9 else "red")
    st.markdown(f"""
    <div style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <span style="font-size:0.75rem;color:#94a3b8;font-weight:600;">Daily Usage</span>
        <span class="pill {color_cls}">{used}/{DAILY_LIMIT}</span>
      </div>
      <div class="rate-bar-wrap">
        <div class="rate-bar" style="width:{int(pct*100)}%;
          background:{'linear-gradient(90deg,#7c3aed,#06b6d4)' if pct<0.9 else 'linear-gradient(90deg,#f59e0b,#ef4444)'};"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    show_sources = st.toggle("📄 Show NCERT sources", value=False)
    answer_depth = st.selectbox("Answer style", ["Simple", "Detailed", "Bullets", "Examples"])

    if st.session_state.messages:
        turns = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f'<div class="pill green">💬 {turns} question{"s" if turns!=1 else ""}</div>',
                    unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages         = []
        st.session_state.selected_chapter = None
        st.session_state.chapter_mode     = False
        st.session_state.quiz_state       = None
        st.rerun()

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

cls  = st.session_state.selected_class
subj = st.session_state.selected_subject
c_color = CLASS_COLORS.get(cls, "#7c3aed")

st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(124,58,237,0.1),rgba(6,182,212,0.06));
     border:1px solid {c_color}33;border-radius:18px;padding:18px 22px;margin-bottom:18px;">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <span style="font-size:1.8rem;">{CLASS_ICONS.get(cls,'🎓')}</span>
    <div>
      <div style="font-size:1.2rem;font-weight:800;color:#e2e8f0;line-height:1.2;">NCERT AI Tutor</div>
      <div style="font-size:0.78rem;color:#64748b;">{cls} · {subj} · powered by NCERT textbooks</div>
    </div>
    <div style="margin-left:auto;display:flex;flex-wrap:wrap;gap:4px;">
      <span class="pill">{CLASS_ICONS.get(cls,'')} {cls}</span>
      <span class="pill blue">{SUBJECT_ICONS.get(subj,'')} {subj}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABS: Chat | Chapters | Quiz | Notes
# ════════════════════════════════════════════════════════════════

tab_chat, tab_idx, tab_quiz, tab_notes = st.tabs(
    ["💬 Chat", "📖 Chapters", "🎯 Quiz", "🗒️ Notes"]
)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def render_sources(chunks: list):
    if not chunks or not show_sources:
        return
    with st.expander(f"📄 {len(chunks)} NCERT passages used"):
        for c in chunks:
            rel = c["relevance"]
            bcolor = "#10b981" if rel >= 70 else "#f59e0b" if rel >= 45 else "#64748b"
            st.markdown(f"""
            <div class="src-card" style="border-left-color:{bcolor};">
              <div style="display:flex;justify-content:space-between;">
                <span style="color:#a78bfa;font-weight:700;font-size:0.78rem;">
                  {CLASS_ICONS.get(c.get('class',''), '')} {c.get('class','')} ·
                  {SUBJECT_ICONS.get(c['subject'],'')} {c['subject']} — {c['chapter'].upper()}
                </span>
                <span style="color:{bcolor};font-size:0.72rem;font-weight:700;">{rel}%</span>
              </div>
              <div style="color:#64748b;font-size:0.7rem;margin:3px 0;">p.{c['page']} · {c['source']}</div>
              <div style="color:#cbd5e1;font-size:0.8rem;line-height:1.55;">
                {c['text'][:260]}{'…' if len(c['text'])>260 else ''}
              </div>
            </div>""", unsafe_allow_html=True)


def rate_limit_check() -> bool:
    """Return True if user is within limit, False if blocked."""
    if st.session_state.usage is None:
        st.session_state.usage = get_usage()
    if st.session_state.usage >= DAILY_LIMIT:
        st.error(
            f"🚫 You've reached your daily limit of **{DAILY_LIMIT} questions**. "
            "Come back tomorrow! 🌅",
            icon="🚫",
        )
        return False
    return True


def process_question(question: str, chapter_context=None, chapter_filter="", chapter_subject=None):
    """Full pipeline: retrieve → generate → track usage."""
    if not rate_limit_check():
        return None, []

    hist   = st.session_state.messages
    chunks = []
    eff_subj = chapter_subject or subj

    if not is_followup(question, hist):
        with st.spinner("🔍 Searching NCERT…"):
            chunks = retrieve_chunks(question, cls, eff_subj, chapter_filter, TOP_K)

    with st.spinner("✍️ Writing your answer…"):
        answer = generate_answer(question, chunks, cls, eff_subj,
                                 answer_depth, hist, chapter_context)

    new_count = increment_usage()
    st.session_state.usage = new_count
    return answer, chunks


# ════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════

with tab_chat:
    chapter_context = None
    chapter_filter  = ""
    chapter_subject = subj

    if st.session_state.chapter_mode and st.session_state.selected_chapter:
        ch_key, ch_subj = st.session_state.selected_chapter
        ch_title = CHAPTER_INDEX.get(cls, {}).get(ch_subj, {}).get(ch_key, ch_key)
        chapter_context = ch_title
        chapter_filter  = ch_key
        chapter_subject = ch_subj

        banner_col, exit_col = st.columns([5, 1])
        with banner_col:
            st.markdown(f"""
            <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.25);
                 border-radius:10px;padding:8px 14px;margin-bottom:10px;">
              <div style="font-size:0.72rem;color:#a78bfa;font-weight:700;">
                📖 Chapter Mode
              </div>
              <div style="font-size:0.85rem;color:#e2e8f0;font-weight:600;">{ch_title}</div>
            </div>""", unsafe_allow_html=True)
        with exit_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✕ Exit", key="exit_ch"):
                st.session_state.chapter_mode     = False
                st.session_state.selected_chapter = None
                st.session_state.messages         = []
                st.rerun()

    # Suggestions when chat is empty
    if not st.session_state.messages:
        st.markdown("##### 💡 Try asking:")
        suggs = SUGGESTIONS.get(cls, {}).get(subj, SUGGESTIONS["Class 8"]["Both"])
        cols = st.columns(2)
        for i, s in enumerate(suggs):
            with cols[i % 2]:
                if st.button(s, key=f"sg{i}"):
                    answer, chunks = process_question(s, chapter_context, chapter_filter, chapter_subject)
                    if answer:
                        st.session_state.messages += [
                            {"role": "user",      "content": s,      "chunks": []},
                            {"role": "assistant", "content": answer, "chunks": chunks},
                        ]
                    st.rerun()
        st.markdown("")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("chunks"):
                    render_sources(msg["chunks"])
                # Save note button under each answer
                if st.button("🗒️ Save as Note", key=f"save_{id(msg)}"):
                    ok = save_note(msg["content"], cls, subj,
                                   chapter_filter if chapter_filter else "General")
                    if ok:
                        st.toast("✅ Note saved!", icon="🗒️")
                    else:
                        st.toast("❌ Could not save note.", icon="⚠️")

    # Input
    if chapter_context:
        ph = f"Ask about {chapter_context}…"
    elif subj != "Both":
        ph = f"Ask anything from {cls} {subj} NCERT…"
    else:
        ph = f"Ask anything from {cls} Maths or Science…"

    question = st.chat_input(ph)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answer, chunks = process_question(question, chapter_context, chapter_filter, chapter_subject)
            if answer:
                st.markdown(answer)
                render_sources(chunks)
                st.session_state.messages += [
                    {"role": "user",      "content": question, "chunks": []},
                    {"role": "assistant", "content": answer,   "chunks": chunks},
                ]


# ════════════════════════════════════════════════════════════════
# TAB 2 — CHAPTER INDEX
# ════════════════════════════════════════════════════════════════

with tab_idx:
    subjects_to_show = ["Mathematics", "Science"] if subj == "Both" else [subj]

    if st.session_state.selected_chapter and not st.session_state.chapter_mode:
        ch_key, ch_subj = st.session_state.selected_chapter
        ch_title = CHAPTER_INDEX.get(cls, {}).get(ch_subj, {}).get(ch_key, ch_key)

        if st.button("← Back", key="back_idx"):
            st.session_state.selected_chapter = None
            st.rerun()

        st.markdown(f"""
        <div class="bf-card" style="border-color:{c_color}44;margin-bottom:14px;">
          <div style="font-size:0.72rem;color:#a78bfa;font-weight:700;margin-bottom:3px;">
            {CLASS_ICONS.get(cls,'')} {cls} · {SUBJECT_ICONS.get(ch_subj,'')} {ch_subj}
          </div>
          <div style="font-size:1rem;font-weight:700;color:#e2e8f0;">{ch_title}</div>
        </div>""", unsafe_allow_html=True)

        with st.spinner("📖 Loading overview…"):
            summary = generate_chapter_summary(ch_key, ch_title, cls, ch_subj)
        st.markdown(summary)
        st.divider()

        if st.button("💬 Chat about this Chapter", use_container_width=True, type="primary"):
            st.session_state.chapter_mode     = True
            st.session_state.selected_subject = ch_subj
            st.session_state.messages         = [{
                "role":    "assistant",
                "content": f"📖 Ready! Ask me anything about **{ch_title}** ({cls} · {ch_subj}).",
                "chunks":  [],
            }]
            st.session_state.active_tab = 0
            st.rerun()

    else:
        for s in subjects_to_show:
            chapters = CHAPTER_INDEX.get(cls, {}).get(s, {})
            if not chapters:
                continue
            st.markdown(f"#### {SUBJECT_ICONS.get(s,'')} {cls} — {s}")
            for ch_key, ch_title in chapters.items():
                parts  = ch_title.split(" — ", 1)
                ch_num = parts[0] if len(parts) > 1 else ""
                ch_nm  = parts[1] if len(parts) > 1 else ch_title
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"""
                    <div class="bf-card">
                      <div style="font-size:0.72rem;color:#7c3aed;font-weight:700;">{ch_num}</div>
                      <div style="font-size:0.9rem;font-weight:600;color:#e2e8f0;">{ch_nm}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("Open", key=f"ch_{cls}_{s}_{ch_key}"):
                        st.session_state.selected_chapter = (ch_key, s)
                        st.rerun()
            st.markdown("")


# ════════════════════════════════════════════════════════════════
# TAB 3 — QUIZ
# ════════════════════════════════════════════════════════════════

with tab_quiz:
    st.markdown("#### 🎯 Quick Quiz")
    st.caption("Test your knowledge with AI-generated MCQs from NCERT")

    quiz_topic = st.text_input(
        "Topic to quiz on",
        placeholder=f"e.g. Photosynthesis, Rational Numbers, Newton's Laws…",
        key="quiz_topic_input",
    )

    gen_col, _ = st.columns([1, 2])
    with gen_col:
        gen_btn = st.button("Generate Question ⚡", use_container_width=True, type="primary")

    if gen_btn:
        if not quiz_topic.strip():
            st.warning("Please enter a topic first.", icon="⚠️")
        elif not rate_limit_check():
            pass
        else:
            with st.spinner("Generating MCQ from NCERT…"):
                chunks = retrieve_chunks(quiz_topic, cls, subj, "", top_k=4)
                q_data = generate_quiz(quiz_topic, cls, subj, chunks)
            if q_data:
                st.session_state.quiz_state    = q_data
                st.session_state.quiz_answered = False
                new_count = increment_usage()
                st.session_state.usage = new_count
            else:
                st.error("Could not generate quiz. Try a more specific topic.", icon="❌")

    if st.session_state.quiz_state:
        q = st.session_state.quiz_state
        answered = st.session_state.quiz_answered

        st.markdown(f"""
        <div class="bf-card" style="border-color:rgba(124,58,237,0.4);margin-top:12px;">
          <div style="font-size:0.72rem;color:#a78bfa;font-weight:700;margin-bottom:6px;">
            {CLASS_ICONS.get(cls,'')} {cls} · {SUBJECT_ICONS.get(subj,'')} {subj}
          </div>
          <div style="font-size:1rem;font-weight:700;color:#e2e8f0;line-height:1.5;">
            {q['question']}
          </div>
        </div>""", unsafe_allow_html=True)

        correct_letter = q.get("correct", "A").upper()

        for opt in q["options"]:
            letter = opt[0].upper()
            if answered:
                style_cls = "correct" if letter == correct_letter else "incorrect"
            else:
                style_cls = ""

            if not answered:
                if st.button(opt, key=f"opt_{letter}", use_container_width=True):
                    st.session_state.quiz_answered = True
                    st.session_state.quiz_last_pick = letter
                    st.rerun()
            else:
                chosen = st.session_state.get("quiz_last_pick", "")
                icon = "✅" if letter == correct_letter else ("❌" if letter == chosen else "")
                st.markdown(f"""
                <div class="quiz-opt {style_cls}">
                  {icon} {opt}
                </div>""", unsafe_allow_html=True)

        if answered:
            chosen = st.session_state.get("quiz_last_pick", "")
            if chosen == correct_letter:
                st.success("🎉 Correct! Well done.")
            else:
                st.error(f"Not quite. The correct answer is **{correct_letter}**.")

            st.markdown(f"""
            <div style="background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.2);
                 border-radius:10px;padding:12px 16px;margin-top:8px;font-size:0.85rem;color:#cbd5e1;">
              💡 <strong>Explanation:</strong> {q.get('explanation', '')}
            </div>""", unsafe_allow_html=True)

            nc1, nc2 = st.columns(2)
            with nc1:
                if st.button("🔄 New Question", use_container_width=True):
                    st.session_state.quiz_state    = None
                    st.session_state.quiz_answered = False
                    st.rerun()
            with nc2:
                if st.button("💬 Ask about this topic", use_container_width=True):
                    st.session_state.active_tab = 0
                    st.rerun()


# ════════════════════════════════════════════════════════════════
# TAB 4 — NOTES
# ════════════════════════════════════════════════════════════════

with tab_notes:
    st.markdown("#### 🗒️ Saved Notes")
    st.caption("Notes are saved per device. Refreshing reloads them.")

    # Quick add
    with st.expander("✏️ Add a custom note"):
        note_text = st.text_area("Your note", placeholder="Write something you want to remember…", height=100)
        if st.button("💾 Save Note", use_container_width=True):
            if note_text.strip():
                ok = save_note(note_text.strip(), cls, subj, "Manual")
                if ok:
                    st.toast("✅ Saved!", icon="🗒️")
                    st.rerun()
                else:
                    st.toast("❌ Could not save.", icon="⚠️")
            else:
                st.warning("Note is empty.", icon="⚠️")

    st.markdown("---")

    notes = get_notes(cls)
    if not notes:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:#64748b;">
          <div style="font-size:2rem;margin-bottom:8px;">🗒️</div>
          <div style="font-weight:600;">No notes yet</div>
          <div style="font-size:0.82rem;margin-top:4px;">
            Hit "Save as Note" under any answer in Chat
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        for note in notes:
            ts  = note.get("created_at", "")[:16].replace("T", " ")
            ch  = note.get("chapter", "")
            txt = note.get("content", "")
            nid = note.get("id")

            # Truncate long AI answers in notes view
            display_txt = txt if len(txt) <= 400 else txt[:400] + "…"

            n1, n2 = st.columns([8, 1])
            with n1:
                st.markdown(f"""
                <div class="note-card">
                  <div class="note-ts">🕐 {ts}</div>
                  <div class="note-ch">
                    {CLASS_ICONS.get(note.get('class',''), '')} {note.get('class','')} ·
                    {SUBJECT_ICONS.get(note.get('subject',''), '')} {note.get('subject','')}
                    {'· 📖 ' + ch if ch and ch != 'General' and ch != 'Manual' else ''}
                  </div>
                  <div class="note-body">{display_txt}</div>
                </div>""", unsafe_allow_html=True)
            with n2:
                if st.button("🗑️", key=f"del_{nid}", help="Delete note"):
                    delete_note(nid)
                    st.rerun()
