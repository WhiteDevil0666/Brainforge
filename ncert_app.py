# ================================================================
# BrainForge — NCERT Chat (Chapter Index + Refined AI Answers)
# ================================================================
# requirements.txt:
#   streamlit
#   groq
#   chromadb==1.5.5
#   pymupdf
#   requests
#   gdown
# ================================================================

import os
import re
import uuid
import zipfile
import gdown
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
import fitz
from groq import Groq

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

CHROMA_DIR      = "./ncert_db"
COLLECTION_NAME = "ncert_class8"
PDF_DIR         = "./ncert_pdfs"
GROQ_MODEL      = "llama-3.3-70b-versatile"
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 100
MAX_HISTORY     = 6

GDRIVE_FILES = {
    "Math.zip":           ("1EUcfrL8JeTz3zkuPHggxHj-dcCOrtgzN", "Mathematics"),
    "Science.zip":        ("1ABT9Fu0Dmmi9AnhqECunbo9ehTt_5Lvk", "Science"),
    "SocialScience1.zip": ("1VLSSrQxa9ljQjv2OZ5xjLM49jK5lh9bf", "History"),
    "SocialScience2.zip": ("1-hXrpg7sSmm3Cvf8QhQ4bIfOsVIkwJ7c", "Geography"),
    "SocialScience3.zip": ("1DI_r-mEgh3uFnAtqDzuztA46tSgNRjcX", "Civics"),
}

SUBJECTS = ["All Subjects", "Mathematics", "Science", "History", "Geography", "Civics"]
SUBJECT_ICONS = {
    "Mathematics": "📐", "Science": "🔬", "History": "🏛️",
    "Geography": "🌍", "Civics": "⚖️", "All Subjects": "📚",
}

# ── Chapter index for each subject ────────────────────────────
# Filename → Chapter Title mapping (Class 8 NCERT)
CHAPTER_INDEX = {
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
    "History": {
        "hess201": "Chapter 1 — How, When and Where",
        "hess202": "Chapter 2 — From Trade to Territory",
        "hess203": "Chapter 3 — Ruling the Countryside",
        "hess204": "Chapter 4 — Tribals, Dikus and the Vision of a Golden Age",
        "hess205": "Chapter 5 — When People Rebel: 1857 and After",
        "hess206": "Chapter 6 — Weavers, Iron Smelters and Factory Owners",
        "hess207": "Chapter 7 — Civilising the Native, Educating the Nation",
        "hess208": "Chapter 8 — Women, Caste and Reform",
        "hess2ps": "Chapter 9 — The Making of the National Movement",
    },
    "Geography": {
        "hess301": "Chapter 1 — Resources",
        "hess302": "Chapter 2 — Land, Soil, Water, Natural Vegetation and Wildlife",
        "hess303": "Chapter 3 — Mineral and Power Resources",
        "hess304": "Chapter 4 — Agriculture",
        "hess305": "Chapter 5 — Industries",
        "hess306": "Chapter 6 — Human Resources",
        "hess307": "Chapter 7 — Human Development",
        "hess308": "Chapter 8 — The United Nations",
        "hess3ps": "Chapter 9 — Public Facilities",
    },
    "Civics": {
        "hess401": "Chapter 1 — The Indian Constitution",
        "hess402": "Chapter 2 — Understanding Secularism",
        "hess403": "Chapter 3 — Why Do We Need a Parliament?",
        "hess404": "Chapter 4 — Understanding Laws",
        "hess405": "Chapter 5 — Judiciary",
        "hess4ps": "Chapter 6 — Social Justice and the Marginalised",
    },
}

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BrainForge — NCERT Chat", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
.stApp { background:linear-gradient(135deg,#0a0f1e,#0f172a,#0a1628); color:#fff; font-family:'Inter',sans-serif; }
h1,h2,h3,h4 { font-family:'Sora',sans-serif !important; color:#fff !important; }
section.main > div { background-color:transparent !important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#060d1a,#0b1525) !important; border-right:1px solid rgba(255,255,255,0.06); }
section[data-testid="stSidebar"] * { color:#fff !important; }
div[data-testid="stChatMessage"] { background:rgba(255,255,255,0.04) !important; border-radius:14px !important; padding:14px !important; border:1px solid rgba(255,255,255,0.07); margin-bottom:8px; }
div[data-testid="stChatMessage"] * { color:#f1f5f9 !important; }
.stButton > button { background:linear-gradient(135deg,#6366f1,#4f46e5); color:white !important; border-radius:12px; font-weight:700; border:none; box-shadow:0 4px 15px rgba(99,102,241,0.3); transition:all 0.2s; }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 20px rgba(99,102,241,0.4); }
div[data-testid="stSelectbox"] label, label[data-testid="stWidgetLabel"] { color:#e2e8f0 !important; font-weight:600 !important; font-size:0.88em !important; }
.source-card { background:rgba(255,255,255,0.03); border-radius:0 10px 10px 0; padding:10px 14px; margin-bottom:8px; }
.stat-pill { display:inline-flex; align-items:center; gap:5px; background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.25); border-radius:20px; padding:4px 12px; font-size:0.78em; font-weight:700; color:#a5b4fc; margin-right:6px; }
.chapter-card {
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:14px;
    padding:14px 18px;
    margin-bottom:10px;
    cursor:pointer;
    transition:border-color 0.2s;
}
.chapter-card:hover { border-color:rgba(99,102,241,0.4); }
.chapter-num { color:#6366f1; font-weight:800; font-size:0.8em; margin-bottom:3px; }
.chapter-title { color:#f1f5f9; font-weight:600; font-size:0.95em; }
.mode-tab { display:inline-flex; align-items:center; gap:6px; padding:8px 18px; border-radius:10px; font-weight:700; font-size:0.85em; cursor:pointer; margin-right:8px; }
.mode-active { background:rgba(99,102,241,0.2); border:1px solid rgba(99,102,241,0.4); color:#a5b4fc; }
.mode-inactive { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); color:#64748b; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# GOOGLE DRIVE + EXTRACTION FUNCTIONS
# ════════════════════════════════════════════════════════════════

def download_from_gdrive(file_id: str, dest_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=True, fuzzy=True)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            return True
        gdown.download(f"https://drive.google.com/file/d/{file_id}/view", dest_path, quiet=True, fuzzy=True)
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000
    except Exception:
        return False


def extract_zip(zip_path: str, extract_to: str, subject: str) -> list:
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                clean_name = os.path.basename(name)
                if not clean_name.lower().endswith(".pdf") or clean_name.startswith(("__",".")):
                    continue
                out_path = os.path.join(extract_to, subject, clean_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with z.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append((out_path, subject, clean_name))
    except Exception:
        pass
    return extracted


def clean_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\u0900-\u097F]', ' ', text)
    text = re.sub(r'\.{3,}', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_pdf_text(pdf_path: str):
    text, pages = "", 0
    try:
        doc   = fitz.open(pdf_path)
        pages = len(doc)
        for i, page in enumerate(doc):
            raw = page.get_text("text")
            if raw.strip():
                text += f"\n[Page {i+1}]\n{clean_text(raw)}"
        doc.close()
    except Exception:
        pass
    return text, pages


def chunk_text(text: str) -> list:
    page_pat = re.compile(r'\[Page (\d+)\]')
    max_page = max((int(m.group(1)) for m in page_pat.finditer(text)), default=1)
    clean    = page_pat.sub('', text).strip()
    total    = max(len(clean), 1)
    chunks, start = [], 0
    while start < len(clean):
        chunk = clean[start:start+CHUNK_SIZE].strip()
        if len(chunk) > 80:
            chunks.append({"text": chunk, "page": max(1, int((start/total)*max_page))})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_pdfs(col, pdf_list, progress_bar, status_text) -> int:
    total_chunks = 0
    for i, (pdf_path, subject, filename) in enumerate(pdf_list):
        chapter = os.path.splitext(filename)[0].lower()
        status_text.text(f"📖 Indexing: {filename} ({i+1}/{len(pdf_list)}) — {subject}")
        progress_bar.progress(int((i / max(len(pdf_list), 1)) * 85) + 10)
        text, _ = extract_pdf_text(pdf_path)
        if not text.strip():
            continue
        chunks = chunk_text(text)
        for j in range(0, len(chunks), 50):
            batch = chunks[j:j+50]
            try:
                col.add(
                    documents=[c["text"] for c in batch],
                    ids=[str(uuid.uuid4()) for _ in batch],
                    metadatas=[{
                        "source":   filename,
                        "subject":  subject,
                        "chapter":  chapter,
                        "page":     c["page"],
                        "class":    "Class 8",
                    } for c in batch],
                )
                total_chunks += len(batch)
            except Exception:
                pass
    return total_chunks

# ════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def get_collection():
    ef     = ONNXMiniLM_L6_V2()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

col = get_collection()

# ════════════════════════════════════════════════════════════════
# FIRST-TIME INDEXING
# ════════════════════════════════════════════════════════════════

if col.count() == 0:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(79,70,229,0.08));
         border:1px solid rgba(99,102,241,0.25);border-radius:20px;padding:28px;margin-bottom:20px;">
      <h2 style="margin:0 0 8px 0;">🔄 Setting Up NCERT Database</h2>
      <p style="color:#94a3b8;margin:0;">Downloading and indexing your NCERT books. Runs <strong>only once</strong> — ~3–5 minutes.</p>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Starting...")
    status_text  = st.empty()
    log_area     = st.container()
    all_pdfs     = []
    os.makedirs(PDF_DIR, exist_ok=True)

    for idx, (zip_name, (file_id, subject)) in enumerate(GDRIVE_FILES.items()):
        progress_bar.progress(int((idx/5)*45), text=f"⬇️ Downloading {zip_name}...")
        status_text.text(f"⬇️ Downloading {zip_name} ({idx+1}/5) — {subject}...")
        zip_path = os.path.join(PDF_DIR, zip_name)
        if download_from_gdrive(file_id, zip_path):
            pdfs = extract_zip(zip_path, PDF_DIR, subject)
            all_pdfs.extend(pdfs)
            log_area.success(f"✅ {zip_name} → {len(pdfs)} PDFs ({subject})")
            try: os.remove(zip_path)
            except: pass
        else:
            log_area.warning(f"⚠️ Could not download {zip_name}")

    if not all_pdfs:
        st.error("❌ No PDFs downloaded. Check Google Drive sharing settings.")
        st.stop()

    total = index_pdfs(col, all_pdfs, progress_bar, status_text)
    progress_bar.progress(100, text=f"✅ Done! {total:,} chunks indexed.")
    status_text.empty()

    if total > 0:
        st.success(f"🎉 {total:,} chunks indexed from {len(all_pdfs)} PDFs.")
        st.rerun()
    else:
        st.error("❌ Indexing failed.")
        st.stop()

# ════════════════════════════════════════════════════════════════
# GROQ CLIENT
# ════════════════════════════════════════════════════════════════

api_key = os.getenv("GROQ_API_KEY", "")
if not api_key:
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY"):
                    api_key = line.split("=", 1)[1].strip().strip('"\'')
    except: pass

groq_client = Groq(api_key=api_key) if api_key else None

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:8px;padding:8px 0 16px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
  <span style="font-size:1.5em;">🧠</span>
  <div>
    <div style="font-family:'Sora',sans-serif;font-weight:800;font-size:1.1em;">BrainForge</div>
    <div style="color:#64748b;font-size:0.72em;">NCERT Class 8 Chat</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📚 Subject")
selected_subject = st.sidebar.radio(
    "Select Subject", SUBJECTS,
    format_func=lambda x: f"{SUBJECT_ICONS.get(x,'📖')} {x}",
    label_visibility="collapsed",
    key="subject_filter",
)

st.sidebar.markdown("---")
total_chunks = col.count()
st.sidebar.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 14px;">
  <p style="margin:0 0 4px 0;font-size:0.78em;color:#94a3b8;">NCERT Class 8</p>
  <p style="margin:0 0 8px 0;font-size:0.85em;font-weight:700;color:#a5b4fc;">📦 {total_chunks:,} chunks ready</p>
  <p style="margin:0 0 2px 0;font-size:0.75em;color:#64748b;">📐 Maths · 🔬 Science</p>
  <p style="margin:0;font-size:0.75em;color:#64748b;">🏛️ History · 🌍 Geography · ⚖️ Civics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
show_sources = st.sidebar.toggle("📄 Show NCERT Sources", value=False)
answer_depth = st.sidebar.selectbox(
    "Answer Style",
    ["Simple (Class 8 level)", "Detailed", "Bullet Points", "With Examples"],
)

msgs = st.session_state.get("messages", [])
if msgs:
    turns = len([m for m in msgs if m["role"] == "user"])
    st.sidebar.markdown(f"""
    <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:10px;padding:10px 14px;margin-top:8px;">
      <p style="margin:0;font-size:0.75em;color:#86efac;font-weight:700;">🧠 Memory — {turns} question{"s" if turns!=1 else ""} in context</p>
    </div>
    """, unsafe_allow_html=True)

if st.sidebar.button("🗑️ Clear Chat & Memory", use_container_width=True):
    st.session_state.messages           = []
    st.session_state.selected_chapter   = None
    st.session_state.chapter_mode       = False
    st.rerun()

# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════

if "messages"         not in st.session_state: st.session_state.messages         = []
if "selected_chapter" not in st.session_state: st.session_state.selected_chapter = None
if "chapter_mode"     not in st.session_state: st.session_state.chapter_mode     = False
if "app_mode"         not in st.session_state: st.session_state.app_mode         = "chat"  # "chat" or "index"

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

icon = SUBJECT_ICONS.get(selected_subject, "📚")
st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(79,70,229,0.08));
     border:1px solid rgba(99,102,241,0.25);border-radius:20px;padding:24px 28px;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
    <span style="font-size:2em;">{icon}</span>
    <div>
      <h1 style="margin:0;font-size:1.6em;font-weight:800;">NCERT Chat</h1>
      <p style="margin:0;color:#94a3b8;font-size:0.85em;">
        AI tutor grounded in Class 8 NCERT · Chapter Index · Conversation Memory
      </p>
    </div>
  </div>
  <div style="margin-top:10px;">
    <span class="stat-pill">{icon} {selected_subject}</span>
    <span class="stat-pill">📦 {total_chunks:,} chunks</span>
    <span class="stat-pill">🧠 AI + NCERT</span>
    <span class="stat-pill">💬 Memory</span>
  </div>
</div>
""", unsafe_allow_html=True)

if groq_client is None:
    st.error("❌ GROQ_API_KEY not set. Go to Streamlit Cloud → Secrets.")
    st.code('GROQ_API_KEY = "your_key_here"')
    st.stop()

# ════════════════════════════════════════════════════════════════
# MODE SWITCHER — Chat vs Chapter Index
# ════════════════════════════════════════════════════════════════

col_c, col_i, col_space = st.columns([1, 1, 4])
with col_c:
    if st.button("💬 Chat Mode", use_container_width=True,
                 type="primary" if st.session_state.app_mode == "chat" else "secondary"):
        st.session_state.app_mode       = "chat"
        st.session_state.selected_chapter = None
        st.rerun()
with col_i:
    if st.button("📖 Chapter Index", use_container_width=True,
                 type="primary" if st.session_state.app_mode == "index" else "secondary"):
        st.session_state.app_mode = "index"
        st.rerun()

st.markdown("")

# ════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ════════════════════════════════════════════════════════════════

def retrieve_chunks(query: str, subject: str, chapter_filter: str = None, top_k: int = 6) -> list:
    try:
        kwargs = dict(
            query_texts=[query],
            n_results=min(top_k, max(col.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        # Build where clause
        if chapter_filter and subject != "All Subjects":
            kwargs["where"] = {"$and": [
                {"subject": {"$eq": subject}},
                {"chapter": {"$eq": chapter_filter}},
            ]}
        elif chapter_filter:
            kwargs["where"] = {"chapter": {"$eq": chapter_filter}}
        elif subject != "All Subjects":
            kwargs["where"] = {"subject": {"$eq": subject}}

        results   = col.query(**kwargs)
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]

        chunks = []
        for doc, meta, dist in zip(docs, metas, distances):
            relevance = round((1 - float(dist)) * 100, 1)
            chunks.append({
                "text":      doc,
                "subject":   meta.get("subject",  "Unknown"),
                "chapter":   meta.get("chapter",  "Unknown"),
                "page":      meta.get("page",     "?"),
                "source":    meta.get("source",   "Unknown"),
                "relevance": relevance,
            })

        return sorted([c for c in chunks if c["relevance"] > 15],
                      key=lambda x: x["relevance"], reverse=True)
    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def generate_refined_answer(
    question: str,
    chunks: list,
    subject: str,
    style: str,
    chat_history: list,
    chapter_context: str = None,
) -> str:
    """
    KEY UPGRADE:
    - Takes raw NCERT chunks
    - AI synthesizes them into a clean, well-written explanation
    - Not a copy-paste — a proper teaching response
    - Uses conversation history for follow-ups
    """

    # Build raw NCERT content block
    raw_ncert = ""
    if chunks:
        raw_ncert = "\n--- RAW NCERT CONTENT (for reference only) ---\n"
        for i, c in enumerate(chunks[:6], 1):
            raw_ncert += f"\n[{c['subject']} | {c['chapter']} | Page {c['page']}]\n{c['text']}\n"
        raw_ncert += "\n--- END OF RAW NCERT CONTENT ---\n"

    chapter_ctx = f"\nStudent is studying: {chapter_context}" if chapter_context else ""

    style_map = {
        "Bullet":   "Use clear numbered headings and bullet points. Break into sections.",
        "Detailed": "Write a comprehensive explanation with all sub-topics covered in depth.",
        "Examples": "Include 2-3 relatable real-life examples after each concept.",
    }
    style_instr = next((v for k, v in style_map.items() if k in style),
                       "Write in simple, clear language. Use short paragraphs. Avoid jargon.")

    system_prompt = f"""You are BrainForge — an expert AI teacher for Class 8–10 students in India.

You receive RAW NCERT textbook content and your job is to TRANSFORM it into a high-quality, student-friendly explanation.

YOUR PROCESS:
1. READ the raw NCERT content carefully
2. UNDERSTAND the key concepts, definitions, and examples
3. REWRITE it as a proper, well-structured teaching response
4. ENRICH with your own knowledge — add analogies, memory tricks, real-world connections
5. NEVER copy-paste from NCERT — always rewrite in your own words

YOUR TEACHING STYLE:
- Write like a friendly, smart teacher — not like a textbook
- Use **bold** for key terms when first introduced
- Break complex ideas into simple steps
- {style_instr}
- Add a "💡 Quick Tip" or "🔑 Key Point" where helpful
- Use relatable Indian examples (cricket, food, daily life) where possible

STUDENT LEVEL: Class 8–10 (age 13–16)
- Simple vocabulary but not childish
- Build from basics to concept
- Make it interesting, not boring

CONVERSATION: You remember the full chat history — handle follow-ups naturally.
{chapter_ctx}

FORMAT YOUR ANSWER AS:
## [Topic Name]
[Your refined explanation]

**Key Points:**
- Point 1
- Point 2

💡 Quick Tip: [memory trick or important note]

📚 NCERT Chapter: [Subject] — [Chapter name]
💬 Want me to explain further, give examples, or quiz you?

Subject: {subject}"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for msg in chat_history[-(MAX_HISTORY * 2):]:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Current question with raw NCERT data
    user_msg = f"""Student's question: {question}
{raw_ncert}
Please synthesize the above NCERT content into a clear, well-written explanation. 
Do NOT copy-paste — rewrite it properly for a Class 8-10 student."""

    messages.append({"role": "user", "content": user_msg})

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=1400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Answer generation failed: {e}"


def generate_chapter_summary(chapter_key: str, chapter_title: str, subject: str) -> str:
    """Generate a chapter overview when student clicks on a chapter."""
    chunks = retrieve_chunks(
        query=chapter_title,
        subject=subject,
        chapter_filter=chapter_key,
        top_k=8,
    )

    raw_ncert = ""
    if chunks:
        for c in chunks[:6]:
            raw_ncert += f"\n{c['text']}\n"

    prompt = f"""You are BrainForge — an expert Class 8 teacher.

A student has opened: {chapter_title} ({subject})

Raw NCERT content from this chapter:
{raw_ncert}

Create a CHAPTER OVERVIEW that includes:

## 📖 {chapter_title}

### What You Will Learn
[3-5 bullet points of key topics covered]

### Key Concepts
[List the 4-6 most important concepts with 1-line explanation each]

### Why This Chapter Matters
[1-2 lines connecting to real life or importance]

### Quick Preview
[2-3 sentences giving a taste of the chapter content]

Keep it engaging, simple, and exciting for a Class 8 student.
End with: "💬 Ask me anything about this chapter!"
"""
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate chapter summary: {e}"


def is_followup(question: str, history: list) -> bool:
    if not history:
        return False
    followup_patterns = [
        r"^(what about|tell me more|explain more|can you|how about|give me|more about)",
        r"^(example|examples|illustrate|show me)",
        r"^(i (don't|do not) understand|unclear|confused|simpler)",
        r"^(test me|quiz me|ask me|mcq)",
    ]
    q = question.lower().strip()
    if len(q.split()) <= 5:
        for p in followup_patterns:
            if re.match(p, q):
                return True
    return False


# ════════════════════════════════════════════════════════════════
# MODE 1 — CHAPTER INDEX
# ════════════════════════════════════════════════════════════════

if st.session_state.app_mode == "index":

    if selected_subject == "All Subjects":
        st.info("👆 Select a specific subject from the sidebar to see its chapter index.")
    else:
        chapters = CHAPTER_INDEX.get(selected_subject, {})
        icon     = SUBJECT_ICONS.get(selected_subject, "📖")

        # If a chapter is selected — show its overview
        if st.session_state.selected_chapter:
            ch_key   = st.session_state.selected_chapter
            ch_title = chapters.get(ch_key, ch_key)

            if st.button("← Back to Chapter List", key="back_btn"):
                st.session_state.selected_chapter = None
                st.rerun()

            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                 border-radius:14px;padding:14px 20px;margin-bottom:16px;">
              <p style="margin:0;font-size:0.75em;color:#a5b4fc;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">
                {icon} {selected_subject}
              </p>
              <p style="margin:4px 0 0 0;font-weight:700;font-size:1.05em;color:#f1f5f9;">{ch_title}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner(f"📖 Loading chapter overview..."):
                summary = generate_chapter_summary(ch_key, ch_title, selected_subject)

            st.markdown(summary)
            st.divider()

            # Set chapter context and switch to chat
            st.markdown("### 💬 Ask anything about this chapter")
            st.caption(f"Questions will be answered from **{ch_title}** content only")

            if st.button(f"💬 Open Chat for this Chapter", use_container_width=True, type="primary"):
                st.session_state.app_mode       = "chat"
                st.session_state.chapter_mode   = True
                st.session_state.messages       = [{
                    "role":    "assistant",
                    "content": f"📖 I'm ready to help you with **{ch_title}** ({selected_subject})!\n\nAsk me anything about this chapter — concepts, examples, definitions, or practice questions.",
                    "chunks":  [],
                }]
                st.rerun()

        else:
            # Show chapter list
            st.markdown(f"### {icon} {selected_subject} — All Chapters")
            st.caption("Click any chapter to see an overview and start studying it")
            st.markdown("")

            for ch_key, ch_title in chapters.items():
                # Parse chapter number and name
                parts    = ch_title.split(" — ", 1)
                ch_num   = parts[0] if len(parts) > 1 else ""
                ch_name  = parts[1] if len(parts) > 1 else ch_title

                col_info, col_btn = st.columns([5, 1])
                with col_info:
                    st.markdown(f"""
                    <div class="chapter-card">
                      <div class="chapter-num">{ch_num}</div>
                      <div class="chapter-title">{ch_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_btn:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("Open →", key=f"ch_{ch_key}", use_container_width=True):
                        st.session_state.selected_chapter = ch_key
                        st.rerun()

# ════════════════════════════════════════════════════════════════
# MODE 2 — CHAT
# ════════════════════════════════════════════════════════════════

else:
    # Chapter context banner (if coming from chapter index)
    chapter_context = None
    chapter_filter  = None

    if st.session_state.get("chapter_mode") and st.session_state.get("selected_chapter"):
        ch_key     = st.session_state.selected_chapter
        subject_ch = CHAPTER_INDEX.get(selected_subject, {})
        ch_title   = subject_ch.get(ch_key, ch_key)
        chapter_context = ch_title
        chapter_filter  = ch_key

        col_banner, col_exit = st.columns([5, 1])
        with col_banner:
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                 border-radius:12px;padding:10px 16px;margin-bottom:12px;">
              <p style="margin:0;font-size:0.8em;color:#a5b4fc;font-weight:700;">
                📖 Chapter Mode: {ch_title}
              </p>
              <p style="margin:2px 0 0 0;font-size:0.72em;color:#64748b;">
                Questions are answered from this chapter only
              </p>
            </div>
            """, unsafe_allow_html=True)
        with col_exit:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✕ Exit", key="exit_chapter"):
                st.session_state.chapter_mode     = False
                st.session_state.selected_chapter = None
                st.session_state.messages         = []
                st.rerun()

    # Suggested questions
    SUGGESTIONS = {
        "All Subjects":  ["What is photosynthesis?", "What are rational numbers?", "What caused the 1857 revolt?", "What is friction?", "What are natural resources?", "What is Parliament?"],
        "Mathematics":   ["What are rational numbers?", "Explain linear equations", "What is Pythagorean theorem?", "Area of a trapezium?", "What are algebraic expressions?", "Explain factorisation"],
        "Science":       ["What is photosynthesis?", "Explain cell structure", "What is friction?", "How does sound travel?", "What are microorganisms?", "Explain force and pressure"],
        "History":       ["What caused the 1857 revolt?", "Who was Tipu Sultan?", "Impact of British rule on trade?", "Role of press in nationalism?", "What was the tribal uprising?"],
        "Geography":     ["What are natural resources?", "Land use patterns in India?", "What is agriculture?", "What are industries?", "What is human resources?"],
        "Civics":        ["What is the Indian Constitution?", "Role of Parliament?", "What are Fundamental Rights?", "What is the judiciary?", "Explain secularism"],
    }

    def render_sources(chunks):
        if not chunks or not show_sources:
            return
        with st.expander(f"📄 {len(chunks)} NCERT sources used"):
            for c in chunks:
                rel   = c["relevance"]
                color = "#22c55e" if rel >= 70 else "#f59e0b" if rel >= 45 else "#94a3b8"
                st.markdown(f"""
                <div class="source-card" style="border-left:3px solid {color};">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-weight:700;color:#a5b4fc;font-size:0.82em;">
                      {SUBJECT_ICONS.get(c['subject'],'📖')} {c['subject']} — {c['chapter'].upper()}
                    </span>
                    <span style="font-size:0.72em;color:{color};font-weight:700;">{rel}% match</span>
                  </div>
                  <div style="font-size:0.72em;color:#64748b;margin-bottom:6px;">📄 Page {c['page']} · {c['source']}</div>
                  <div style="color:#cbd5e1;font-size:0.82em;line-height:1.6;">{c['text'][:280]}{'...' if len(c['text'])>280 else ''}</div>
                </div>""", unsafe_allow_html=True)

    def process_question(question: str):
        history = st.session_state.messages
        chunks  = []

        if not is_followup(question, history):
            with st.spinner("🔍 Searching NCERT books..."):
                chunks = retrieve_chunks(
                    query=question,
                    subject=selected_subject,
                    chapter_filter=chapter_filter,
                    top_k=6,
                )

        with st.spinner("✍️ Crafting your answer..."):
            answer = generate_refined_answer(
                question=question,
                chunks=chunks,
                subject=selected_subject,
                style=answer_depth,
                chat_history=history,
                chapter_context=chapter_context,
            )

        return answer, chunks

    # Show suggestions on empty chat
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        suggestions = SUGGESTIONS.get(selected_subject, SUGGESTIONS["All Subjects"])
        cols = st.columns(3)
        for i, sugg in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(sugg, key=f"s{i}", use_container_width=True):
                    answer, chunks = process_question(sugg)
                    st.session_state.messages += [
                        {"role": "user",      "content": sugg,   "chunks": []},
                        {"role": "assistant", "content": answer, "chunks": chunks},
                    ]
                    st.rerun()
        st.markdown("")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("chunks"):
                render_sources(msg["chunks"])

    # Chat input
    question = st.chat_input(
        f"Ask about {chapter_context}..." if chapter_context
        else f"Ask anything from Class 8 {selected_subject} — I remember our chat!"
        if selected_subject != "All Subjects"
        else "Ask anything from Class 8 NCERT — Maths, Science, History, Geography, Civics"
    )

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answer, chunks = process_question(question)
            st.markdown(answer)
            if chunks:
                render_sources(chunks)
        st.session_state.messages += [
            {"role": "user",      "content": question, "chunks": []},
            {"role": "assistant", "content": answer,   "chunks": chunks},
        ]
