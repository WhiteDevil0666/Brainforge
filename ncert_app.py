# ================================================================
# BrainForge — NCERT Chat (Class 8, 9, 10 | Maths & Science)
# Downloads all ZIPs from a single Google Drive folder
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
COLLECTION_NAME = "ncert_multiclass"
PDF_DIR         = "./ncert_pdfs"
GROQ_MODEL      = "llama-3.3-70b-versatile"
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 100
MAX_HISTORY     = 6

# ── Google Drive Folder ID ────────────────────────────────────
# Folder: https://drive.google.com/drive/folders/1kmdsKKLdP_NJuAYwt_Exj-hekF7Fs2Lh
# Must be shared as "Anyone with the link can view"
GDRIVE_FOLDER_ID = "1kmdsKKLdP_NJuAYwt_Exj-hekF7Fs2Lh"

# ── ZIP filename → (subject, class_label) ─────────────────────
# Keys match EXACTLY the filenames in your Drive folder (from screenshot)
ZIP_METADATA = {
    "8th Math.zip":     ("Mathematics", "Class 8"),
    "8th Science.zip":  ("Science",     "Class 8"),
    "9th Math.zip":     ("Mathematics", "Class 9"),
    "9th Science.zip":  ("Science",     "Class 9"),
    "10th Math.zip":    ("Mathematics", "Class 10"),
    "10th Science.zip": ("Science",     "Class 10"),
}

CLASSES  = ["Class 8", "Class 9", "Class 10"]
SUBJECTS = ["Both", "Mathematics", "Science"]

SUBJECT_ICONS = {"Mathematics": "📐", "Science": "🔬", "Both": "📚"}
CLASS_ICONS   = {"Class 8": "8️⃣",  "Class 9": "9️⃣",  "Class 10": "🔟"}
CLASS_COLORS  = {"Class 8": "#6366f1", "Class 9": "#0ea5e9", "Class 10": "#10b981"}

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
.stat-pill { display:inline-flex; align-items:center; gap:5px; background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.25); border-radius:20px; padding:4px 12px; font-size:0.78em; font-weight:700; color:#a5b4fc; margin-right:6px; margin-bottom:4px; }
.chapter-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:14px 18px; margin-bottom:10px; transition:border-color 0.2s; }
.chapter-card:hover { border-color:rgba(99,102,241,0.4); }
.chapter-num   { color:#6366f1; font-weight:800; font-size:0.8em; margin-bottom:3px; }
.chapter-title { color:#f1f5f9; font-weight:600; font-size:0.95em; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# DOWNLOAD + EXTRACTION HELPERS
# ════════════════════════════════════════════════════════════════

def download_folder(folder_id: str, dest_dir: str) -> list:
    """Download all files from a Google Drive folder using gdown."""
    os.makedirs(dest_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        gdown.download_folder(url, output=dest_dir, quiet=False, use_cookies=False)
        return [
            os.path.join(dest_dir, f)
            for f in os.listdir(dest_dir)
            if f.endswith(".zip")
        ]
    except Exception as e:
        st.error(f"Folder download failed: {e}")
        return []


def extract_zip(zip_path: str, extract_to: str, subject: str, class_label: str) -> list:
    """Extract a ZIP and return list of (pdf_path, subject, class_label, filename)."""
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                clean_name = os.path.basename(name)
                if not clean_name.lower().endswith(".pdf") or clean_name.startswith(("__", ".")):
                    continue
                out_dir  = os.path.join(extract_to, class_label.replace(" ", ""), subject)
                out_path = os.path.join(out_dir, clean_name)
                os.makedirs(out_dir, exist_ok=True)
                with z.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append((out_path, subject, class_label, clean_name))
    except Exception as e:
        st.warning(f"Could not extract {os.path.basename(zip_path)}: {e}")
    return extracted


def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\u0900-\u097F]", " ", text)
    text = re.sub(r"\.{3,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    page_pat = re.compile(r"\[Page (\d+)\]")
    max_page = max((int(m.group(1)) for m in page_pat.finditer(text)), default=1)
    clean    = page_pat.sub("", text).strip()
    total    = max(len(clean), 1)
    chunks, start = [], 0
    while start < len(clean):
        chunk = clean[start : start + CHUNK_SIZE].strip()
        if len(chunk) > 80:
            chunks.append({"text": chunk, "page": max(1, int((start / total) * max_page))})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_pdfs(col, pdf_list, progress_bar, status_text) -> int:
    total_chunks = 0
    for i, (pdf_path, subject, class_label, filename) in enumerate(pdf_list):
        chapter = os.path.splitext(filename)[0].lower()
        status_text.text(f"📖 Indexing: {filename} ({i+1}/{len(pdf_list)}) — {class_label} {subject}")
        progress_bar.progress(int((i / max(len(pdf_list), 1)) * 85) + 10)
        text, _ = extract_pdf_text(pdf_path)
        if not text.strip():
            continue
        chunks = chunk_text(text)
        for j in range(0, len(chunks), 50):
            batch = chunks[j : j + 50]
            try:
                col.add(
                    documents=[c["text"] for c in batch],
                    ids=[str(uuid.uuid4()) for _ in batch],
                    metadatas=[{
                        "source":  filename,
                        "subject": subject,
                        "chapter": chapter,
                        "page":    c["page"],
                        "class":   class_label,
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
      <p style="color:#94a3b8;margin:0;">
        Downloading 6 ZIP files from Google Drive and indexing all PDFs for Class 8, 9 & 10.
        Runs <strong>only once</strong> — ~5–10 minutes.
      </p>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Starting...")
    status_text  = st.empty()
    log_area     = st.container()
    all_pdfs     = []

    # ── Step 1: Download entire Drive folder ─────────────────
    zip_download_dir = os.path.join(PDF_DIR, "_zips")
    status_text.text("⬇️ Connecting to Google Drive folder...")
    progress_bar.progress(5, text="⬇️ Downloading from Google Drive...")

    downloaded_zips = download_folder(GDRIVE_FOLDER_ID, zip_download_dir)

    if not downloaded_zips:
        st.error(
            "❌ No ZIP files downloaded.\n\n"
            "Make sure the Drive folder is shared as **'Anyone with the link can view'**."
        )
        st.stop()

    log_area.success(f"✅ Downloaded {len(downloaded_zips)} ZIP file(s) from Google Drive")
    progress_bar.progress(15, text="📦 Extracting ZIP files...")

    # ── Step 2: Extract each ZIP ──────────────────────────────
    for i, zip_path in enumerate(downloaded_zips):
        zip_name = os.path.basename(zip_path)

        # Exact match first, then case-insensitive fallback
        meta = ZIP_METADATA.get(zip_name)
        if not meta:
            for key, val in ZIP_METADATA.items():
                if key.lower() == zip_name.lower():
                    meta = val
                    break

        if not meta:
            log_area.warning(
                f"⚠️ '{zip_name}' not found in ZIP_METADATA — skipping.\n"
                f"Expected one of: {list(ZIP_METADATA.keys())}"
            )
            continue

        subject, class_label = meta
        status_text.text(f"📦 Extracting {zip_name} → {class_label} {subject} ({i+1}/{len(downloaded_zips)})")
        progress_bar.progress(15 + int((i / len(downloaded_zips)) * 20))

        pdfs = extract_zip(zip_path, PDF_DIR, subject, class_label)
        all_pdfs.extend(pdfs)
        log_area.success(f"✅ {zip_name} → {len(pdfs)} PDFs  ({class_label} · {subject})")

        try:
            os.remove(zip_path)
        except Exception:
            pass

    if not all_pdfs:
        st.error("❌ No PDFs extracted. Check ZIP contents and ZIP_METADATA keys.")
        st.stop()

    # ── Step 3: Index all PDFs ────────────────────────────────
    status_text.text(f"🔍 Indexing {len(all_pdfs)} PDFs into vector database...")
    total = index_pdfs(col, all_pdfs, progress_bar, status_text)
    progress_bar.progress(100, text=f"✅ Done! {total:,} chunks indexed.")
    status_text.empty()

    if total > 0:
        st.success(f"🎉 {total:,} chunks indexed from {len(all_pdfs)} PDFs across Classes 8, 9 & 10.")
        st.rerun()
    else:
        st.error("❌ Indexing failed — PDFs may be scanned images or empty.")
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
                    api_key = line.split("=", 1)[1].strip().strip("\"'")
    except Exception:
        pass

groq_client = Groq(api_key=api_key) if api_key else None

# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "messages":         [],
        "selected_chapter": None,
        "chapter_mode":     False,
        "app_mode":         "chat",
        "selected_class":   "Class 8",
        "selected_subject": "Both",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:8px;padding:8px 0 16px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
  <span style="font-size:1.5em;">🧠</span>
  <div>
    <div style="font-family:'Sora',sans-serif;font-weight:800;font-size:1.1em;">BrainForge</div>
    <div style="color:#64748b;font-size:0.72em;">NCERT Class 8 · 9 · 10</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Class Selector ────────────────────────────────────────────
st.sidebar.markdown("### 🎓 Class")
selected_class = st.sidebar.radio(
    "Select Class", CLASSES,
    format_func=lambda x: f"{CLASS_ICONS.get(x, '')} {x}",
    label_visibility="collapsed",
    key="class_radio",
)

if selected_class != st.session_state.selected_class:
    st.session_state.selected_class   = selected_class
    st.session_state.selected_chapter = None
    st.session_state.chapter_mode     = False
    st.session_state.messages         = []

st.sidebar.markdown("---")

# ── Subject Selector ──────────────────────────────────────────
st.sidebar.markdown("### 📚 Subject")
selected_subject = st.sidebar.radio(
    "Select Subject", SUBJECTS,
    format_func=lambda x: f"{SUBJECT_ICONS.get(x, '📖')} {x}",
    label_visibility="collapsed",
    key="subject_radio",
)

if selected_subject != st.session_state.selected_subject:
    st.session_state.selected_subject = selected_subject
    st.session_state.selected_chapter = None
    st.session_state.chapter_mode     = False
    st.session_state.messages         = []

st.sidebar.markdown("---")

# ── Stats ─────────────────────────────────────────────────────
total_chunks = col.count()
st.sidebar.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 14px;">
  <p style="margin:0 0 4px 0;font-size:0.78em;color:#94a3b8;">Database</p>
  <p style="margin:0 0 8px 0;font-size:0.85em;font-weight:700;color:#a5b4fc;">📦 {total_chunks:,} chunks indexed</p>
  <p style="margin:0 0 2px 0;font-size:0.75em;color:#64748b;">8️⃣ Class 8 · 9️⃣ Class 9 · 🔟 Class 10</p>
  <p style="margin:0;font-size:0.75em;color:#64748b;">📐 Mathematics · 🔬 Science</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
show_sources = st.sidebar.toggle("📄 Show NCERT Sources", value=False)
answer_depth = st.sidebar.selectbox(
    "Answer Style",
    ["Simple (student level)", "Detailed", "Bullet Points", "With Examples"],
)

msgs = st.session_state.messages
if msgs:
    turns = len([m for m in msgs if m["role"] == "user"])
    st.sidebar.markdown(f"""
    <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:10px;padding:10px 14px;margin-top:8px;">
      <p style="margin:0;font-size:0.75em;color:#86efac;font-weight:700;">🧠 {turns} question{"s" if turns!=1 else ""} in memory</p>
    </div>
    """, unsafe_allow_html=True)

if st.sidebar.button("🗑️ Clear Chat & Memory", use_container_width=True):
    st.session_state.messages         = []
    st.session_state.selected_chapter = None
    st.session_state.chapter_mode     = False
    st.rerun()

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

cls     = selected_class
subj    = selected_subject
s_icon  = SUBJECT_ICONS.get(subj, "📚")
c_icon  = CLASS_ICONS.get(cls, "🎓")
c_color = CLASS_COLORS.get(cls, "#6366f1")

st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(79,70,229,0.08));
     border:1px solid {c_color}44;border-radius:20px;padding:24px 28px;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
    <span style="font-size:2em;">{c_icon}</span>
    <div>
      <h1 style="margin:0;font-size:1.6em;font-weight:800;">NCERT Chat</h1>
      <p style="margin:0;color:#94a3b8;font-size:0.85em;">
        {cls} · {subj} · AI tutor grounded in NCERT textbooks
      </p>
    </div>
  </div>
  <div style="margin-top:10px;">
    <span class="stat-pill" style="border-color:{c_color}55;color:{c_color};">{c_icon} {cls}</span>
    <span class="stat-pill">{s_icon} {subj}</span>
    <span class="stat-pill">📦 {total_chunks:,} chunks</span>
    <span class="stat-pill">💬 Memory</span>
  </div>
</div>
""", unsafe_allow_html=True)

if groq_client is None:
    st.error("❌ GROQ_API_KEY not set. Go to Streamlit Cloud → Secrets and add it.")
    st.code('GROQ_API_KEY = "gsk_your_key_here"')
    st.stop()

# ════════════════════════════════════════════════════════════════
# MODE SWITCHER
# ════════════════════════════════════════════════════════════════

col_c, col_i, col_space = st.columns([1, 1, 4])
with col_c:
    if st.button("💬 Chat Mode", use_container_width=True,
                 type="primary" if st.session_state.app_mode == "chat" else "secondary"):
        st.session_state.app_mode         = "chat"
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

def build_where_clause(class_label: str, subject: str, chapter_filter: str = None):
    conditions = [{"class": {"$eq": class_label}}]
    if subject != "Both":
        conditions.append({"subject": {"$eq": subject}})
    if chapter_filter:
        conditions.append({"chapter": {"$eq": chapter_filter}})
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_chunks(query: str, class_label: str, subject: str,
                    chapter_filter: str = None, top_k: int = 6) -> list:
    try:
        where  = build_where_clause(class_label, subject, chapter_filter)
        kwargs = dict(
            query_texts=[query],
            n_results=min(top_k, max(col.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
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
                "class":     meta.get("class",    "Unknown"),
                "relevance": relevance,
            })
        return sorted(
            [c for c in chunks if c["relevance"] > 15],
            key=lambda x: x["relevance"], reverse=True,
        )
    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def generate_refined_answer(question, chunks, class_label, subject,
                             style, chat_history, chapter_context=None):
    raw_ncert = ""
    if chunks:
        raw_ncert = "\n--- RAW NCERT CONTENT ---\n"
        for c in chunks[:6]:
            raw_ncert += f"\n[{c['class']} | {c['subject']} | {c['chapter']} | Page {c['page']}]\n{c['text']}\n"
        raw_ncert += "\n--- END ---\n"

    chapter_ctx = f"\nStudent is studying: {chapter_context}" if chapter_context else ""
    style_map   = {
        "Bullet":   "Use clear numbered headings and bullet points.",
        "Detailed": "Write a comprehensive explanation covering all sub-topics in depth.",
        "Examples": "Include 2-3 relatable real-life examples after each concept.",
    }
    style_instr = next(
        (v for k, v in style_map.items() if k in style),
        "Write in simple, clear language. Use short paragraphs. Avoid jargon.",
    )
    age_map = {"Class 8": "13–14", "Class 9": "14–15", "Class 10": "15–16"}

    system_prompt = f"""You are BrainForge — an expert AI teacher for CBSE students in India.

You receive RAW NCERT textbook content and TRANSFORM it into a high-quality student-friendly explanation.

YOUR PROCESS:
1. READ the raw NCERT content carefully
2. REWRITE as a well-structured teaching response — never copy-paste
3. ENRICH with analogies, memory tricks, real-world connections
4. For Class 10: include board exam tips and key formulas

STYLE: {style_instr}
- Use **bold** for key terms
- Add 💡 Quick Tip or 🔑 Key Point where helpful
- Use Indian examples (cricket, food, daily life) where possible

STUDENT: {class_label} CBSE, age {age_map.get(class_label, "13–16")}
{chapter_ctx}

FORMAT:
## [Topic]
[Explanation]

**Key Points:**
- ...

💡 Quick Tip: [memory trick]

📚 NCERT: {class_label} {subject}
💬 Want examples, a quiz, or deeper explanation?"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-(MAX_HISTORY * 2):]:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"Question: {question}\n{raw_ncert}\nRewrite for a {class_label} student.",
    })

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.4, max_tokens=1400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Answer generation failed: {e}"


def generate_chapter_summary(chapter_key, chapter_title, class_label, subject):
    chunks    = retrieve_chunks(chapter_title, class_label, subject, chapter_key, top_k=8)
    raw_ncert = "\n".join(c["text"] for c in chunks[:6])
    prompt    = f"""You are BrainForge — expert {class_label} CBSE teacher.
Student opened: {chapter_title} ({class_label} · {subject})

NCERT content:
{raw_ncert}

Write a CHAPTER OVERVIEW:
## 📖 {chapter_title}
### What You Will Learn
[3-5 bullet points]
### Key Concepts
[4-6 concepts with 1-line explanations]
### Why This Chapter Matters
[1-2 lines on real-life relevance or board exam importance]
### Quick Preview
[2-3 engaging sentences]
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
# SUGGESTIONS
# ════════════════════════════════════════════════════════════════

SUGGESTIONS = {
    "Class 8": {
        "Both":        ["What is photosynthesis?", "What are rational numbers?", "Explain cell structure", "What is friction?", "Metals vs non-metals?", "Explain linear equations"],
        "Mathematics": ["What are rational numbers?", "Explain linear equations", "What is factorisation?", "Area of a trapezium?", "What are algebraic expressions?", "Explain comparing quantities"],
        "Science":     ["What is photosynthesis?", "Explain cell structure", "What is friction?", "How does sound travel?", "What are microorganisms?", "Explain force and pressure"],
    },
    "Class 9": {
        "Both":        ["What are irrational numbers?", "Explain Newton's laws", "What is a polynomial?", "How does gravitation work?", "What are tissues?", "What is coordinate geometry?"],
        "Mathematics": ["What are irrational numbers?", "Explain Heron's formula", "What is coordinate geometry?", "What are polynomials?", "Explain Euclid's geometry", "Lines and angles?"],
        "Science":     ["Explain Newton's three laws", "What is gravitation?", "Atoms vs molecules?", "What are tissues?", "How is sound produced?", "What is work and energy?"],
    },
    "Class 10": {
        "Both":        ["What are real numbers?", "Explain chemical reactions", "What is trigonometry?", "Acids vs bases?", "What is probability?", "Explain light reflection"],
        "Mathematics": ["What are real numbers?", "Explain quadratic equations", "What is trigonometry?", "Explain arithmetic progressions", "Surface area of a cone?", "What is probability?"],
        "Science":     ["Explain chemical reactions", "Acids vs bases?", "What is heredity?", "How does the human eye work?", "What is electricity?", "Explain life processes"],
    },
}

# ════════════════════════════════════════════════════════════════
# MODE 1 — CHAPTER INDEX
# ════════════════════════════════════════════════════════════════

if st.session_state.app_mode == "index":

    subjects_to_show = ["Mathematics", "Science"] if subj == "Both" else [subj]

    if st.session_state.selected_chapter:
        ch_key, ch_subj = st.session_state.selected_chapter
        ch_title        = CHAPTER_INDEX.get(cls, {}).get(ch_subj, {}).get(ch_key, ch_key)

        if st.button("← Back to Chapter List", key="back_btn"):
            st.session_state.selected_chapter = None
            st.rerun()

        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.08);border:1px solid {c_color}44;
             border-radius:14px;padding:14px 20px;margin-bottom:16px;">
          <p style="margin:0;font-size:0.75em;color:#a5b4fc;font-weight:700;text-transform:uppercase;">
            {c_icon} {cls} · {SUBJECT_ICONS.get(ch_subj, '')} {ch_subj}
          </p>
          <p style="margin:4px 0 0 0;font-weight:700;font-size:1.05em;color:#f1f5f9;">{ch_title}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("📖 Loading chapter overview..."):
            summary = generate_chapter_summary(ch_key, ch_title, cls, ch_subj)
        st.markdown(summary)
        st.divider()

        if st.button("💬 Open Chat for this Chapter", use_container_width=True, type="primary"):
            st.session_state.app_mode         = "chat"
            st.session_state.chapter_mode     = True
            st.session_state.selected_subject = ch_subj
            st.session_state.messages         = [{
                "role":    "assistant",
                "content": f"📖 Ready to help with **{ch_title}** ({cls} · {ch_subj})!\n\nAsk me anything — concepts, examples, definitions, or practice questions.",
                "chunks":  [],
            }]
            st.rerun()

    else:
        for s in subjects_to_show:
            chapters = CHAPTER_INDEX.get(cls, {}).get(s, {})
            if not chapters:
                continue
            st.markdown(f"### {SUBJECT_ICONS.get(s, '')} {cls} — {s}")
            st.caption("Click any chapter to see an overview")
            st.markdown("")
            for ch_key, ch_title in chapters.items():
                parts   = ch_title.split(" — ", 1)
                ch_num  = parts[0] if len(parts) > 1 else ""
                ch_name = parts[1] if len(parts) > 1 else ch_title
                c_col, b_col = st.columns([5, 1])
                with c_col:
                    st.markdown(f"""
                    <div class="chapter-card">
                      <div class="chapter-num">{ch_num}</div>
                      <div class="chapter-title">{ch_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with b_col:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("Open →", key=f"ch_{cls}_{s}_{ch_key}", use_container_width=True):
                        st.session_state.selected_chapter = (ch_key, s)
                        st.rerun()
            st.markdown("")

# ════════════════════════════════════════════════════════════════
# MODE 2 — CHAT
# ════════════════════════════════════════════════════════════════

else:
    chapter_context = None
    chapter_filter  = None
    chapter_subject = subj

    if st.session_state.get("chapter_mode") and st.session_state.get("selected_chapter"):
        ch_key, ch_subj = st.session_state.selected_chapter
        ch_title        = CHAPTER_INDEX.get(cls, {}).get(ch_subj, {}).get(ch_key, ch_key)
        chapter_context = ch_title
        chapter_filter  = ch_key
        chapter_subject = ch_subj

        col_banner, col_exit = st.columns([5, 1])
        with col_banner:
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                 border-radius:12px;padding:10px 16px;margin-bottom:12px;">
              <p style="margin:0;font-size:0.8em;color:#a5b4fc;font-weight:700;">
                📖 Chapter Mode: {ch_title} · {cls}
              </p>
              <p style="margin:2px 0 0 0;font-size:0.72em;color:#64748b;">
                Answers pulled from this chapter only
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
                      {CLASS_ICONS.get(c.get('class',''), '')} {c.get('class','')} ·
                      {SUBJECT_ICONS.get(c['subject'], '📖')} {c['subject']} — {c['chapter'].upper()}
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
                    query=question, class_label=cls, subject=chapter_subject,
                    chapter_filter=chapter_filter, top_k=6,
                )
        with st.spinner("✍️ Crafting your answer..."):
            answer = generate_refined_answer(
                question=question, chunks=chunks, class_label=cls,
                subject=chapter_subject, style=answer_depth,
                chat_history=history, chapter_context=chapter_context,
            )
        return answer, chunks

    # Suggestions on empty chat
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        suggestions = SUGGESTIONS.get(cls, {}).get(subj, SUGGESTIONS["Class 8"]["Both"])
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
    if chapter_context:
        placeholder = f"Ask about {chapter_context} ({cls})..."
    elif subj != "Both":
        placeholder = f"Ask anything from {cls} {subj} NCERT..."
    else:
        placeholder = f"Ask anything from {cls} Maths or Science NCERT..."

    question = st.chat_input(placeholder)

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
