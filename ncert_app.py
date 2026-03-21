# ================================================================
# BrainForge — NCERT Chat (Complete Single File)
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
GROQ_MODEL      = "llama-3.1-8b-instant"
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 100

# ── CORRECTED Google Drive ZIP IDs ────────────────────────────
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
.stButton > button { background:linear-gradient(135deg,#6366f1,#4f46e5); color:white !important; border-radius:12px; font-weight:700; border:none; box-shadow:0 4px 15px rgba(99,102,241,0.3); }
div[data-testid="stSelectbox"] label, label[data-testid="stWidgetLabel"] { color:#e2e8f0 !important; font-weight:600 !important; font-size:0.88em !important; }
.source-card { background:rgba(255,255,255,0.03); border-radius:0 10px 10px 0; padding:10px 14px; margin-bottom:8px; }
.stat-pill { display:inline-flex; align-items:center; gap:5px; background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.25); border-radius:20px; padding:4px 12px; font-size:0.78em; font-weight:700; color:#a5b4fc; margin-right:6px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# DOWNLOAD FROM GOOGLE DRIVE
# ════════════════════════════════════════════════════════════════

def download_from_gdrive(file_id: str, dest_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=True, fuzzy=True)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            return True
        url2 = f"https://drive.google.com/file/d/{file_id}/view"
        gdown.download(url2, dest_path, quiet=True, fuzzy=True)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            return True
        return False
    except Exception:
        return False

# ════════════════════════════════════════════════════════════════
# EXTRACT ZIP
# Subject is taken from GDRIVE_FILES mapping — NOT from filename
# This ensures correct subject tagging regardless of PDF names
# ════════════════════════════════════════════════════════════════

def extract_zip(zip_path: str, extract_to: str, subject: str) -> list:
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                clean_name = os.path.basename(name)
                if not clean_name.lower().endswith(".pdf"):
                    continue
                if clean_name.startswith("__") or clean_name.startswith("."):
                    continue
                out_path = os.path.join(extract_to, subject, clean_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with z.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append((out_path, subject, clean_name))
    except Exception:
        pass
    return extracted

# ════════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION
# ════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\u0900-\u097F]', ' ', text)
    text = re.sub(r'\.{3,}', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_pdf_text(pdf_path: str):
    text  = ""
    pages = 0
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
    chunks   = []
    start    = 0
    while start < len(clean):
        end   = start + CHUNK_SIZE
        chunk = clean[start:end].strip()
        if len(chunk) > 80:
            est_page = max(1, int((start / total) * max_page))
            chunks.append({"text": chunk, "page": est_page})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ════════════════════════════════════════════════════════════════
# INDEX INTO CHROMADB
# ════════════════════════════════════════════════════════════════

def index_pdfs(col, pdf_list: list, progress_bar, status_text) -> int:
    total_chunks = 0
    total_pdfs   = len(pdf_list)

    for i, (pdf_path, subject, filename) in enumerate(pdf_list):
        chapter = os.path.splitext(filename)[0].upper()
        status_text.text(f"📖 Indexing: {filename} ({i+1}/{total_pdfs}) — {subject}")
        progress_bar.progress(int((i / max(total_pdfs, 1)) * 85) + 10)

        text, _ = extract_pdf_text(pdf_path)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        BATCH = 50
        for j in range(0, len(chunks), BATCH):
            batch     = chunks[j:j+BATCH]
            texts     = [c["text"] for c in batch]
            ids       = [str(uuid.uuid4()) for _ in batch]
            metadatas = [
                {
                    "source":  filename,
                    "subject": subject,   # from ZIP mapping — always correct
                    "chapter": chapter,
                    "page":    c["page"],
                    "class":   "Class 8",
                }
                for c in batch
            ]
            try:
                col.add(documents=texts, ids=ids, metadatas=metadatas)
                total_chunks += len(batch)
            except Exception:
                pass

    return total_chunks

# ════════════════════════════════════════════════════════════════
# DATABASE SETUP — force delete old wrong index, start fresh
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def get_collection():
    ef     = ONNXMiniLM_L6_V2()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection to force fresh re-index with correct subjects
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return col


col = get_collection()

# ════════════════════════════════════════════════════════════════
# FIRST-TIME INDEXING UI
# ════════════════════════════════════════════════════════════════

if col.count() == 0:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(79,70,229,0.08));
         border:1px solid rgba(99,102,241,0.25);border-radius:20px;padding:28px;margin-bottom:20px;">
      <h2 style="margin:0 0 8px 0;">🔄 Setting Up NCERT Database</h2>
      <p style="color:#94a3b8;margin:0;">
        Downloading and indexing your NCERT books from Google Drive.
        This runs <strong>only once</strong> — takes about 3–5 minutes.
        After this the app loads instantly every time.
      </p>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Starting...")
    status_text  = st.empty()
    log_area     = st.container()
    all_pdfs     = []

    os.makedirs(PDF_DIR, exist_ok=True)

    for idx, (zip_name, (file_id, subject)) in enumerate(GDRIVE_FILES.items()):
        pct = int((idx / 5) * 45)
        progress_bar.progress(pct, text=f"⬇️ Downloading {zip_name}...")
        status_text.text(f"⬇️ Downloading {zip_name} ({idx+1}/5) — {subject}...")

        zip_path = os.path.join(PDF_DIR, zip_name)
        success  = download_from_gdrive(file_id, zip_path)

        if not success:
            log_area.warning(f"⚠️ Could not download {zip_name} — skipping")
            continue

        status_text.text(f"📦 Extracting {zip_name}...")
        pdfs = extract_zip(zip_path, PDF_DIR, subject)
        all_pdfs.extend(pdfs)
        log_area.success(f"✅ {zip_name} → {len(pdfs)} PDFs ({subject})")

        try:
            os.remove(zip_path)
        except Exception:
            pass

    if not all_pdfs:
        st.error("❌ No PDFs could be downloaded.")
        st.markdown("""
        **How to fix:**
        1. Open your Google Drive folder
        2. Right-click each ZIP → **Share** → **Anyone with the link** → **Viewer**
        3. Reboot the app from Streamlit Cloud → Manage App
        """)
        st.stop()

    status_text.text(f"🧠 Indexing {len(all_pdfs)} PDFs into vector database...")
    total = index_pdfs(col, all_pdfs, progress_bar, status_text)
    progress_bar.progress(100, text=f"✅ Done! {total:,} chunks indexed.")
    status_text.empty()

    if total > 0:
        st.success(f"🎉 Setup complete! {total:,} chunks indexed from {len(all_pdfs)} PDFs.")
        st.rerun()
    else:
        st.error("❌ Indexing failed — no chunks created.")
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
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass

groq_client = Groq(api_key=api_key) if api_key else None

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:8px;padding:8px 0 16px 0;
     border-bottom:1px solid rgba(255,255,255,0.06);">
  <span style="font-size:1.5em;">🧠</span>
  <div>
    <div style="font-family:'Sora',sans-serif;font-weight:800;font-size:1.1em;">BrainForge</div>
    <div style="color:#64748b;font-size:0.72em;">NCERT Class 8 Chat</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📚 Filter by Subject")
selected_subject = st.sidebar.radio(
    "", SUBJECTS,
    format_func=lambda x: f"{SUBJECT_ICONS.get(x,'📖')} {x}",
    key="subject_filter",
)

st.sidebar.markdown("---")
total_chunks = col.count()
st.sidebar.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 14px;">
  <p style="margin:0 0 6px 0;font-size:0.78em;color:#94a3b8;">NCERT Class 8 — Ready</p>
  <p style="margin:0 0 8px 0;font-size:0.85em;font-weight:700;color:#a5b4fc;">📦 {total_chunks:,} chunks indexed</p>
  <p style="margin:0 0 3px 0;font-size:0.78em;color:#64748b;">📐 Mathematics</p>
  <p style="margin:0 0 3px 0;font-size:0.78em;color:#64748b;">🔬 Science</p>
  <p style="margin:0 0 3px 0;font-size:0.78em;color:#64748b;">🏛️ History</p>
  <p style="margin:0 0 3px 0;font-size:0.78em;color:#64748b;">🌍 Geography</p>
  <p style="margin:0;font-size:0.78em;color:#64748b;">⚖️ Civics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
show_sources = st.sidebar.toggle("📄 Show Sources", value=True)
answer_depth = st.sidebar.selectbox(
    "Answer Style",
    ["Simple (Class 8 level)", "Detailed", "Bullet Points", "With Examples"],
)

if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

icon = SUBJECT_ICONS.get(selected_subject, "📚")
st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(79,70,229,0.08));
     border:1px solid rgba(99,102,241,0.25);border-radius:20px;
     padding:24px 28px;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
    <span style="font-size:2em;">{icon}</span>
    <div>
      <h1 style="margin:0;font-size:1.6em;font-weight:800;">NCERT Chat</h1>
      <p style="margin:0;color:#94a3b8;font-size:0.85em;">
        Answers from Class 8 NCERT books · Powered by your PDFs
      </p>
    </div>
  </div>
  <div style="margin-top:10px;">
    <span class="stat-pill">{icon} {selected_subject}</span>
    <span class="stat-pill">📦 {total_chunks:,} chunks</span>
    <span class="stat-pill">📄 Page Citations</span>
  </div>
</div>
""", unsafe_allow_html=True)

if groq_client is None:
    st.error("❌ GROQ_API_KEY not set.")
    st.markdown("Go to **Streamlit Cloud → App Settings → Secrets** and add:")
    st.code('GROQ_API_KEY = "your_key_here"')
    st.stop()

# ════════════════════════════════════════════════════════════════
# SEARCH + ANSWER
# ════════════════════════════════════════════════════════════════

def retrieve_chunks(query: str, subject: str, top_k: int = 6) -> list:
    try:
        kwargs = dict(
            query_texts=[query],
            n_results=min(top_k, max(col.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        if subject != "All Subjects":
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

        return sorted(
            [c for c in chunks if c["relevance"] > 15],
            key=lambda x: x["relevance"], reverse=True,
        )

    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def generate_answer(question: str, chunks: list, subject: str, style: str) -> str:
    if not chunks:
        return "I couldn't find relevant content. Try selecting a specific subject or rephrasing."

    context = ""
    for i, c in enumerate(chunks[:6], 1):
        context += f"\n[Source {i}: {c['subject']} | {c['chapter']} | Page {c['page']}]\n{c['text']}\n"

    style_map = {
        "Bullet":   "Format as clear bullet points.",
        "Detailed": "Give a detailed thorough explanation.",
        "Examples": "Use real-life examples a Class 8 student can relate to.",
    }
    style_instr = next((v for k, v in style_map.items() if k in style),
                       "Use simple clear language for a Class 8 student.")

    prompt = f"""You are a friendly NCERT tutor for Class 8 students in India.
Answer using ONLY the NCERT content provided below.

Subject: {subject}
Question: {question}

NCERT Content:
{context}

Rules:
- {style_instr}
- Only use the content above — no outside knowledge
- For Maths: show steps clearly
- For Science: explain simply
- If content does not fully answer, say so honestly
- End with: "📚 From: [Subject] — [Chapter]"
"""
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Answer generation failed: {e}"


# ════════════════════════════════════════════════════════════════
# SUGGESTED QUESTIONS
# ════════════════════════════════════════════════════════════════

SUGGESTIONS = {
    "All Subjects":  ["What is photosynthesis?", "What are rational numbers?", "What caused the 1857 revolt?", "What is friction?", "What are natural resources?", "What is Parliament?"],
    "Mathematics":   ["What are rational numbers?", "Explain linear equations", "What is Pythagorean theorem?", "Area of a trapezium?", "What are algebraic expressions?", "Explain factorisation"],
    "Science":       ["What is photosynthesis?", "Explain cell structure", "What is friction?", "How does sound travel?", "What are microorganisms?", "Explain force and pressure"],
    "History":       ["What caused the 1857 revolt?", "Who was Tipu Sultan?", "Impact of British rule on trade?", "Role of press in Indian nationalism?", "What was the tribal uprising?"],
    "Geography":     ["What are natural resources?", "Land use patterns in India?", "What is agriculture?", "What are industries?", "What is human resources?"],
    "Civics":        ["What is the Indian Constitution?", "Role of Parliament?", "What are Fundamental Rights?", "What is the judiciary?", "Explain secularism"],
}

# ════════════════════════════════════════════════════════════════
# CHAT UI
# ════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(chunks):
    with st.expander(f"📄 {len(chunks)} sources from NCERT PDFs"):
        for c in chunks:
            rel   = c["relevance"]
            color = "#22c55e" if rel >= 70 else "#f59e0b" if rel >= 45 else "#94a3b8"
            st.markdown(f"""
            <div class="source-card" style="border-left:3px solid {color};">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-weight:700;color:#a5b4fc;font-size:0.82em;">
                  {SUBJECT_ICONS.get(c['subject'],'📖')} {c['subject']} — {c['chapter']}
                </span>
                <span style="font-size:0.72em;color:{color};font-weight:700;">{rel}% match</span>
              </div>
              <div style="font-size:0.72em;color:#64748b;margin-bottom:6px;">
                📄 Page {c['page']} · {c['source']}
              </div>
              <div style="color:#cbd5e1;font-size:0.82em;line-height:1.6;">
                {c['text'][:350]}{'...' if len(c['text'])>350 else ''}
              </div>
            </div>""", unsafe_allow_html=True)


def process_question(q):
    with st.spinner("🔍 Searching NCERT books..."):
        chunks = retrieve_chunks(q, selected_subject)
    with st.spinner("🤖 Generating answer..."):
        answer = generate_answer(q, chunks, selected_subject, answer_depth)
    return answer, chunks


# Suggestions
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
        if msg["role"] == "assistant" and show_sources and msg.get("chunks"):
            render_sources(msg["chunks"])

# Chat input
question = st.chat_input(
    f"Ask anything from Class 8 {selected_subject} NCERT..."
    if selected_subject != "All Subjects"
    else "Ask anything from Class 8 NCERT (Maths, Science, History, Geography, Civics)..."
)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer, chunks = process_question(question)
        st.markdown(answer)
        if show_sources and chunks:
            render_sources(chunks)
    st.session_state.messages += [
        {"role": "user",      "content": question, "chunks": []},
        {"role": "assistant", "content": answer,   "chunks": chunks},
    ]
