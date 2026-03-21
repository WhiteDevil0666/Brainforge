# ================================================================
# BrainForge — NCERT Chat (Streamlit Cloud Fixed)
# ================================================================

import os
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

CHROMA_DIR      = "."
COLLECTION_NAME = "ncert_class8"
GROQ_MODEL      = "llama-3.1-8b-instant"

SUBJECTS = ["All Subjects", "Mathematics", "Science", "History", "Geography", "Civics"]

SUBJECT_ICONS = {
    "Mathematics": "📐",
    "Science":     "🔬",
    "History":     "🏛️",
    "Geography":   "🌍",
    "Civics":      "⚖️",
    "All Subjects":"📚",
}

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BrainForge — NCERT Chat",
    page_icon="🧠",
    layout="wide",
)

# ════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #0a1628 100%);
    color: #ffffff; font-family: 'Inter', sans-serif;
}
h1,h2,h3,h4 { font-family: 'Sora', sans-serif !important; color: #ffffff !important; }
section.main > div { background-color: transparent !important; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1a 0%, #0b1525 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }
div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 14px !important; padding: 14px !important;
    border: 1px solid rgba(255,255,255,0.07); margin-bottom: 8px;
}
div[data-testid="stChatMessage"] * { color: #f1f5f9 !important; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    color: white !important; border-radius: 12px; font-weight: 700;
    border: none; box-shadow: 0 4px 15px rgba(99,102,241,0.3);
}
div[data-testid="stSelectbox"] label, label[data-testid="stWidgetLabel"] {
    color: #e2e8f0 !important; font-weight: 600 !important; font-size: 0.88em !important;
}
.source-card {
    background: rgba(255,255,255,0.03);
    border-radius: 0 10px 10px 0; padding: 10px 14px; margin-bottom: 8px;
}
.stat-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px; padding: 4px 12px; font-size: 0.78em;
    font-weight: 700; color: #a5b4fc; margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# LOAD RESOURCES — KEY FIX: no custom Settings, use get_or_create
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def load_resources():
    """Load ChromaDB + embedder in a single cached function to avoid conflicts."""
    errors = []

    # ── ChromaDB ──────────────────────────────────────────────
    collection = None
    try:
        # Use get_or_create to avoid "already exists" conflict
        client     = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        count      = collection.count()
        if count == 0:
            errors.append("Collection exists but is empty.")
    except Exception as e:
        errors.append(f"ChromaDB error: {e}")

    # ── Embedder ──────────────────────────────────────────────
    embedder = None
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        errors.append(f"Embedder error: {e}")

    return collection, embedder, errors

collection, embedder, load_errors = load_resources()

# ── Groq ──────────────────────────────────────────────────────
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
    "",
    SUBJECTS,
    format_func=lambda x: f"{SUBJECT_ICONS.get(x, '📖')} {x}",
    key="subject_filter",
)

st.sidebar.markdown("---")

if collection:
    try:
        total = collection.count()
        st.sidebar.markdown("### 📊 Database Info")
        st.sidebar.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 14px;">
          <p style="margin:0 0 6px 0;font-size:0.78em;color:#94a3b8;">Your indexed NCERT books</p>
          <p style="margin:0 0 8px 0;font-size:0.85em;font-weight:700;color:#a5b4fc;">📦 {total:,} chunks stored</p>
          <p style="margin:0 0 3px 0;font-size:0.78em;color:#64748b;">📐 Mathematics</p>
          <p style="margin:0 0 3px 0;font-size:0.78em;color:#64748b;">🔬 Science</p>
          <p style="margin:0 0 3px 0;font-size:0.78em;color:#64748b;">🏛️ History</p>
          <p style="margin:0 0 3px 0;font-size:0.78em;color:#64748b;">🌍 Geography</p>
          <p style="margin:0;font-size:0.78em;color:#64748b;">⚖️ Civics</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

st.sidebar.markdown("---")
show_sources = st.sidebar.toggle("📄 Show Sources", value=True)
answer_depth = st.sidebar.selectbox(
    "Answer Style",
    ["Simple (Class 8 level)", "Detailed", "Bullet Points", "With Examples"],
    index=0,
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
        Answers from your Class 8 NCERT books · No internet · Your PDFs only
      </p>
    </div>
  </div>
  <div style="margin-top:10px;">
    <span class="stat-pill">{icon} {selected_subject}</span>
    <span class="stat-pill">🏠 Runs Locally</span>
    <span class="stat-pill">📄 Page Citations</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Show any load errors as warnings (not blockers) ───────────
for err in load_errors:
    st.warning(f"⚠️ {err}")

# ── Hard stops ────────────────────────────────────────────────
if collection is None:
    st.error("❌ NCERT database not found. Make sure all ChromaDB files are in the repo root.")
    st.stop()
if embedder is None:
    st.error("❌ Embedding model failed to load. Check requirements.txt includes sentence-transformers.")
    st.stop()
if groq_client is None:
    st.error("❌ GROQ_API_KEY not set. Go to Streamlit Cloud → App Settings → Secrets and add it.")
    st.code("GROQ_API_KEY = \"your_key_here\"")
    st.stop()

# ════════════════════════════════════════════════════════════════
# SEARCH + ANSWER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def retrieve_chunks(query: str, subject: str, top_k: int = 5) -> list:
    """Search ChromaDB for relevant NCERT chunks."""
    try:
        q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]

        kwargs = dict(
            query_embeddings=[q_emb],
            n_results=min(top_k, max(collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )

        # Only add where filter for specific subjects
        if subject != "All Subjects":
            kwargs["where"] = {"subject": {"$eq": subject}}

        results   = collection.query(**kwargs)
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]

        chunks = []
        for doc, meta, dist in zip(docs, metas, distances):
            relevance = round((1 - float(dist)) * 100, 1)
            chunks.append({
                "text":      doc,
                "subject":   meta.get("subject",      "Unknown"),
                "chapter":   meta.get("chapter",      "Unknown"),
                "page":      meta.get("page",         "?"),
                "source":    meta.get("source",       "Unknown"),
                "type":      meta.get("content_type", "theory"),
                "relevance": relevance,
            })

        chunks = [c for c in chunks if c["relevance"] > 15]
        return sorted(chunks, key=lambda x: x["relevance"], reverse=True)

    except Exception as e:
        # ── Keyword fallback ──────────────────────────────────
        try:
            get_kwargs = dict(include=["documents", "metadatas"], limit=300)
            if subject != "All Subjects":
                get_kwargs["where"] = {"subject": {"$eq": subject}}

            all_data    = collection.get(**get_kwargs)
            docs        = all_data.get("documents", [])
            metas       = all_data.get("metadatas",  [])
            query_words = set(query.lower().split())
            scored      = []

            for doc, meta in zip(docs, metas):
                overlap = len(query_words & set(doc.lower().split()))
                if overlap > 0:
                    scored.append({
                        "text":      doc,
                        "subject":   meta.get("subject",      "Unknown"),
                        "chapter":   meta.get("chapter",      "Unknown"),
                        "page":      meta.get("page",         "?"),
                        "source":    meta.get("source",       "Unknown"),
                        "type":      meta.get("content_type", "theory"),
                        "relevance": min(overlap * 12, 90),
                    })

            scored.sort(key=lambda x: x["relevance"], reverse=True)
            return scored[:top_k]

        except Exception as e2:
            st.error(f"Search failed: {e2}")
            return []


def generate_answer(question: str, chunks: list, subject: str, style: str) -> str:
    """Generate NCERT-grounded answer using Groq."""
    if not chunks:
        return "I couldn't find relevant content for this question. Try selecting a specific subject or rephrasing."

    context = ""
    for i, c in enumerate(chunks[:4], 1):
        context += f"\n[Source {i}: {c['subject']} | {c['chapter']} | Page {c['page']}]\n{c['text']}\n"

    style_map = {
        "Bullet":   "Format your answer as clear bullet points.",
        "Detailed": "Give a detailed, thorough explanation.",
        "Examples": "Use real-life examples a Class 8 student can relate to.",
    }
    style_instr = next((v for k, v in style_map.items() if k in style),
                       "Use simple, clear language for a Class 8 student.")

    prompt = f"""You are a friendly NCERT tutor for Class 8 students in India.
Answer using ONLY the NCERT content provided.

Subject: {subject}
Question: {question}

NCERT Content:
{context}

Rules:
- {style_instr}
- Only use the content above — no outside knowledge
- For Maths: show steps. For Science: explain simply.
- If content doesn't fully answer, say so honestly.
- End with: "📚 From: [Subject] — [Chapter]"
"""

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Answer generation failed: {e}"


# ════════════════════════════════════════════════════════════════
# SUGGESTED QUESTIONS
# ════════════════════════════════════════════════════════════════

SUGGESTIONS = {
    "All Subjects": [
        "What is photosynthesis?",
        "What are rational numbers?",
        "What were the causes of the 1857 revolt?",
        "What is friction and its types?",
        "What are natural resources?",
        "What is the Parliament of India?",
    ],
    "Mathematics": [
        "What are rational numbers?",
        "Explain linear equations in one variable",
        "What is the Pythagorean theorem?",
        "How to find the area of a trapezium?",
        "What are algebraic expressions?",
        "Explain factorisation",
    ],
    "Science": [
        "What is photosynthesis?",
        "Explain the structure of a cell",
        "What is friction?",
        "How does sound travel?",
        "What are microorganisms?",
        "Explain force and pressure",
    ],
    "History": [
        "What were the causes of the 1857 revolt?",
        "Who was Tipu Sultan?",
        "What was the impact of British rule on Indian trade?",
        "Explain the role of the press in Indian nationalism",
        "What was the tribal uprising?",
    ],
    "Geography": [
        "What are natural resources?",
        "Explain land use patterns in India",
        "What is agriculture?",
        "What are industries?",
        "What is human resources?",
    ],
    "Civics": [
        "What is the Indian Constitution?",
        "Explain the role of Parliament",
        "What are Fundamental Rights?",
        "What is the judiciary?",
        "Explain the concept of secularism",
    ],
}

# ════════════════════════════════════════════════════════════════
# CHAT UI
# ════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(chunks: list):
    with st.expander(f"📄 {len(chunks)} sources from your NCERT PDFs"):
        for chunk in chunks:
            rel       = chunk["relevance"]
            color     = "#22c55e" if rel >= 70 else "#f59e0b" if rel >= 45 else "#94a3b8"
            subj_icon = SUBJECT_ICONS.get(chunk["subject"], "📖")
            st.markdown(f"""
            <div class="source-card" style="border-left:3px solid {color};">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-weight:700;color:#a5b4fc;font-size:0.82em;">
                  {subj_icon} {chunk['subject']} — {chunk['chapter']}
                </span>
                <span style="font-size:0.72em;color:{color};font-weight:700;">{rel}% match</span>
              </div>
              <div style="font-size:0.72em;color:#64748b;margin-bottom:6px;">
                📄 Page {chunk['page']} · {chunk['source']} · {chunk['type']}
              </div>
              <div style="color:#cbd5e1;font-size:0.82em;line-height:1.6;">
                {chunk['text'][:350]}{'...' if len(chunk['text']) > 350 else ''}
              </div>
            </div>
            """, unsafe_allow_html=True)


def process_question(q: str):
    with st.spinner("🔍 Searching your NCERT books..."):
        chunks = retrieve_chunks(q, selected_subject)
    with st.spinner("🤖 Generating answer..."):
        answer = generate_answer(q, chunks, selected_subject, answer_depth)
    return answer, chunks


# Suggestions when chat is empty
if not st.session_state.messages:
    st.markdown(f"### 💡 Try asking:")
    suggestions = SUGGESTIONS.get(selected_subject, SUGGESTIONS["All Subjects"])
    cols        = st.columns(3)
    for i, sugg in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                answer, chunks = process_question(sugg)
                st.session_state.messages.append({"role": "user",      "content": sugg,   "chunks": []})
                st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})
                st.rerun()
    st.markdown("")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("chunks"):
            render_sources(msg["chunks"])

# Chat input
placeholder = (
    f"Ask anything from Class 8 {selected_subject} NCERT..."
    if selected_subject != "All Subjects"
    else "Ask anything from Class 8 NCERT (Maths, Science, History, Geography, Civics)..."
)

question = st.chat_input(placeholder)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer, chunks = process_question(question)
        st.markdown(answer)
        if show_sources and chunks:
            render_sources(chunks)
    st.session_state.messages.append({"role": "user",      "content": question, "chunks": []})
    st.session_state.messages.append({"role": "assistant", "content": answer,   "chunks": chunks})
