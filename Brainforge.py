# ================================================================
# BrainForge — NCERT Chat (Complete Standalone App)
# ================================================================
# SETUP:
#   1. pip install streamlit chromadb sentence-transformers groq
#   2. Place chroma_study_db/ folder next to this file
#   3. Set your GROQ_API_KEY in .env or environment variable
#   4. streamlit run ncert_app.py
# ================================================================

import os
import re
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

CHROMA_DIR       = "."
COLLECTION_NAME  = "ncert_class8"
GROQ_MODEL       = "llama-3.1-8b-instant"
GROQ_MODEL_LARGE = "llama-3.3-70b-versatile"

# Subjects detected from YOUR database
SUBJECTS = [
    "All Subjects",
    "Mathematics",
    "Science",
    "History",
    "Geography",
    "Civics",
]

# Subject icons
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
    color: #ffffff;
    font-family: 'Inter', sans-serif;
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
    border-radius: 14px !important;
    padding: 14px !important;
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 8px;
}
div[data-testid="stChatMessage"] * { color: #f1f5f9 !important; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    color: white !important;
    border-radius: 12px;
    font-weight: 700;
    border: none;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4);
}

div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
label[data-testid="stWidgetLabel"] {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.88em !important;
}

code {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
    padding: 3px 8px !important;
    border-radius: 6px !important;
}

.source-card {
    background: rgba(255,255,255,0.03);
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78em;
    font-weight: 700;
    color: #a5b4fc;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# LOAD RESOURCES (cached — loads only once)
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="📚 Loading NCERT database...")
def load_collection():
    try:
        client     = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        return None

@st.cache_resource(show_spinner="🧠 Loading AI model...")
def load_embedder():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        return None

collection = load_collection()
embedder   = load_embedder()

# ── Groq client ───────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY", "")
if not api_key:
    # Try reading from .env manually
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

# DB Stats in sidebar
if collection:
    total = collection.count()
    st.sidebar.markdown("### 📊 Database Info")
    st.sidebar.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 14px;">
      <p style="margin:0 0 6px 0;font-size:0.78em;color:#94a3b8;">Your indexed NCERT books</p>
      <p style="margin:0 0 4px 0;font-size:0.85em;font-weight:700;color:#a5b4fc;">📦 {total:,} chunks stored</p>
      <p style="margin:0 0 4px 0;font-size:0.78em;color:#64748b;">📐 Mathematics</p>
      <p style="margin:0 0 4px 0;font-size:0.78em;color:#64748b;">🔬 Science</p>
      <p style="margin:0 0 4px 0;font-size:0.78em;color:#64748b;">🏛️ History</p>
      <p style="margin:0 0 4px 0;font-size:0.78em;color:#64748b;">🌍 Geography</p>
      <p style="margin:0;font-size:0.78em;color:#64748b;">⚖️ Civics</p>
    </div>
    """, unsafe_allow_html=True)

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
# MAIN AREA
# ════════════════════════════════════════════════════════════════

# Header
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
    <span class="stat-pill">📚 {selected_subject}</span>
    <span class="stat-pill">🏠 Runs Locally</span>
    <span class="stat-pill">📄 Page Citations</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Error states ──────────────────────────────────────────────
if collection is None:
    st.error("❌ NCERT database not found. Make sure `chroma_study_db/` is in the same folder as this file.")
    st.stop()

if embedder is None:
    st.error("❌ Could not load embedding model. Run: `pip install sentence-transformers`")
    st.stop()

if groq_client is None:
    st.error("❌ GROQ_API_KEY not found. Set it in your `.env` file or environment variables.")
    st.info("Create a `.env` file with: `GROQ_API_KEY=your_key_here`")
    st.stop()

# ════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ════════════════════════════════════════════════════════════════

def retrieve_chunks(query: str, subject: str, top_k: int = 5) -> list:
    """Search ChromaDB for relevant NCERT content."""
    try:
        q_emb = embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        where = {"subject": subject} if subject != "All Subjects" else None

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            relevance = round((1 - dist) * 100, 1)
            if relevance > 15:
                chunks.append({
                    "text":      doc,
                    "subject":   meta.get("subject",      "Unknown"),
                    "chapter":   meta.get("chapter",      "Unknown"),
                    "page":      meta.get("page",         "?"),
                    "source":    meta.get("source",       "Unknown"),
                    "folder":    meta.get("folder",       ""),
                    "type":      meta.get("content_type", "theory"),
                    "relevance": relevance,
                })

        return sorted(chunks, key=lambda x: x["relevance"], reverse=True)

    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def build_answer_instruction(style: str) -> str:
    if "Bullet" in style:
        return "Format your answer as clear bullet points."
    elif "Detailed" in style:
        return "Give a detailed, thorough explanation with all relevant concepts."
    elif "Examples" in style:
        return "Explain with real-life examples and analogies the student can relate to."
    else:
        return "Use simple, clear language suitable for a Class 8 student."


def generate_answer(question: str, chunks: list, subject: str, style: str) -> str:
    """Generate answer using Groq LLM grounded in NCERT content."""
    if not chunks:
        return (
            "I couldn't find relevant content in the NCERT books for this question. "
            "Try rephrasing your question or selecting a specific subject from the sidebar."
        )

    # Build context
    context = ""
    for i, c in enumerate(chunks[:4], 1):
        context += f"\n[Source {i}: {c['subject']} | {c['chapter']} | Page {c['page']}]\n{c['text']}\n"

    style_instruction = build_answer_instruction(style)
    subject_line      = f"Subject: {subject}" if subject != "All Subjects" else "Searching across all Class 8 subjects"

    prompt = f"""You are a friendly NCERT tutor for Class 8 students in India.
Answer the student's question using ONLY the NCERT content provided below.

{subject_line}
Student's question: {question}

NCERT Content:
{context}

Instructions:
- {style_instruction}
- Use ONLY the content provided above — do not add outside knowledge
- If the content is about Maths, show steps clearly
- If the content is about Science, explain the concept simply
- Mention the chapter or subject naturally in your answer
- If the retrieved content does not fully answer the question, say so honestly
- End with: "📚 From: [Subject] — [Chapter name]"
"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate answer: {e}"


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
# CHAT SESSION
# ════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show suggestions only when chat is empty
if not st.session_state.messages:
    st.markdown(f"### 💡 Ask something from {selected_subject}")
    suggestions = SUGGESTIONS.get(selected_subject, SUGGESTIONS["All Subjects"])

    cols = st.columns(3)
    for i, sugg in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                # Process suggestion as a question
                with st.spinner("🔍 Searching NCERT books..."):
                    chunks = retrieve_chunks(sugg, selected_subject)

                with st.spinner("🤖 Generating answer..."):
                    answer = generate_answer(sugg, chunks, selected_subject, answer_depth)

                st.session_state.messages.append({
                    "role": "user", "content": sugg, "chunks": []
                })
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "chunks": chunks
                })
                st.rerun()

    st.markdown("")

# ── Render chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Sources expander for assistant messages
        if msg["role"] == "assistant" and show_sources and msg.get("chunks"):
            with st.expander(f"📄 {len(msg['chunks'])} sources from your NCERT PDFs"):
                for chunk in msg["chunks"]:
                    rel   = chunk["relevance"]
                    color = "#22c55e" if rel >= 70 else "#f59e0b" if rel >= 45 else "#94a3b8"
                    subj_icon = SUBJECT_ICONS.get(chunk['subject'], "📖")

                    st.markdown(f"""
                    <div class="source-card" style="border-left: 3px solid {color};">
                      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-weight:700;color:#a5b4fc;font-size:0.82em;">
                          {subj_icon} {chunk['subject']} — {chunk['chapter']}
                        </span>
                        <span style="font-size:0.72em;color:{color};font-weight:700;">
                          {rel}% match
                        </span>
                      </div>
                      <div style="font-size:0.72em;color:#64748b;margin-bottom:6px;">
                        📄 Page {chunk['page']}  ·  {chunk['source']}  ·  {chunk['type']}
                      </div>
                      <div style="color:#cbd5e1;font-size:0.82em;line-height:1.6;">
                        {chunk['text'][:350]}{'...' if len(chunk['text']) > 350 else ''}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────
placeholder = (
    f"Ask anything from Class 8 {selected_subject} NCERT..."
    if selected_subject != "All Subjects"
    else "Ask anything from Class 8 NCERT (Maths, Science, History, Geography, Civics)..."
)

question = st.chat_input(placeholder)

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve + Answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching your NCERT books..."):
            chunks = retrieve_chunks(question, selected_subject, top_k=5)

        if not chunks:
            answer = (
                "I couldn't find relevant content for this question. "
                "Try selecting a specific subject from the sidebar, or rephrase your question."
            )
            st.markdown(answer)
        else:
            with st.spinner("🤖 Generating answer from NCERT content..."):
                answer = generate_answer(question, chunks, selected_subject, answer_depth)

            st.markdown(answer)

            if show_sources:
                with st.expander(f"📄 {len(chunks)} sources from your NCERT PDFs"):
                    for chunk in chunks:
                        rel   = chunk["relevance"]
                        color = "#22c55e" if rel >= 70 else "#f59e0b" if rel >= 45 else "#94a3b8"
                        subj_icon = SUBJECT_ICONS.get(chunk['subject'], "📖")

                        st.markdown(f"""
                        <div class="source-card" style="border-left: 3px solid {color};">
                          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-weight:700;color:#a5b4fc;font-size:0.82em;">
                              {subj_icon} {chunk['subject']} — {chunk['chapter']}
                            </span>
                            <span style="font-size:0.72em;color:{color};font-weight:700;">
                              {rel}% match
                            </span>
                          </div>
                          <div style="font-size:0.72em;color:#64748b;margin-bottom:6px;">
                            📄 Page {chunk['page']}  ·  {chunk['source']}  ·  {chunk['type']}
                          </div>
                          <div style="color:#cbd5e1;font-size:0.82em;line-height:1.6;">
                            {chunk['text'][:350]}{'...' if len(chunk['text']) > 350 else ''}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({
        "role": "user", "content": question, "chunks": []
    })
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "chunks": chunks
    })
