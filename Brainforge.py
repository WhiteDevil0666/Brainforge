# ================================================================
# NCERT CHAT — BrainForge Integration
# ----------------------------------------------------------------
# HOW TO ADD THIS TO YOUR APP:
#
# 1. Copy this ENTIRE file's content
# 2. Open your app.py
# 3. Find this line in the sidebar section:
#       page = st.sidebar.radio("", [
#           "🏠 My Learning Journey",
#           "📚 AI Study Tutor",
#           "📝 Practice Test",
#           "💬 Ask Your Tutor",
#           "🎯 Study Copilot",
#       ])
#
# 4. Add "📖 NCERT Chat" to that list:
#       page = st.sidebar.radio("", [
#           "🏠 My Learning Journey",
#           "📚 AI Study Tutor",
#           "📝 Practice Test",
#           "💬 Ask Your Tutor",
#           "🎯 Study Copilot",
#           "📖 NCERT Chat",        ← ADD THIS LINE
#       ])
#
# 5. Paste the code below at the END of your app.py
#    (after the last elif block, before the final line)
# ================================================================


# ══════════════════════════════════════════════════════════════════════
# ████████████  📖 NCERT CHAT  ████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════

elif page == "📖 NCERT Chat":

    st.session_state.current_feature = "NCERT_Chat"
    exam_cfg = EXAM_CONFIG[selected_exam]

    # ── Header ────────────────────────────────────────────────
    st.markdown(f"""
    <h1 style="margin-bottom:4px;">📖 NCERT Chat</h1>
    <p style="color:#64748b;margin:0 0 16px 0;">
        Ask anything directly from your Class 8 NCERT books —
        answers come from YOUR indexed PDFs, not the internet.
    </p>
    """, unsafe_allow_html=True)

    # ── Check ChromaDB is available ───────────────────────────
    NCERT_COLLECTION = "ncert_class8"
    NCERT_CHROMA_DIR = "chroma_study_db"

    @st.cache_resource
    def _get_ncert_collection():
        """Load the indexed NCERT ChromaDB collection."""
        try:
            client     = chromadb.PersistentClient(path=NCERT_CHROMA_DIR)
            collection = client.get_collection(name=NCERT_COLLECTION)
            return collection
        except Exception as e:
            return None

    @st.cache_resource
    def _get_ncert_embedder():
        """Load the SentenceTransformer model for querying."""
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    ncert_collection = _get_ncert_collection()
    ncert_embedder   = _get_ncert_embedder()

    # ── Show error if DB not found ────────────────────────────
    if ncert_collection is None or ncert_embedder is None:
        st.error("❌ NCERT database not found.")
        st.markdown("""
        <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);
             border-radius:14px;padding:20px 24px;margin-top:12px;">
          <p style="font-weight:700;color:#fca5a5;margin:0 0 8px 0;">How to fix this:</p>
          <p style="color:#fca5a5;margin:0;">
            1. Make sure <code>index_ncert.py</code> is in your BrainForge folder<br>
            2. Run: <code>python index_ncert.py</code><br>
            3. Wait for indexing to finish<br>
            4. Restart the Streamlit app
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── DB Stats ──────────────────────────────────────────────
    total_chunks = ncert_collection.count()

    st.markdown(f"""
    <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);
         border-radius:12px;padding:12px 18px;margin-bottom:16px;
         display:flex;align-items:center;gap:10px;">
      <span style="font-size:1.2em;">✅</span>
      <span style="color:#86efac;font-weight:600;font-size:0.9em;">
        NCERT database ready — {total_chunks:,} chunks indexed from your PDFs
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── NCERT Retrieval Function ──────────────────────────────
    def retrieve_from_ncert(query: str, subject_filter: str = None, top_k: int = 4) -> list:
        """
        Search ChromaDB for relevant NCERT chunks.
        Returns list of dicts with text, subject, chapter, page, source.
        """
        try:
            query_embedding = ncert_embedder.encode(
                [query], normalize_embeddings=True
            ).tolist()[0]

            # Build filter
            where = None
            if subject_filter and subject_filter != "All Subjects":
                where = {"subject": subject_filter}

            # Query ChromaDB
            results = ncert_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, total_chunks),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Parse results
            chunks = []
            docs      = results.get("documents", [[]])[0]
            metas     = results.get("metadatas",  [[]])[0]
            distances = results.get("distances",  [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                relevance = round((1 - dist) * 100, 1)  # cosine similarity to %
                if relevance > 20:  # only include reasonably relevant chunks
                    chunks.append({
                        "text":      doc,
                        "subject":   meta.get("subject",  "Unknown"),
                        "chapter":   meta.get("chapter",  "Unknown"),
                        "page":      meta.get("page",     "?"),
                        "source":    meta.get("source",   "Unknown"),
                        "type":      meta.get("content_type", "theory"),
                        "relevance": relevance,
                    })

            return chunks

        except Exception as e:
            st.error(f"Retrieval error: {e}")
            return []


    def generate_ncert_answer(question: str, chunks: list, subject: str) -> str:
        """
        Send retrieved NCERT chunks + question to Groq LLM.
        Answer is grounded in actual NCERT content.
        """
        if not chunks:
            return "I couldn't find relevant content in the NCERT books for this question. Try rephrasing or selecting a different subject."

        # Build context from retrieved chunks
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {c['subject']} | {c['chapter']} | Page {c['page']}]\n{c['text']}"
            )
        context = "\n\n".join(context_parts)

        subject_line = f"Subject filter: {subject}" if subject != "All Subjects" else "Searching across all subjects"

        prompt = f"""You are a helpful NCERT tutor for Class 8 students.
Answer the student's question using ONLY the NCERT content provided below.

{subject_line}
Student's question: {question}

NCERT Content Retrieved:
{context}

Instructions:
- Answer clearly and in simple language suitable for Class 8
- Use the NCERT content above as your primary source
- If the content covers the answer, explain it step by step
- Mention which chapter/subject the answer is from
- If the retrieved content is not enough to fully answer, say so honestly
- Do NOT make up information not present in the content
- End with: "📚 Source: [Subject] — [Chapter]"
"""
        return safe_llm_call(
            MAIN_MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        ) or "Could not generate answer. Please try again."


    # ── Available Subjects ────────────────────────────────────
    NCERT_SUBJECTS = [
        "All Subjects",
        "Mathematics",
        "Science",
        "History",
        "Geography",
        "Civics",
        "English",
        "Hindi",
    ]

    # ── Session Init ──────────────────────────────────────────
    if "ncert_messages"       not in st.session_state:
        st.session_state.ncert_messages       = []
    if "ncert_subject_filter" not in st.session_state:
        st.session_state.ncert_subject_filter = "All Subjects"
    if "ncert_show_sources"   not in st.session_state:
        st.session_state.ncert_show_sources   = True

    # ── Controls ──────────────────────────────────────────────
    col_subj, col_toggle = st.columns([2, 1])

    with col_subj:
        selected_subject = st.selectbox(
            "📚 Filter by Subject",
            NCERT_SUBJECTS,
            index=NCERT_SUBJECTS.index(st.session_state.ncert_subject_filter),
            key="ncert_subject_select",
            help="Select a subject to search only that book, or All Subjects to search everything"
        )
        st.session_state.ncert_subject_filter = selected_subject

    with col_toggle:
        st.markdown("<br>", unsafe_allow_html=True)
        show_sources = st.toggle(
            "Show Sources",
            value=st.session_state.ncert_show_sources,
            key="ncert_sources_toggle",
            help="Show which page and chapter each answer came from"
        )
        st.session_state.ncert_show_sources = show_sources

    st.markdown("")

    # ── Suggested Questions ───────────────────────────────────
    if not st.session_state.ncert_messages:
        st.markdown("**💡 Try asking:**")

        suggestions_map = {
            "All Subjects": [
                "What is photosynthesis?",
                "Explain the Pythagorean theorem",
                "What were the causes of the 1857 revolt?",
                "What is a rational number?",
            ],
            "Mathematics": [
                "What are rational numbers?",
                "Explain linear equations in one variable",
                "What is the Pythagorean theorem?",
                "How do you find the area of a trapezium?",
            ],
            "Science": [
                "What is photosynthesis?",
                "Explain cell structure",
                "What is friction?",
                "How does sound travel?",
            ],
            "History": [
                "What were the causes of the 1857 revolt?",
                "Who was Tipu Sultan?",
                "What was the colonial impact on Indian trade?",
                "Explain the role of the press in Indian nationalism",
            ],
            "Geography": [
                "What are natural resources?",
                "Explain land use patterns",
                "What is agriculture?",
                "What are industries?",
            ],
            "Civics": [
                "What is the Indian Constitution?",
                "Explain the role of the Parliament",
                "What is the judiciary?",
                "What are Fundamental Rights?",
            ],
            "English": [
                "Summarize the chapter Bepin Choudhury's Lapse of Memory",
                "What is the theme of the poem The Last Bargain?",
            ],
            "Hindi": [
                "ध्वनि कविता का सारांश बताइए",
                "लाख की चूड़ियाँ पाठ का विषय क्या है?",
            ],
        }

        suggestions = suggestions_map.get(selected_subject, suggestions_map["All Subjects"])
        sugg_cols   = st.columns(2)

        for i, sugg in enumerate(suggestions):
            with sugg_cols[i % 2]:
                if st.button(sugg, key=f"ncert_sugg_{i}", use_container_width=True):
                    if not check_request_limit():
                        st.stop()

                    with st.spinner("🔍 Searching your NCERT books..."):
                        chunks = retrieve_from_ncert(
                            sugg,
                            subject_filter=selected_subject,
                            top_k=4,
                        )

                    with st.spinner("🤖 Generating answer..."):
                        answer = generate_ncert_answer(sugg, chunks, selected_subject)

                    st.session_state.ncert_messages.append({
                        "role":    "user",
                        "content": sugg,
                        "chunks":  [],
                    })
                    st.session_state.ncert_messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "chunks":  chunks,
                    })
                    st.rerun()

        st.markdown("")

    # ── Chat History ──────────────────────────────────────────
    for msg in st.session_state.ncert_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources for assistant messages
            if msg["role"] == "assistant" and show_sources and msg.get("chunks"):
                with st.expander("📄 Sources from your NCERT PDFs", expanded=False):
                    for i, chunk in enumerate(msg["chunks"], 1):
                        relevance_color = (
                            "#22c55e" if chunk["relevance"] >= 70
                            else "#f59e0b" if chunk["relevance"] >= 45
                            else "#94a3b8"
                        )
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.03);
                             border-left:3px solid {relevance_color};
                             border-radius:0 10px 10px 0;
                             padding:10px 14px;margin-bottom:8px;">
                          <div style="display:flex;justify-content:space-between;
                               align-items:center;margin-bottom:6px;">
                            <span style="font-weight:700;color:#a5b4fc;font-size:0.82em;">
                              📚 {chunk['subject']} — {chunk['chapter']}
                            </span>
                            <span style="font-size:0.72em;color:{relevance_color};font-weight:700;">
                              {chunk['relevance']}% match
                            </span>
                          </div>
                          <div style="font-size:0.72em;color:#64748b;margin-bottom:6px;">
                            📄 Page {chunk['page']}  ·  {chunk['source']}  ·  {chunk['type']}
                          </div>
                          <div style="color:#cbd5e1;font-size:0.82em;line-height:1.6;">
                            {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

    # ── Chat Input ────────────────────────────────────────────
    user_question = st.chat_input(
        f"Ask anything from Class 8 NCERT {'(' + selected_subject + ')' if selected_subject != 'All Subjects' else '(All Subjects)'}..."
    )

    if user_question:
        if not check_request_limit():
            st.stop()

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)

        # Retrieve from NCERT
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching your NCERT books..."):
                chunks = retrieve_from_ncert(
                    user_question,
                    subject_filter=selected_subject,
                    top_k=4,
                )

            with st.spinner("🤖 Generating answer from NCERT content..."):
                answer = generate_ncert_answer(user_question, chunks, selected_subject)

            st.markdown(answer)

            # Show sources inline
            if show_sources and chunks:
                with st.expander("📄 Sources from your NCERT PDFs", expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        relevance_color = (
                            "#22c55e" if chunk["relevance"] >= 70
                            else "#f59e0b" if chunk["relevance"] >= 45
                            else "#94a3b8"
                        )
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.03);
                             border-left:3px solid {relevance_color};
                             border-radius:0 10px 10px 0;
                             padding:10px 14px;margin-bottom:8px;">
                          <div style="display:flex;justify-content:space-between;
                               align-items:center;margin-bottom:6px;">
                            <span style="font-weight:700;color:#a5b4fc;font-size:0.82em;">
                              📚 {chunk['subject']} — {chunk['chapter']}
                            </span>
                            <span style="font-size:0.72em;color:{relevance_color};font-weight:700;">
                              {chunk['relevance']}% match
                            </span>
                          </div>
                          <div style="font-size:0.72em;color:#64748b;margin-bottom:6px;">
                            📄 Page {chunk['page']}  ·  {chunk['source']}  ·  {chunk['type']}
                          </div>
                          <div style="color:#cbd5e1;font-size:0.82em;line-height:1.6;">
                            {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

        # Save to session
        st.session_state.ncert_messages.append({
            "role":    "user",
            "content": user_question,
            "chunks":  [],
        })
        st.session_state.ncert_messages.append({
            "role":    "assistant",
            "content": answer,
            "chunks":  chunks,
        })
        st.session_state.journey_step = max(st.session_state.get("journey_step", 0), 2)

    # ── Footer Controls ───────────────────────────────────────
    if st.session_state.ncert_messages:
        st.markdown("")
        st.divider()
        col_clear, col_info = st.columns([1, 3])

        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.ncert_messages = []
                st.rerun()

        with col_info:
            st.caption(
                f"💡 Answers are generated from your {total_chunks:,} indexed NCERT chunks. "
                f"Use subject filter for more precise answers."
            )
