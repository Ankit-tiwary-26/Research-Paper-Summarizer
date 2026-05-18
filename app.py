import os
import re
import json
import tempfile
import streamlit as st

from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()
GROQ_API_KEY = st.secrets("GROQ_API_KEY")
# ==========================================
# STREAMLIT CONFIG
# ==========================================

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .chat-user {
        background: #e8f4fd;
        border-left: 4px solid #1976D2;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    .chat-ai {
        background: #f3f9f1;
        border-left: 4px solid #388E3C;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .suggestion-btn {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px;
        cursor: pointer;
        font-size: 0.85rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

defaults = {
    "qa_chain": None,
    "chat_history": [],
    "raw_chat_history": [],        # LangChain format
    "summary_cache": {},           # Cache summaries by type
    "paper_metadata": {},          # Extracted metadata
    "documents": None,             # Raw loaded pages
    "keywords": [],
    "pdf_name": "",
    "processing_done": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==========================================
# SIDEBAR — SETTINGS & CONTROLS
# ==========================================

with st.sidebar:
    st.header(" Settings")

    st.subheader(" Model")
    model_choice = st.selectbox(
        "Model",
        options=["llama-3.1-8b-instant","llama-3.3-70b-versatile"],
        help="Flash is faster; Pro is more detailed"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower = factual, Higher = creative"
    )

    st.subheader(" Chunking")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 400, 200, 50)

    st.subheader(" Retrieval")
    top_k = st.slider(
        "Top-K Chunks",
        1, 8, 4,
        help="How many chunks to retrieve per query"
    )

    st.divider()

    if st.session_state.processing_done:
        st.success(f" **{st.session_state.pdf_name}**")
        st.caption(f"{len(st.session_state.documents)} pages loaded")

    st.divider()

    if st.button(" Reset Session", use_container_width=True):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()

    st.divider()
    st.caption("Research Paper Assistant v2.0")

# ==========================================
# HEADER
# ==========================================

st.title("📄 Research Paper Assistant")
st.markdown(
    "Upload a research paper PDF to get summaries, insights, keywords, "
    "and an interactive chat interface."
)
st.divider()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load HuggingFace embeddings model (cached globally)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def extract_metadata(documents, qa_chain):
    """Use LLM to extract paper metadata."""
    response = qa_chain.invoke({
        "question": (
            "Extract the following from the paper and respond ONLY in valid JSON "
            "with these exact keys: title, authors, year, journal, abstract_summary "
            "(2 sentences max), domain. If unknown, use null."
        )
    })
    raw = response["answer"]
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}


def extract_keywords(qa_chain):
    """Extract top keywords/topics."""
    response = qa_chain.invoke({
        "question": (
            "List the 10 most important technical keywords or topics from this paper. "
            "Return ONLY a comma-separated list, no numbering, no extra text."
        )
    })
    raw = response["answer"]
    return [kw.strip() for kw in raw.split(",") if kw.strip()][:10]


def build_qa_chain(chunks, model_name, temp, top_k):
    """Build and return the ConversationalRetrievalChain."""
    embeddings = load_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    llm = ChatGroq(
        model=model_name,
        groq_api_key=GROQ_API_KEY,
        temperature=temp
    )

    # Custom prompt to keep answers grounded in the paper
    condense_prompt = PromptTemplate.from_template(
        "Given the conversation history and the new question, "
        "rephrase it as a standalone question.\n\n"
        "History:\n{chat_history}\n\nQuestion: {question}\n\nStandalone:"
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert research paper analyst. "
            "Use ONLY the provided context from the paper to answer. "
            "If the answer is not in the context, say so clearly.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer (be precise and cite page numbers if possible):"
        )
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )


def get_summary(summary_type: str, prompt: str):
    """Get a summary, using cache to avoid re-querying."""
    if summary_type in st.session_state.summary_cache:
        return st.session_state.summary_cache[summary_type]
    with st.spinner(f"Generating {summary_type}..."):
        response = st.session_state.qa_chain.invoke({"question": prompt})
        result = response["answer"]
        st.session_state.summary_cache[summary_type] = result
        return result


def export_chat_history():
    """Export chat history as a formatted text string."""
    lines = [f"# Chat History — {st.session_state.pdf_name}",
             f"# Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for sender, message in st.session_state.chat_history:
        lines.append(f"## {sender}\n{message}\n")
    return "\n".join(lines)

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a Research Paper (PDF)",
    type="pdf",
    help="Maximum recommended size: 50MB"
)

# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_file and (not st.session_state.processing_done or
                      uploaded_file.name != st.session_state.pdf_name):

    st.session_state.pdf_name = uploaded_file.name
    st.session_state.summary_cache = {}   # clear cache for new file
    st.session_state.chat_history = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    progress = st.progress(0, text="Starting...")

    try:
        # Step 1: Load
        progress.progress(10, text="Loading PDF...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        st.session_state.documents = documents

        # Step 2: Split
        progress.progress(30, text=" Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        # Step 3: Build chain
        progress.progress(60, text="Building vector store & LLM chain...")
        qa_chain = build_qa_chain(chunks, model_choice, temperature, top_k)
        st.session_state.qa_chain = qa_chain

        # Step 4: Metadata
        progress.progress(80, text="🔍 Extracting metadata & keywords...")
        st.session_state.paper_metadata = extract_metadata(documents, qa_chain)
        st.session_state.keywords = extract_keywords(qa_chain)

        progress.progress(100, text=" Done!")
        st.session_state.processing_done = True

    except Exception as e:
        st.error(f" Error processing PDF: {e}")
        st.stop()
    finally:
        os.unlink(pdf_path)
        progress.empty()

    st.success(f"✅ **{uploaded_file.name}** processed — "
               f"{len(documents)} pages, {len(chunks)} chunks")

# ==========================================
# MAIN CONTENT (only after processing)
# ==========================================

if st.session_state.processing_done:

    # ---- PAPER METADATA CARD ----
    meta = st.session_state.paper_metadata
    if meta:
        with st.container():
            st.subheader(" Paper Overview")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Title**\n\n{meta.get('title', 'N/A')}")
            c2.markdown(f"**Authors**\n\n{meta.get('authors', 'N/A')}")
            c3.markdown(
                f"**Year / Journal**\n\n"
                f"{meta.get('year', '?')} / {meta.get('journal', 'N/A')}"
            )
            if meta.get("abstract_summary"):
                st.info(f" **Abstract:** {meta['abstract_summary']}")
        st.divider()


    # ---- STATS ----
    docs = st.session_state.documents
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(" Pages", len(docs))
    total_words = sum(len(d.page_content.split()) for d in docs)
    c2.metric(" Words", f"{total_words:,}")
    c3.metric("Model", model_choice.split("-")[-1].upper())
    c4.metric(" Q&A Turns", len(st.session_state.chat_history) // 2)
    st.divider()

    # ==========================================
    # TABS: Summary | Insights | Chat
    # ==========================================

    tab1, tab2, tab3 = st.tabs(["📌 Summaries", "🔬 Deep Insights", "💬 Chat"])

    # ---- TAB 1: SUMMARIES ----
    with tab1:
        st.subheader("Generate Summaries")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Full Summary", use_container_width=True):
                result = get_summary("full", """
                    Provide a structured analysis with these sections:
                    1. **Technical Summary** (3-4 sentences for experts)
                    2. **Simple Summary** (3-4 sentences for non-experts)
                    3. **Main Contributions** (bullet points)
                    4. **Methodology** (how they did it)
                    5. **Key Results** (with numbers if possible)
                    6. **Limitations** (bullet points)
                    7. **Future Scope** (bullet points)
                """)
                st.markdown(result)

        with col2:
            if st.button(" Quick Details", use_container_width=True):
                result = get_summary("tldr",
                    "Summarize this paper in exactly 3 bullet points. "
                    "Be concise and specific.")
                st.markdown(result)

        with col3:
            if st.button(" Methodology Only", use_container_width=True):
                result = get_summary("methodology",
                    "Describe the methodology, datasets, experimental setup, "
                    "and evaluation metrics used in this paper in detail.")
                st.markdown(result)

        st.divider()

        # Download summaries
        if st.session_state.summary_cache:
            combined = "\n\n---\n\n".join(
                f"### {k.upper()}\n{v}"
                for k, v in st.session_state.summary_cache.items()
            )
            st.download_button(
                " Download All Summaries (.txt)",
                data=combined,
                file_name=f"{st.session_state.pdf_name}_summaries.txt",
                mime="text/plain",
                use_container_width=True
            )

    # ---- TAB 2: DEEP INSIGHTS ----
    with tab2:
        st.subheader("Deep Insights")

        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            if st.button("Datasets & Benchmarks", use_container_width=True):
                result = get_summary("datasets",
                    "List all datasets, benchmarks, and evaluation metrics "
                    "mentioned in this paper with brief descriptions.")
                st.markdown(result)

        with insight_col2:
            if st.button(" Compared Methods", use_container_width=True):
                result = get_summary("baselines",
                    "List all baseline methods or competing approaches the "
                    "authors compared against, and briefly state the outcome.")
                st.markdown(result)

        insight_col3, insight_col4 = st.columns(2)

        with insight_col3:
            if st.button(" Key Equations / Formulas", use_container_width=True):
                result = get_summary("equations",
                    "Identify and explain the key mathematical equations, "
                    "formulas, or algorithms presented in this paper.")
                st.markdown(result)

        with insight_col4:
            if st.button(" Research Gaps Identified", use_container_width=True):
                result = get_summary("gaps",
                    "What research gaps, open problems, or future directions "
                    "do the authors identify in this paper?")
                st.markdown(result)

        st.divider()

        # Critical review
        if st.button("Generate Critical Review", use_container_width=True):
            result = get_summary("critical",
                "Provide a balanced critical review of this paper: "
                "strengths, weaknesses, reproducibility concerns, "
                "and whether the claims are well-supported by evidence.")
            st.markdown(result)

    # ---- TAB 3: CHAT ----
    with tab3:
        st.subheader("Chat with the Paper")

        # Suggested questions
        st.markdown("** Suggested Questions:**")
        suggestions = [
            "What problem does this paper solve?",
            "What are the main results?",
            "How does this compare to prior work?",
            "What are the limitations?",
            "Can you explain the architecture?",
        ]
        sug_cols = st.columns(len(suggestions))
        for i, (col, sug) in enumerate(zip(sug_cols, suggestions)):
            if col.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["prefill_question"] = sug

        st.divider()

        # Chat input
        prefill = st.session_state.pop("prefill_question", "")
        user_question = st.chat_input("Ask anything about the paper...")

        # Use prefill if button was clicked
        active_question = user_question or prefill

        if active_question:
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain.invoke({
                        "question": active_question
                    })
                    answer = result["answer"]
                    sources = result.get("source_documents", [])

                    st.session_state.chat_history.append(("You", active_question))
                    st.session_state.chat_history.append(("AI", answer))
                    st.session_state["last_sources"] = sources

                except Exception as e:
                    st.error(f" Error: {e}")

        # Render chat history
        if st.session_state.chat_history:
            for sender, message in st.session_state.chat_history:
                with st.chat_message("user" if sender == "You" else "assistant"):
                    st.markdown(message)

            # Sources for last response
            last_sources = st.session_state.get("last_sources", [])
            if last_sources:
                with st.expander(f" Source References ({len(last_sources)} chunks)"):
                    for i, doc in enumerate(last_sources):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i+1} — Page {page}**")
                        st.markdown(
                            f'<div style="background:#f5f5f5;padding:10px;'
                            f'border-radius:6px;font-size:0.85rem">'
                            f'{doc.page_content[:400]}...</div>',
                            unsafe_allow_html=True
                        )

            # Export chat
            st.divider()
            col_a, col_b = st.columns([3, 1])
            col_a.download_button(
                "⬇Download Chat History",
                data=export_chat_history(),
                file_name=f"{st.session_state.pdf_name}_chat.txt",
                mime="text/plain",
                use_container_width=True
            )
            if col_b.button(" Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state["last_sources"] = []
                st.rerun()

else:
    # ---- PLACEHOLDER ----
    st.info(
        " Upload a PDF above to get started. "
        "You can then generate summaries, explore insights, "
        "and chat with the paper."
    )
    with st.expander(" What can this tool do?"):
        st.markdown("""
        -  **Full structured summaries** — technical, simple, contributions, limitations
        -  **TL;DR** — 3-bullet quick overview
        -  **Deep insights** — datasets, baselines, equations, research gaps
        -  **Critical review** — strengths, weaknesses, reproducibility
        -  **Interactive chat** — ask anything about the paper
        -  **Auto keyword extraction**
        -  **Paper metadata** — title, authors, year, journal
        -  **Export** summaries and chat history
        """)
