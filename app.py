import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import google.generativeai as genai
from setup import build_vectorstore
build_vectorstore()

load_dotenv()
# temporary debug - remove after fixing
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    models = list(genai.list_models())
    st.write(f"API connected. Models available: {len(models)}")
except Exception as e:
    st.error(f"API Error: {str(e)}")
    st.stop()
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BU Student Advisor",
    page_icon="🎓",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; border-bottom: 2px solid #f0f0f0; margin-bottom: 1.5rem; }
    .source-box { background: #f8f9fa; border-left: 3px solid #4CAF50; padding: 0.5rem 1rem; margin-top: 0.4rem; border-radius: 4px; font-size: 0.85rem; color: #555; }
    .disclaimer { text-align: center; color: #888; font-size: 0.8rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2>🎓 BU Student Advisor</h2>
    <p>Ask me anything about courses, course outlines (bscs only), policies, university guidelines, scholarships and other queries you may have.</p>
</div>
""", unsafe_allow_html=True)

# ── Load resources once ───────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )
    return vectorstore, llm

vectorstore, llm = load_resources()

# ── Core RAG function ─────────────────────────────────────────────────────────
def ask_question(question, chat_history):
    # build history string
    history_text = ""
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            history_text += f"Student: {msg['content']}\n"
        else:
            history_text += f"Advisor: {msg['content']}\n"

    # detect if student is asking for a course outline
    outline_keywords = ["course outline", "syllabus", "weekly plan", 
                   "topics covered", "course content", "weekly topics",
                   "show me the outline", "give me the outline"]
    is_outline_request = any(kw in question.lower() for kw in outline_keywords)

    if is_outline_request:
        # get more chunks and filter by source document
        docs = vectorstore.similarity_search(question, k=20)
        
        # find which document is most relevant
        source_counts = {}
        for doc in docs:
            source = doc.metadata.get("source", "")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # get the top matching document
        if source_counts:
            top_source = max(source_counts, key=source_counts.get)
            # now get ALL chunks from that specific document
            all_docs = vectorstore.similarity_search(question, k=50)
            docs = [d for d in all_docs if d.metadata.get("source", "") == top_source]
            # sort by page number so outline is in order
            docs = sorted(docs, key=lambda d: d.metadata.get("page", 0))
    else:
        docs = vectorstore.similarity_search(question, k=6)

    # build context from docs
    context = "\n\n".join([doc.page_content for doc in docs])

    # build sources
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "")
        filename = os.path.basename(source)
        page = doc.metadata.get("page", "")
        label = f"{filename} — page {int(page)+1}" if page != "" else filename
        if label not in seen:
            seen.add(label)
            sources.append(label)

    # build prompt
    if is_outline_request:
        prompt = f"""You are a helpful student advisor at Bahria University.
The student is asking for a course outline. Provide the COMPLETE course outline from the document below.
Include all weeks, topics, textbooks, grading, and any other details present.
Answer ONLY the current question. Ignore any previous conversation context.

Course document content:
{context}

Student question: {question}

Complete Course Outline:"""
    else:
        prompt = f"""You are a helpful student advisor at Bahria University.
Answer ONLY this specific question using the context below from official university documents.
Do NOT reference or answer any previous questions.
If the answer is not in the context, say "I don't have that information. Please contact your advisor directly."
Be friendly, clear and concise.

Relevant university documents:
{context}

Student question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, sources

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Suggested questions (only shown at start) ─────────────────────────────────
if len(st.session_state.messages) == 0:
    st.markdown("#### Common questions students ask:")
    col1, col2 = st.columns(2)
    with col1:
        
        if st.button("📚 CS courses offered"):
            st.session_state.pending = "What CS courses are offered?"
    with col2:
        if st.button("📅 Attendance policy"):
            st.session_state.pending = "What is the attendance policy?"
        if st.button("🎓 Grading system"):
            st.session_state.pending = "How does the grading system work?"

# ── Display chat history ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f'<div class="source-box">{source}</div>',
                               unsafe_allow_html=True)

# ── Handle pending question from buttons ──────────────────────────────────────
if "pending" in st.session_state:
    question = st.session_state.pop("pending")
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching university documents..."):
            answer, sources = ask_question(question, st.session_state.messages)
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for source in sources:
                    st.markdown(f'<div class="source-box">{source}</div>',
                               unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about BU policies, courses, or guidelines..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching university documents..."):
            answer, sources = ask_question(prompt, st.session_state.messages)
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for source in sources:
                    st.markdown(f'<div class="source-box">{source}</div>',
                               unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    Answers are based on official BU documents. Always verify important decisions with your academic advisor.
</div>
""", unsafe_allow_html=True)