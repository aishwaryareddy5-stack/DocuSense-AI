import hashlib
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuSense AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(r"""
<style>
.stApp { background: #f5f9fa; }
.block-container { padding-top: 1.5rem; padding-bottom: 6rem; max-width: 1250px; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5eeee; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
.sidebar-brand { text-align: center; padding: 8px 10px 18px 10px; }
.sidebar-icon { font-size: 3.2rem; line-height: 1; margin-bottom: 10px; }
.sidebar-title { font-size: 1.45rem; font-weight: 750; color: #087f7a; margin-bottom: 8px; }
.sidebar-subtitle { font-size: .82rem; line-height: 1.55; color: #64748b; }
.sidebar-divider { height: 1px; background: #e5eeee; margin: 12px 0 20px 0; }
.about-title { font-size: 1.05rem; font-weight: 700; color: #102a43; margin-bottom: 10px; }
.about-text { font-size: .86rem; line-height: 1.65; color: #64748b; margin-bottom: 18px; }
.status-card { background: #f0fdfa; border: 1px solid #b8eee6; border-radius: 12px; padding: 11px 12px; color: #087f7a; font-size: .82rem; margin-bottom: 10px; }
.status-card.neutral { background: #f8fafc; border-color: #e2e8f0; color: #64748b; }
.status-dot { color: #0d9488; margin-right: 6px; }
.upload-title { font-size: .92rem; font-weight: 700; color: #102a43; margin-bottom: 8px; }
.upload-help { font-size: .76rem; line-height: 1.5; color: #64748b; margin-bottom: 10px; }
.disclaimer { background: #f0fdfa; border: 1px solid #b8eee6; border-radius: 14px; padding: 14px; margin-top: 18px; }
.disclaimer-title { font-size: .88rem; font-weight: 700; color: #087f7a; margin-bottom: 8px; }
.disclaimer-text { font-size: .76rem; line-height: 1.55; color: #475569; }
section[data-testid="stSidebar"] .stButton > button { border: 1px solid #d7e2e5; border-radius: 10px; background: #ffffff; color: #334e68; font-weight: 600; padding: .55rem .8rem; transition: all .2s ease; }
section[data-testid="stSidebar"] .stButton > button:hover { border-color: #14b8a6; color: #087f7a; background: #f0fdfa; }
.hero { position: relative; overflow: hidden; background: linear-gradient(135deg, #087f7a 0%, #0d9488 50%, #14b8a6 100%); border-radius: 24px; padding: 38px 42px; margin-bottom: 22px; box-shadow: 0 15px 35px rgba(8,127,122,.15); }
.hero::after { content: ""; position: absolute; width: 170px; height: 170px; right: -45px; top: -75px; border-radius: 50%; background: rgba(255,255,255,.10); }
.hero-content { position: relative; z-index: 2; }
.hero-title { color: white; font-size: 2.35rem; font-weight: 800; margin: 0; letter-spacing: -.7px; }
.hero-subtitle { color: rgba(255,255,255,.92); font-size: 1rem; margin-top: 8px; }
.hero-badge { display: inline-block; margin-top: 18px; padding: 7px 13px; border-radius: 30px; background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.25); color: white; font-size: .78rem; font-weight: 600; }
.welcome-card { background: white; border: 1px solid #e3ecee; border-radius: 20px; padding: 28px 30px; margin-bottom: 24px; box-shadow: 0 8px 25px rgba(15,23,42,.05); }
.welcome-title { color: #102a43; font-size: 1.35rem; font-weight: 750; margin-bottom: 10px; }
.welcome-text { color: #64748b; font-size: .94rem; line-height: 1.7; margin-bottom: 18px; }
.feature-row { display: flex; gap: 10px; flex-wrap: wrap; }
.feature-pill { display: inline-block; background: #f0fdfa; border: 1px solid #b8eee6; color: #087f7a; border-radius: 30px; padding: 7px 12px; font-size: .76rem; font-weight: 600; }
.answer-card { background: #ffffff; border: 1px solid #dcebed; border-radius: 16px; padding: 18px 20px; margin: 8px 0 14px 0; }
.answer-label { color: #087f7a; font-size: .86rem; font-weight: 800; margin-bottom: 8px; }
.extension-label { color: #475569; font-size: .86rem; font-weight: 800; margin-bottom: 8px; }
[data-testid="stChatMessage"] { border-radius: 16px; margin-bottom: 12px; }
[data-testid="stChatMessageContent"] { line-height: 1.65; }
[data-testid="stChatInput"] { background: white; border-top: 1px solid #e5eeee; padding-top: 10px; }
[data-testid="stChatInput"] textarea { border-radius: 16px; }
@media (max-width: 768px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .hero { padding: 28px 24px; border-radius: 18px; }
  .hero-title { font-size: 1.8rem; }
  .welcome-card { padding: 22px; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# RAG CONFIGURATION
# ============================================================

HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ANSWER_PROMPT = """
You are a careful educational assistant answering questions strictly from the uploaded PDF.

Use the provided context as the primary and only source for the "Answer from PDF" section.

Give a comprehensive, educational answer using as much relevant information from the retrieved PDF context as possible. Do not give an unnecessarily short answer when the context contains more useful details.

Explain the topic clearly and in a structured way. When the PDF provides definitions, concepts, steps, characteristics, examples, classifications, comparisons, advantages, disadvantages, or other relevant details, include them when they directly help answer the user's question.

Preserve the meaning and terminology of the PDF. Do not invent information or add outside facts to this section.

If the retrieved context does not contain enough information to answer the question, clearly say that the PDF does not contain enough information to answer it rather than guessing.

Context:

{context}

Question:

{question}

Write a detailed but focused answer based only on the provided PDF context.
"""

# ============================================================
# MODEL / EMBEDDING HELPERS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm():
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing. Add it to your .env file locally or to your deployment secrets.")

    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        max_new_tokens=1536,
        huggingfacehub_api_token=HF_TOKEN,
    )
    return ChatHuggingFace(llm=llm)


def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])


# ============================================================
# PDF -> CHUNKS -> EMBEDDINGS -> FAISS
# ============================================================

@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_bytes, file_signature):
    del file_signature  # used by Streamlit as a stable cache key

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        documents = PyPDFLoader(temp_path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No readable text was found in this PDF. Try a text-based PDF rather than a scanned image-only PDF.")

        vectorstore = FAISS.from_documents(chunks, get_embedding_model())
        return vectorstore, len(documents), len(chunks)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def get_pdf_answer(vectorstore, question):
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(ANSWER_PROMPT)},
    )
    response = qa_chain.invoke({"query": question})
    return response["result"], response["source_documents"]

EXTENSION_PROMPT = """
You are an AI learning assistant.

The user asked:

{question}

The PDF-grounded answer was:

{answer}

Give a useful educational extension that helps the user understand the topic better.

This section may use general knowledge and does NOT have to be present in the PDF.

Do not pretend that the extension came from the PDF.

When useful, provide roughly 4-6 clear sentences with an intuitive explanation, practical connection, simple example, or helpful context. Avoid repeating the PDF answer unnecessarily. If no useful extension is possible, say so.
"""

def get_extension(question, answer):
    response = get_llm().invoke(
        EXTENSION_PROMPT.format(question=question, answer=answer)
    )
    return response.content if hasattr(response, "content") else str(response)


def make_response(answer, extension):
    return f"### 📄 Answer from PDF\n{answer}\n\n### 🤖 Extension from AI\n{extension}"


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_signature" not in st.session_state:
    st.session_state.document_signature = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "document_pages" not in st.session_state:
    st.session_state.document_pages = 0
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = 0

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.html("""
        <div class="sidebar-brand">
            <div class="sidebar-icon">📚</div>
            <div class="sidebar-title">DocuSense AI</div>
            <div class="sidebar-subtitle">
                Intelligent document assistant powered by Retrieval-Augmented Generation
            </div>
        </div>
    """)

    st.html("""
        <div class="sidebar-divider"></div>
        <div class="upload-title">📎 Upload your PDF</div>
        <div class="upload-help">
            Upload notes, textbooks, research papers, reports, manuals, or any other text-based PDF.
        </div>
    """)

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        signature = hashlib.sha256(pdf_bytes).hexdigest()

        if signature != st.session_state.document_signature:
            try:
                with st.spinner("Preparing your PDF for RAG..."):
                    vectorstore, pages, chunks = build_vectorstore(pdf_bytes, signature)
                st.session_state.vectorstore = vectorstore
                st.session_state.document_signature = signature
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_pages = pages
                st.session_state.document_chunks = chunks
                st.session_state.messages = []
                st.success("PDF is ready for questions.")
            except Exception as exc:
                st.session_state.vectorstore = None
                st.error(f"Could not process the PDF: {exc}")

    st.html('<div class="sidebar-divider"></div>')

    if st.session_state.vectorstore is not None:
        st.html(f"""
            <div class="status-card">
                <span class="status-dot">●</span>
                <strong>Document ready</strong><br>
                {st.session_state.document_name}<br>
                {st.session_state.document_pages} pages · {st.session_state.document_chunks} chunks
            </div>
        """)
    else:
        st.html("""
            <div class="status-card neutral">
                <span>●</span>
                Upload a PDF to activate the document chat.
            </div>
        """)

    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.html("""
        <div class="sidebar-divider"></div>
        <div class="about-title">About DocuSense AI</div>
        <div class="about-text">
            DocuSense AI converts an uploaded PDF into searchable chunks,
            creates vector embeddings, retrieves relevant passages, and
            uses an LLM to answer questions from the document.
        </div>
        <div class="disclaimer">
            <div class="disclaimer-title">ℹ️ Response Transparency</div>
            <div class="disclaimer-text">
                “Answer from PDF” is grounded in retrieved document context.
                “Extension from AI” is an additional explanation and may use
                general model knowledge beyond the uploaded PDF.
            </div>
        </div>
    """)

# ============================================================
# MAIN APPLICATION
# ============================================================

st.html("""
    <div class="hero">
        <div class="hero-content">
            <div class="hero-title">📚 DocuSense AI</div>
            <div class="hero-subtitle">Your intelligent PDF document assistant</div>
            <div class="hero-badge">✦ Retrieval-Augmented Generation</div>
        </div>
    </div>
""")

if len(st.session_state.messages) == 0:
    st.html("""
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome to DocuSense AI</div>
            <div class="welcome-text">
                Upload any PDF from the sidebar and ask questions about it.
                The app retrieves relevant information from your document first,
                then provides a clearly separated AI extension to help you learn more.
            </div>
            <div class="feature-row">
                <div class="feature-pill">📎 Custom PDF upload</div>
                <div class="feature-pill">🔎 RAG-powered retrieval</div>
                <div class="feature-pill">🧠 HuggingFace embeddings</div>
                <div class="feature-pill">🤖 LLM extension</div>
            </div>
        </div>
    """)

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input(
    "Ask a question about your PDF...",
    disabled=st.session_state.vectorstore is None,
)

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and generating an answer..."):
                answer, source_documents = get_pdf_answer(st.session_state.vectorstore, prompt)
                extension = get_extension(prompt, answer)
                final_response = make_response(answer, extension)
            st.markdown(final_response)

            with st.expander("🔎 Retrieved PDF sources"):
                if source_documents:
                    for index, document in enumerate(source_documents, start=1):
                        page_number = document.metadata.get("page")
                        page_text = document.page_content.strip()
                        page_label = f"Page {page_number + 1}" if isinstance(page_number, int) else "PDF passage"
                        st.markdown(f"**Source {index} — {page_label}**")
                        st.caption(page_text[:900] + ("..." if len(page_text) > 900 else ""))

        st.session_state.messages.append({"role": "assistant", "content": final_response})

    except Exception as exc:
        error_message = f"I couldn't process that question. Error: {exc}"
        st.error(error_message)
