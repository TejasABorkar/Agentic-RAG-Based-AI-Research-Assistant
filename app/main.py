import os
import sys
import tempfile
import uuid

import streamlit as st

# Make the "src" folder importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunker import chunk_pages
from src.llm import LLMError
from src.pdf_loader import load_pdf
from src.qa import answer_question
from src.summarizer import summarize_document
from src.vector_store import add_chunks

st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    layout="wide",
    page_icon=":books:",
)

st.title(":books: Agentic RAG Research Assistant")
st.markdown(
    "*Upload a PDF, get a summary, and chat with it. "
    "Answers come with page citations, or from web search if the document "
    "does not have the answer.*"
)

# ---------- session state ----------
if "collection" not in st.session_state:
    st.session_state.collection = "doc_" + uuid.uuid4().hex
    st.session_state.processed = False
    st.session_state.summary = None
    st.session_state.messages = []
    st.session_state.doc_info = ""


def render_assistant(result, show_thoughts):
    st.write(result["answer"])

    source = result["source"]
    if source == "document":
        pages = ", ".join(str(p) for p in result["pages"])
        st.success(f"Source: Document - page(s) {pages}")
    elif source == "web":
        st.warning("Source: Web search")
        for s in result.get("web_sources", []):
            st.markdown(f"- [{s['title'] or s['url']}]({s['url']})")
    elif source == "none":
        st.info("No reliable source was found.")
    else:
        st.error("The language model could not be reached.")

    if show_thoughts:
        with st.expander("Agent thoughts"):
            for step in result.get("steps", []):
                st.write("- " + step)
            for i, h in enumerate(result.get("chunks", []), 1):
                st.caption(
                    f"Chunk {i} - page {h['page']} - similarity {h['similarity']}"
                )
                st.text(h["text"][:500])


# ---------- sidebar ----------
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Processing document... (this may take a minute)"):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                pages = load_pdf(tmp_path, display_name=uploaded_file.name)

                if not pages:
                    st.error(
                        "No readable text found. This may be a scanned PDF "
                        "(OCR is not supported yet)."
                    )
                else:
                    chunks, metadatas = chunk_pages(pages)
                    add_chunks(chunks, metadatas, st.session_state.collection)

                    try:
                        summary = summarize_document(pages)
                    except LLMError as e:
                        summary = None
                        st.warning(f"Summary unavailable: {str(e)[:200]}")

                    st.session_state.processed = True
                    st.session_state.summary = summary
                    st.session_state.messages = []
                    st.session_state.doc_info = (
                        f"{uploaded_file.name}: {len(pages)} pages, "
                        f"{len(chunks)} chunks"
                    )
                    st.success(
                        f"Processed {len(pages)} pages into {len(chunks)} chunks!"
                    )
            except Exception as e:
                st.error(f"Could not process the PDF: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.divider()
    show_thoughts = st.checkbox("Show agent thoughts (debug info)", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []

    st.divider()
    st.caption("Built with Streamlit, ChromaDB, Gemini and Groq")

# ---------- main area ----------
if st.session_state.processed:
    st.caption(st.session_state.doc_info)

    if st.session_state.summary:
        st.subheader("Document Summary")
        st.info(st.session_state.summary)

    st.divider()
    st.subheader("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant(msg["meta"], show_thoughts)
            else:
                st.write(msg["content"])

    question = st.chat_input("Ask a question about the document")
    if question:
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                result = answer_question(
                    question, st.session_state.collection, history
                )
            render_assistant(result, show_thoughts)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "meta": result}
        )
else:
    st.info("Please upload a PDF from the sidebar and click 'Process Document'.")