import streamlit as st
import tempfile
import os
import sys

# src फोल्डर इम्पोर्ट करण्यासाठी पाथ सेट करा
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_loader import load_pdf
from src.chunker import chunk_text
from src.vector_store import add_chunks
from src.summarizer import summarize_document
from src.qa import answer_question

# पेज कॉन्फिगरेशन
st.set_page_config(page_title="Agentic RAG Research Assistant", layout="wide", page_icon="📚")

st.title("📚 Agentic RAG Research Assistant")
st.markdown("*Upload a PDF, get a summary, and ask questions with page citations or web search.*")

# साइडबार - फाईल अपलोड
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Processing document... (This may take a minute)"):
            # तात्पुरती फाईल सेव्ह करा
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            # १. PDF लोड करा
            pages = load_pdf(tmp_path)
            
            # २. चंकिंग करा
            all_chunks = []
            all_metadata = []
            for p in pages:
                chunks = chunk_text(p["text"])
                for c in chunks:
                    all_chunks.append(c)
                    all_metadata.append({"page": p["page"], "source": p["source"]})
            
            # ३. ChromaDB मध्ये सेव्ह करा
            add_chunks(all_chunks, all_metadata)
            
            # ४. समरी तयार करा
            summary = summarize_document(pages)
            
            # ५. सेशन स्टेटमध्ये सेव्ह करा
            st.session_state["processed"] = True
            st.session_state["summary"] = summary
            st.session_state["total_pages"] = len(pages)
            st.session_state["total_chunks"] = len(all_chunks)
            
            # तात्पुरती फाईल डिलीट करा
            os.remove(tmp_path)
            
            st.success(f"Processed {len(pages)} pages into {len(all_chunks)} chunks!")

# मुख्य भाग - समरी आणि प्रश्न
if st.session_state.get("processed"):
    st.subheader("📝 Document Summary")
    st.info(st.session_state["summary"])
    
    st.divider()
    st.subheader("💬 Ask a Question")
    
    question = st.text_input("Enter your question about the document:")
    show_thoughts = st.checkbox("Show agent thoughts (debug info)", value=True)
    
    if question and st.button("Ask", type="primary"):
        with st.spinner("Agent is thinking..."):
            result = answer_question(question)
        
        st.markdown("### Answer")
        st.write(result["answer"])
        
        # सोर्स लेबल
        if result["source"] == "From document":
            st.success(f"📄 **Source:** From Document — Page(s): {result['pages']}")
        elif result["source"] == "From web":
            st.warning(f"🌐 **Source:** From Web Search")
            for url in result.get("urls", []):
                st.write(f"- [{url}]({url})")
        else:
            st.error(result["source"])
        
        # एजंट थॉट्स (डिबग मोड)
        if show_thoughts:
            with st.expander("🔍 Agent Thoughts"):
                st.write(f"**Decision:** {result['source']}")
                if "chunks_used" in result:
                    st.write("**Retrieved Chunks:**")
                    for i, c in enumerate(result["chunks_used"]):
                        st.text_area(f"Chunk {i+1}", c[:500], height=100)
else:
    st.info("👈 Please upload a PDF file from the sidebar to get started.")

# फूटर
st.sidebar.divider()
st.sidebar.caption("Built with ❤️ using Streamlit, ChromaDB & Groq")