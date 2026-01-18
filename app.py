import streamlit as st
import time
from src.rag_pipeline import run_rag

st.set_page_config(page_title="Hybrid RAG System", layout="wide")

st.title("Hybrid RAG System")

query = st.text_input("Ask your question:")

if st.button("Search") and query.strip():

    start_time = time.time()

    result = run_rag(query)

    end_time = time.time()
    latency = round(end_time - start_time, 2)

    # ---------------- ANSWER ----------------
    st.subheader("Answer")
    st.write(result["answer"])

    st.caption(f"Response Time: {latency} seconds")

    # ---------------- SOURCES ----------------
    st.subheader("Top Sources")

    for url in result["sources"]:
        st.write(url)

    # ---------------- RETRIEVAL DETAILS ----------------
    st.subheader("Retrieval Details")

    tab1, tab2, tab3 = st.tabs(["Dense Retrieval", "Sparse Retrieval (BM25)", "RRF Fusion"])

    # ---------- Dense ----------
    with tab1:
        for item in result["dense_results"][:5]:
            st.markdown(f"""
            **Rank:** {item['rank']}  
            **Score:** {round(item['score'], 4)}  
            **Source:** {item['url']}  
            **Chunk:** {item['chunk'][:300]}...
            ---
            """)

    # ---------- Sparse ----------
    with tab2:
        for item in result["sparse_results"][:5]:
            st.markdown(f"""
            **Rank:** {item['rank']}  
            **Score:** {round(item['score'], 4)}  
            **Source:** {item['url']}  
            **Chunk:** {item['chunk'][:300]}...
            ---
            """)

    # ---------- RRF ----------
    with tab3:
        for i, item in enumerate(result["rrf_results"][:5]):
            st.markdown(f"""
            **Final Rank:** {i+1}  
            **RRF Score:** {round(item['rrf_score'], 4)}  
            **Source:** {item['url']}  
            **Chunk:** {item['chunk'][:300]}...
            ---
            """)
