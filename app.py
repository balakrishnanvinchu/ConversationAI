import streamlit as st
from code.rag_pipeline import run_rag

st.title("Hybrid RAG System")

query = st.text_input("Ask your question:")

if st.button("Search"):

    answer, urls = run_rag(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for u in urls[:5]:
        st.write(u)
