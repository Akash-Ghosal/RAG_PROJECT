import streamlit as st
from embeddings import get_top_k
from utils import generate_answer
from agents import agent_router
st.write("✅ App is running!")
st.title("📚 RAG-Powered Q&A Assistant")

query = st.text_input("What are the product features?")
if query:
    route = agent_router(query)
    st.write(f"🧠 Agent Decision: `{route}`")

    if route == "rag":
        top_docs = get_top_k(query)
        st.write("📄 Top Matching Chunks:")
        for doc in top_docs:
            st.code(doc[:300])  # first 300 chars
        answer = generate_answer(query, "\n".join(top_docs))
        st.success(answer)

    elif route == "calculator":
        st.write("🧮 This is a calculator task (you can plug in eval(query) or another tool).")

    elif route == "dictionary":
        st.write("📖 This is a dictionary lookup task (you can call an API like Owlbot).")


