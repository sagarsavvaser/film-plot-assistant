import streamlit as st
from rag import load_docs, chunk_docs, build_index, retrieve, answer

st.set_page_config(page_title="Film Plot Assistant")

@st.cache_resource
def setup():
    chunks = chunk_docs(load_docs("Films"))
    return chunks, build_index(chunks)

chunks, index = setup()

st.title("Film Plot Assistant")
st.caption(
    "Ask questions about your movie collection. "
    "Answers are based on your film text files using RAG."
)

q = st.text_input(
    "Your question",
    placeholder="Who directed Skyfall?"
)

if q:
    hits = retrieve(q, index, chunks)

    with st.spinner("Thinking..."):
        result = answer(
            q,
            hits,
            st.secrets["GEMINI_API_KEY"],
            st.secrets["GEMINI_MODEL"]
        )
        st.write(result)

    with st.expander("Sources"):
        if hits:
            for title in dict.fromkeys(h["title"] for h in hits):
                st.write("•", title)
        else:
            st.write("No relevant sources found.")
