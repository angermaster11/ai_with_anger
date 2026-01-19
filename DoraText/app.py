import streamlit as st
from functions import generate_text, generate_summary, generate_entities
from pypdf import PdfReader

st.title("DoraText: Your AI Text Companion")
st.write("Welcome to DoraText! This application leverages advanced AI models to assist you with")

tab1 , tab2, tab3 = st.tabs(["Search","Summarize","NER"])


with tab1:
    st.header("🔍 Search")
    st.write("Search and explore text-based content quickly and efficiently.")
    text = st.text_area("Enter your text", placeholder="Type something...")
    if st.button("Generate Text"):
        if text.strip()=="":
            st.warning("Please enter some text first.")
        else:
            st.subheader("Output")
            data = generate_text(text)
            st.info(data)

with tab2:
    st.header("📝 Summarize")
    st.write("Generate concise summaries from long documents using AI.")

    file_uploader = st.file_uploader("Upload a PDF file", type=["pdf"])
    if file_uploader is not None:
        if st.button("Summarize"):
            st.subheader("Summary")
            reader = PdfReader(file_uploader)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            res = generate_summary(text,40,100)
            st.info(res)

with tab3:
    st.header("🏷️ Named Entity Recognition (NER)")
    st.write("Identify people, places, organizations, and other entities from text.")
    text = st.text_area("Enter your text...", placeholder="Type something...")
    if st.button("Generate NER"):
        if text.strip()=="":
            st.warning("Please enter some text first.")
        else:
            st.subheader("Output")
            data = generate_entities(text)
            st.info(data)