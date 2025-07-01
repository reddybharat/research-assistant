import streamlit as st
import os
import shutil

from core.helper.ingest_data import IngestData
from research_assistant.utils.llm import LLM

DATA_PATH = ".\src\core\data\\user_files"

st.set_page_config(page_title="Research Assistant", layout="centered")

st.title("Research Assistant")

# Step 1: File uploader
st.header("1. Upload PDF files")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files", 
    type=["pdf"], 
    accept_multiple_files=True
)

saved_file_paths = []
if uploaded_files:
    st.info("Saving uploaded files...")
    # Ensure data directory exists and is empty
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
        st.info("Existing data directory cleared.")
    os.makedirs(DATA_PATH, exist_ok=True)
    # Save uploaded files
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_file_paths.append(file_path)
    st.success(f"{len(uploaded_files)} file(s) uploaded.")

# Step 2: Ingest button
if st.button("Continue (Ingest Data)", disabled=not uploaded_files):
    with st.spinner("Ingesting data into vector database..."):
        st.info("Starting ingestion process...")
        ingestor = IngestData()
        st.write(f"Ingesting files: {saved_file_paths}")
        ingestor.ingest(saved_file_paths)
        st.success("Data ingested successfully! You can now ask questions.")
        st.write("Ingestion process complete.")

    # Set a session state flag to show the query UI
    st.session_state["ingested"] = True

# Step 3: Query box (only after ingestion)
if st.session_state.get("ingested", False):
    st.header("2. Ask a question")
    user_query = st.text_input("Enter your question:")
    if st.button("Get Response", disabled=not user_query):
        with st.spinner("Getting response from LLM..."):
            llm = LLM()
            response = llm.run_llm(user_query)
            st.write("**Answer:**")
            st.write(response.get("answer", "No answer found."))
