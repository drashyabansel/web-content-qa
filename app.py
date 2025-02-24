# Import Libraries
# Import Streamlit for dashboard
import streamlit as st
# Request to get the HTML Content of the Page
import requests
# Extract Useful imformation fromthe page
from bs4 import BeautifulSoup
# To Call model from HuggingFace.
from transformers import pipeline

# Initialize the QA pipeline with a better model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def extract_text_from_url(url):
    """Extracts and cleans text content from a given URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # Prevent blocking by some websites
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        # Apply beatiful soup for easy reading
        soup = BeautifulSoup(response.text, "html.parser")
        divs = soup.find_all("div")
        extracted_text = " ".join([div.get_text() for div in divs if div.get_text()])
        return extracted_text if extracted_text else "No relevant text found on the page."
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {str(e)}"

st.title("Web Content Q&A Tool")

# User input for URLs
urls = st.text_area("Enter URLs (one per line):")
urls = urls.split("\n") if urls else []

try:
    if st.button("Ingest Content"):
        st.session_state["content"] = ""
        for url in urls:
            if url.strip():
                text_content = extract_text_from_url(url.strip())
                st.session_state["content"] += text_content + "\n\n"
        st.success("Content Ingested Successfully!")
except Exception as e:
    st.warning(f"Got an Exception while working\n{e}")

# User input for questions
question = st.text_input("Ask a question based on the ingested content:")

if st.button("Get Answer"):
    if "content" in st.session_state and st.session_state["content"].strip():
        context = st.session_state["content"]
        result = qa_pipeline(question=question, context=context)
        st.write("**Answer:**", result["answer"])
        st.write("**Score:**", result["score"])
    else:
        st.warning("Please ingest content first before asking questions.")
