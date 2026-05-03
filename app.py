from typing import Optional
import streamlit as st
from transformers import pipeline
import requests
import PyPDF2
from bs4 import BeautifulSoup

# Function: Fetch text from URL
def fetch_text_from_url(url: str) -> str:
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    return "\n\n".join(paragraphs)

# Function: Clean + truncate text
def clean_and_truncate_text(text: str, max_words: int = 800) -> str:
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)

# Summarizer Class
class Summarizer:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.pipeline = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40, do_sample: bool = False) -> str:
        result = self.pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample
        )
        return result[0].get('summary_text', "")

# Streamlit UI
def main():
    st.set_page_config(page_title="Text Summarizer App", layout="wide")
    st.title("Text Summarizer App - Python + HuggingFace")
    st.markdown("Summarize text, files, PDFs, and web pages")

    left, right = st.columns([1, 2])

    with left:
        input_mode = st.radio("Input Type:", ("Text", "File", "URL", "PDF"))

        model_choice = st.selectbox(
            "Choose Model:",
            ("sshleifer/distilbart-cnn-12-6", "facebook/bart-large-cnn")
        )

        min_length = st.number_input("Min tokens", 5, 100, 40)
        max_length = st.number_input("Max tokens", 10, 300, 150)
        do_sample = st.checkbox("Use Sampling", False)

        raw_text: Optional[str] = None
        uploaded_file = None
        url_input: Optional[str] = None

        # Inputs
        if input_mode == "Text":
            raw_text = st.text_area("Enter text", height=250)

        elif input_mode == "File":
            uploaded_file = st.file_uploader("Upload TXT or MD file", type=["txt", "md"])

        elif input_mode == "URL":
            url_input = st.text_input("Enter URL")

        elif input_mode == "PDF":
            uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

        summarize_button = st.button("Summarize")

    with right:
        st.subheader("Original Text")
        source_container = st.empty()

        st.subheader("Summary Output")
        summary_container = st.empty()

        if summarize_button:

            user_input_text = ""

            # TEXT INPUT
            if input_mode == "Text" and raw_text:
                user_input_text = raw_text

            # FILE INPUT
            elif input_mode == "File" and uploaded_file:
                try:
                    user_input_text = uploaded_file.read().decode("utf-8")
                except:
                    user_input_text = uploaded_file.read().decode("latin-1")

            # URL INPUT
            elif input_mode == "URL" and url_input:
                try:
                    user_input_text = fetch_text_from_url(url_input)
                except Exception as e:
                    st.error(f"URL Error: {e}")
                    return

            # PDF INPUT (FIXED)
            elif input_mode == "PDF" and uploaded_file:
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text = ""

                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                    user_input_text = text

                except Exception as e:
                    st.error(f"PDF Error: {e}")
                    return

            # VALIDATION
            if not user_input_text.strip():
                st.warning("Please provide valid input.")
                return

            if min_length >= max_length:
                st.error("Min length must be less than max length.")
                return

            # CLEAN TEXT
            # CLEAN TEXT (limit input size for both models)
            user_input_text = clean_and_truncate_text(user_input_text, 800)

            source_container.code(user_input_text[:8000])

            # MODEL
            summarizer = Summarizer(model_name=model_choice)

            with st.spinner("Generating summary..."):
                try:
                    summary_text = summarizer.summarize(
                        user_input_text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=do_sample
                    )
                except Exception as e:
                    st.error(f"Summarization Error: {e}")
                    return

            if not summary_text.strip():
                st.warning("Empty summary generated.")
                return

            summary_container.text_area("Summary", summary_text, height=250)


if __name__ == "__main__":
    main()