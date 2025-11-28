import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from collections import defaultdict
import math


# ...existing code...
def ensure_nltk_data():
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        # try a quiet download (no GUI); may still fail in restricted environments
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


ensure_nltk_data()

# --- NLP summarization functions ---


def preprocess_text(text):
    # Sentence tokenization: prefer NLTK punkt, but fall back to a simple regex splitter
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # fallback: split on sentence-ending punctuation
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    # Stopwords: if NLTK stopwords are missing, use an empty set
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    # Word tokenization: prefer NLTK word_tokenize, fallback to simple splitting
    words = []
    for sentence in sentences:
        try:
            tokenized = word_tokenize(sentence.lower())
        except LookupError:
            tokenized = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
        words.append(tokenized)
    filtered_words = [
        [w for w in word_list if w.isalpha() and w not in stop_words]
        for word_list in words
    ]
    return sentences, filtered_words


def compute_tf(filtered_words):
    tf_scores = []
    for word_list in filtered_words:
        freq = defaultdict(int)
        for w in word_list:
            freq[w] += 1
        tf_scores.append(freq)
    return tf_scores


def compute_idf(tf_scores):
    idf_freq = defaultdict(int)
    N = len(tf_scores)
    for tf in tf_scores:
        for word in tf.keys():
            idf_freq[word] += 1
    for word in idf_freq:
        idf_freq[word] = math.log(N / idf_freq[word]) if idf_freq[word] != 0 else 0
    return idf_freq


def score_sentences(sentences, tf_scores, idf_scores):
    sentence_scores = []
    for i, tf in enumerate(tf_scores):
        score = 0
        for word, freq in tf.items():
            score += freq * idf_scores.get(word, 0)
        sentence_scores.append((sentences[i], score))
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    return sentence_scores


def summarize(text, num_sentences=3):
    sentences, filtered_words = preprocess_text(text)
    if len(sentences) == 0:
        return "No valid sentences found to summarize."
    tf_scores = compute_tf(filtered_words)
    idf_scores = compute_idf(tf_scores)
    scored_sentences = score_sentences(sentences, tf_scores, idf_scores)
    # Limit number of sentences to total sentences if fewer
    num_sentences = min(num_sentences, len(scored_sentences))
    summary = " ".join([sent[0] for sent in scored_sentences[:num_sentences]])
    return summary


def summarize_points(text, num_points=5, truncate=240):
    """Return a list of (sentence, score, relevance_percent) for top sentences.

    - `num_points` limits number of points returned.
    - `truncate` shortens very long sentences for display.
    """
    sentences, filtered_words = preprocess_text(text)
    if len(sentences) == 0:
        return []

    tf_scores = compute_tf(filtered_words)
    idf_scores = compute_idf(tf_scores)
    scored_sentences = score_sentences(sentences, tf_scores, idf_scores)
    if not scored_sentences:
        return []

    num_points = min(num_points, len(scored_sentences))
    max_score = scored_sentences[0][1] if scored_sentences[0][1] > 0 else 1.0
    points = []
    for sent, score in scored_sentences[:num_points]:
        display_text = (
            sent if len(sent) <= truncate else sent[:truncate].rstrip() + "..."
        )
        relevance = (score / max_score) * 100.0
        points.append((display_text, score, relevance))

    return points


# --- Streamlit UI ---

st.set_page_config(page_title="Notes Summarizer", page_icon="📝", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #2E4053;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 35px;
        font-weight: 700;
        text-align: center;
    }
    .subtitle {
        color: #34495E;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title">Notes Summarizer Using NLP</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">The Notes Summarizer using NLP project focuses on automatically converting long text documents or notes into short, meaningful summaries that are easy to read and understand. It uses Natural Language Processing techniques to clean the input text, identify the most important sentences, and generate a concise summary without losing the main idea of the original content. This helps students, researchers, and professionals save time and quickly grasp key points from large volumes of information</div>',
    unsafe_allow_html=True,
)

with st.container():
    # Ensure NLTK data is available before any summarization
    ensure_nltk_data()
    input_method = st.radio(
        "Choose input method:",
        ("Paste text", "Upload text file (.txt)", "Upload PDF file"),
    )

    input_text = ""
    if input_method == "Paste text":
        input_text = st.text_area("Enter or paste your notes here:", height=250)
    elif input_method == "Upload text file (.txt)":
        uploaded_txt = st.file_uploader("Upload your .txt file", type=["txt"])
        if uploaded_txt:
            input_bytes = uploaded_txt.read()
            try:
                input_text = input_bytes.decode("utf-8")
            except UnicodeDecodeError:
                input_text = input_bytes.decode("latin1")
    elif input_method == "Upload PDF file":
        uploaded_pdf = st.file_uploader("Upload your PDF file", type=["pdf"])
        if uploaded_pdf:
            try:
                # Read PDF with PyPDF2
                import PyPDF2

                pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
                text_list = []
                for page in pdf_reader.pages:
                    text_list.append(page.extract_text())
                input_text = "\n".join(text_list)
            except Exception:
                st.error("Error reading PDF file. Please upload a valid PDF.")

    if input_text and len(input_text.strip()) > 0:
        summary_length = st.slider(
            "Select number of sentences for summary:",
            min_value=1,
            max_value=10,
            value=3,
        )
        style = st.radio("Choose output format:", ("Paragraph", "Points (numbered)"))

        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                if style == "Paragraph":
                    summary = summarize(input_text, summary_length)
                    st.success("Summary generated successfully!")
                    st.markdown("### Summary:")
                    st.write(summary)

                    # Copy to clipboard button (works in some browsers)
                    st.code(summary, language=None)
                else:
                    points = summarize_points(input_text, num_points=summary_length)
                    if not points:
                        st.info("No summary points available.")
                    else:
                        st.success("Points-style summary generated successfully!")
                        st.markdown("### Summary (Points):")
                        for i, (text_pt, score, relevance) in enumerate(
                            points, start=1
                        ):
                            st.markdown(f"**{i}.** {text_pt}")
                            st.caption(
                                f"Score: {score:.2f} — Relevance: {relevance:.0f}%"
                            )
    else:
        st.info("Please provide some input text or upload a file to summarize.")

st.markdown("---")
st.markdown(
    "<center>Developed by M.Mallesh | JB Institute of Engineering and Technology</center>",
    unsafe_allow_html=True,
)
