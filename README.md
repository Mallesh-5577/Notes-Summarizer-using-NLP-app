# Notes-Summarizer-using-NLP-app

A Streamlit web application that automatically summarizes text documents or notes using Natural Language Processing (NLP) techniques. The app allows users to paste text, upload .txt files, or upload PDF files and generates concise summaries in either paragraph or points format.

## Description

In academic, research, and professional settings, people often need to quickly extract key points from lengthy documents or notes. Manually reading and summarizing large texts is time-consuming and inefficient. This project automates the process of summarizing text documents using NLP, making it easier for users to grasp essential information quickly.

## Features

- Summarize text using TF-IDF based sentence scoring
- Input options: Paste text, upload .txt files, or upload PDF files
- Output formats: Paragraph or numbered points
- Adjustable summary length
- Relevance scoring for each sentence in points format

## Methodology

- **Text Preprocessing:** The input text is cleaned, tokenized into sentences and words, and filtered to remove stopwords and non-alphabetic tokens.
- **TF-IDF Scoring:** Each sentence is scored based on Term Frequency-Inverse Document Frequency (TF-IDF), which helps identify the most important sentences.
- **Summary Generation:** The highest-scoring sentences are selected to form the summary, with the option to output as a paragraph or numbered points.

## Technologies Used

- Streamlit (for the web interface)
- NLTK (for NLP preprocessing)
- PyPDF2 (for PDF text extraction)
- Python (core language)

## Usage

- Start the app with `streamlit run app.py`
- Choose your input method: paste text, upload a .txt file, or upload a PDF file
- Select the summary length and output format
- Click "Generate Summary" to see the results

## Use Cases

- Students summarizing lecture notes or research papers
- Professionals summarizing meeting minutes or reports
- Researchers quickly extracting key points from lengthy articles

## Future Enhancements

- Support for additional file formats (e.g., DOCX)
- Integration with advanced transformer-based models for abstractive summarization
- Multi-language support for summarizing text in different languages

## Credits

Developed by M. Mallesh  
JB Institute of Engineering and Technology

---

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

