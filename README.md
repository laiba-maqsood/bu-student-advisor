# BU Student Advisor AI

An AI-powered student advisor chatbot for Bahria University built 
using Retrieval Augmented Generation (RAG).

## Live Demo
[Click here to try it](https://bu-student-advisor-2xzuad9q5jpktsyidv2zjv.streamlit.app/)

## What it does
- Answers student questions from official university documents
- Covers course outlines, policies, grading, attendance, and more
- Shows source citations for every answer
- Remembers conversation context

## Tech Stack
- Python, LangChain, FAISS, Google Gemini API
- HuggingFace Sentence Transformers
- Streamlit (frontend + deployment)

## Architecture
Documents → PyPDF/Docx loader → Text chunking → 
HuggingFace embeddings → FAISS vector store → 
Gemini 2.5 Flash → Streamlit UI
