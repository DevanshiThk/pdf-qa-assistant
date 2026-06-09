#  PDF Q&A Assistant

An AI-powered application that lets you upload any PDF and ask questions 
about its contents using Retrieval Augmented Generation (RAG).

## Live Demo
[Try it on Hugging Face Spaces](#) *(coming soon)*

## How It Works
User uploads PDF
↓
Document split into 500-word chunks
↓
Chunks converted to embeddings (sentence-transformers)
↓
Stored in FAISS vector database
↓
User asks a question
↓
Most relevant chunks retrieved
↓
LLaMA 3.1 answers using only those chunks


## Features
- Upload any PDF — research papers, textbooks, reports, manuals
- Answers grounded strictly in the document — no hallucinations
- Admits when it doesn't know rather than making things up
- Clean purple UI built with Gradio
- Fast responses powered by Groq inference

## Tech Stack
| Component | Technology |
|-----------|-----------|
| LLM | LLaMA 3.1 8B via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Framework | LangChain |
| UI | Gradio |

## Example Questions to Try
- What is the main finding of this document?
- What methodology was used?
- What are the key conclusions?
- Summarise the document in a few sentences.
- What are the limitations mentioned?

## Tested On
- Akiyama et al. 2019 — First Image of a Black Hole (Event Horizon Telescope)

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to a `.env` file: `GROQ_API_KEY=your_key_here`
4. Run: `python app.py`

## Libraries Used
LangChain, LangChain-Groq, LangChain-HuggingFace, FAISS, Gradio, 
Sentence-Transformers, PyPDF, Python-dotenv