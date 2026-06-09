import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import gradio as gr

# Load API key
load_dotenv(r'C:\Users\grand\ml-projects\phase4-rag-app\.env')
api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM and embeddings
llm = ChatGroq(
    api_key=api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_chunks = retriever.invoke(question)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    prompt = f"""Answer the question based only on the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
    response = llm.invoke(prompt)
    return response.content

def process_pdf_and_ask(pdf_file, question):
    if pdf_file is None:
        return "⚠️ Please upload a PDF first!"
    if not question:
        return "⚠️ Please ask a question!"
    try:
        vectorstore = load_and_process_pdf(pdf_file.name)
        answer = ask_question(vectorstore, question)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

theme = gr.themes.Default(
    primary_hue="purple",
    secondary_hue="violet",
)

with gr.Blocks(title="PDF Q&A Assistant", theme=theme) as app:
    gr.Markdown("""
    # 📄 PDF Q&A Assistant
    ### Powered by LLaMA 3.1 + RAG
    Upload any PDF document and ask questions about its contents.
    The AI answers **only from the document** — no hallucinations!
    """)
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="📂 Upload PDF",
                file_types=[".pdf"]
            )
            gr.Markdown("*Supports any PDF — research papers, textbooks, reports, manuals*")
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="💬 Your Question",
                placeholder="e.g. What is the main finding of this document?",
                lines=3
            )
            submit_btn = gr.Button("Ask! 🚀", variant="primary", size="lg")
    answer_output = gr.Textbox(
        label="🤖 Answer",
        lines=6,
        placeholder="Answer will appear here..."
    )
    gr.Markdown("""
    ---
    ### 💡 Example Questions to Try
    - *What is the main topic of this document?*
    - *What are the key findings or conclusions?*
    - *What methodology or approach was used?*
    - *Summarise the document in a few sentences.*
    - *What are the limitations mentioned?*
    """)
    gr.Markdown("""
    ---
    *Built with LangChain · Groq · FAISS · Gradio* |
    *Model: LLaMA 3.1 8B* |
    *Embeddings: sentence-transformers/all-MiniLM-L6-v2*
    """)
    submit_btn.click(
        fn=process_pdf_and_ask,
        inputs=[pdf_input, question_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    app.launch()s