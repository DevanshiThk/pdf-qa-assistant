import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import gradio as gr


load_dotenv(r'C:\Users\grand\ml-projects\phase4-rag-app\.env')
api_key = os.getenv("GROQ_API_KEY")


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
        return "Please upload a PDF first!"
    if not question:
        return "Please ask a question!"
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

with gr.Blocks(title="Space Science Q&A", theme=theme) as app:
    gr.Markdown("# Ask Questions About Any PDF")
    gr.Markdown("Upload a scientific paper, your college notes, or any document and ask questions about it!")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        question_input = gr.Textbox(label="Ask a question",
                                     placeholder="Type your question here")
    
    submit_btn = gr.Button("Ask!", variant="primary")
    answer_output = gr.Textbox(label="Answer", lines=5)
    
    submit_btn.click(
        fn=process_pdf_and_ask,
        inputs=[pdf_input, question_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    app.launch()