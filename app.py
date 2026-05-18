import os
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCS_DIR
# Imports from our modularized components
from src.loaders.pdf import load_pdf
from src.loaders.docx import load_docx
from src.loaders.txt import load_txt
from src.preprocessing.metadata import clean_documents, deduplicate_documents, add_chunk_metadata_v2, build_ingestion_signature
from src.indexer.bm25_index import build_bm25_index
from src.indexer.vector_db import get_or_create_vectorstore
from src.query_engine import chat_function_engine, get_query_engine_embeddings

def main():
    # 1. Ingestion Pipeline
    documents = []
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    files_list = os.listdir(DOCS_DIR)
    for filename in files_list:
        file_path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            print(f"Loading PDF: {filename}")
            documents.extend(load_pdf(file_path))
        elif filename.endswith(".docx"):
            print(f"Loading DOCX: {filename}")
            documents.extend(load_docx(file_path))
        elif filename.endswith(".txt"):
            print(f"Loading TXT: {filename}")
            documents.extend(load_txt(file_path))

    # Clean & Deduplicate
    documents = clean_documents(documents)
    documents = deduplicate_documents(documents)

    # Dump extracted tables for inspection
    with open("extracted_tables.md", "w", encoding="utf-8") as f:
        for doc in documents:
            if doc.metadata.get("content_type") == "table":
                f.write(f"## Table (Pages {doc.metadata.get('page')} to {doc.metadata.get('end_page')})\n\n")
                f.write(doc.page_content)
                f.write("\n\n")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    final_docs = []
    for doc in documents:
        if doc.metadata.get("content_type") == "table":
            final_docs.append(doc)
        else:
            final_docs.extend(splitter.split_documents([doc]))

    docs = add_chunk_metadata_v2(final_docs)
    signature = build_ingestion_signature(docs)

    # 2. Build or Load Indexes
    bm25 = build_bm25_index(docs)
    embeddings = get_query_engine_embeddings()
    vectorstore = get_or_create_vectorstore(docs, embeddings, signature, len(files_list))

    # 3. Setup Gradio Chat Interface
    def chat_interface_fn(message, history):
        # We delegate the entire chat engine logic to query_engine
        yield from chat_function_engine(message, history, vectorstore, bm25, docs)

    print("Starting Gradio Interface...")
    demo = gr.ChatInterface(
        fn=chat_interface_fn,
        title="RTPS / Sewasetu Integration Chatbot",
        description="Ask questions about the API documentation and integration details.",
    )
    
    demo.launch()

if __name__ == "__main__":
    main()
