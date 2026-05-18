import os
from langchain_community.vectorstores import Chroma
from config import DB_PATH, COLLECTION_NAME

def get_or_create_vectorstore(docs, embeddings, signature, total_files):
    collection_metadata = {
        "ingestion_signature": signature,
        "total_files": total_files
    }
    
    if os.path.exists(DB_PATH):
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            collection_metadata=collection_metadata
        )
        existing_count = vectorstore._collection.count()
        existing_metadata = vectorstore._collection.metadata or {}
        existing_signature = existing_metadata.get("ingestion_signature")
        
        if existing_count == len(docs) and existing_signature == signature:
            print(f"Loaded existing vector DB with {existing_count} chunks")
            return vectorstore
            
        if existing_count > 0:
            print(f"Rebuilding vector DB: stored chunks do not match current extraction ({existing_count} vs {len(docs)})")
            vectorstore.delete_collection()
            
    print("Creating new vector DB...")
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        collection_metadata=collection_metadata
    )
