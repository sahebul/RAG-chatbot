import hashlib
from datetime import datetime
from typing import List
from langchain_core.documents import Document

from .text_cleaner import clean_document
from .classifiers import classify_content, extract_section_title, detect_semantic_features

def get_doc_identity(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or hashlib.md5(
        doc.page_content.encode()
    ).hexdigest()

def get_doc_page(doc: Document):
    return doc.metadata.get("source_page", doc.metadata.get("page"))


def get_doc_end_page(doc: Document):
    return doc.metadata.get("end_page", get_doc_page(doc))


def deduplicate_documents(docs: List[Document]) -> List[Document]:

    seen = set()
    unique_docs = []

    for doc in docs:

        content_hash = hashlib.md5(
            doc.page_content.strip().encode()
        ).hexdigest()

        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)

    return unique_docs


# =========================================================
# DOCUMENT CLEANING PIPELINE
# =========================================================
def clean_documents(documents: List[Document]) -> List[Document]:

    cleaned_docs = []

    for doc in documents:

        raw_text = doc.page_content

        if doc.metadata.get("content_type") == "table":
            cleaned_docs.append(doc)
            continue

        cleaned_text = clean_document(raw_text)

        if not cleaned_text.strip():
            continue

        doc.metadata["raw_text"] = raw_text[:2000]

        doc.page_content = cleaned_text

        cleaned_docs.append(doc)

    return cleaned_docs



def add_chunk_metadata_v2(
    docs: List[Document],
    default_doc_type: str = "api_documentation"
) -> List[Document]:

    total_chunks = len(docs)

    for i, doc in enumerate(docs):

        section_title = extract_section_title(
            doc.page_content
        )

        original_content_type = doc.metadata.get("content_type")
        semantic_content_type = classify_content(
            doc.page_content
        )
        content_type = (
            original_content_type
            if original_content_type == "table"
            else semantic_content_type
        )

        semantic_features = detect_semantic_features(
            doc.page_content
        )

        chunk_hash = hashlib.md5(
            doc.page_content.encode()
        ).hexdigest()[:12]

        doc.metadata.update({

            # Source tracking (preserve if already set by loader)
            "source_file": doc.metadata.get("source_file", "unknown_source"),
            "doc_type": doc.metadata.get("doc_type", default_doc_type),

            # Chunk tracking
            "chunk_index": i,
            "total_chunks": total_chunks,
            "chunk_id": chunk_hash,

            # Section awareness
            "section_title": section_title,

            # Classification
            "content_type": content_type,
            "semantic_content_type": semantic_content_type,

            # Debugging / observability
            "char_count": len(doc.page_content),
            "word_count": len(doc.page_content.split()),

            # Timestamp
            "ingestion_timestamp": datetime.now().isoformat(),

            # Semantic features
            **semantic_features
        })

        # Preserve source page if available
        if "page" in doc.metadata:
            doc.metadata["source_page"] = doc.metadata["page"]

    return docs

def build_ingestion_signature(docs: List[Document]) -> str:
    signature_parts = []

    for doc in docs:
        metadata = doc.metadata
        signature_parts.append("|".join([
            str(metadata.get("page", "")),
            str(metadata.get("end_page", "")),
            str(metadata.get("content_type", "")),
            str(metadata.get("table_rows", "")),
            doc.page_content,
        ]))

    return hashlib.md5("\n---DOC---\n".join(signature_parts).encode()).hexdigest()
