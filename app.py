from dotenv import load_dotenv
import os
import sys
import re
import hashlib
import pandas as pd
import pdfplumber
from datetime import datetime
from typing import List


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from openai import OpenAI

from rank_bm25 import BM25Okapi


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENCODE_API_KEY"),
    base_url="https://opencode.ai/zen/v1"
)

# new cleaning 
# =========================================================
# OCR / PDF NOISE CLEANING
# =========================================================

def remove_ocr_noise(text: str) -> str:
    """
    Remove obvious OCR garbage and meaningless long random strings.
    """

    # Remove extremely long random alphanumeric garbage
    text = re.sub(r'\b[A-Za-z0-9]{35,}\b', ' ', text)

    # Remove repeated underscores/dashes
    text = re.sub(r'[_\-]{4,}', ' ', text)

    # Remove broken page artifacts
    text = re.sub(r'\bPage\s+\d+\b', ' ', text, flags=re.IGNORECASE)

    # Remove common footer/header noise
    text = re.sub(
        r'Document Version\s*\d+(\.\d+)?',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    return text


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    """
    Normalize spacing while preserving
    API symbols, JSON structure, and technical syntax.
    """

    # Normalize newlines
    text = text.replace('\r', '\n')

    # Remove excessive empty lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # Trim spaces around newlines
    text = re.sub(r' *\n *', '\n', text)

    return text.strip()


# =========================================================
# MAIN CLEANING FUNCTION
# =========================================================

def clean_document(text: str) -> str:
    """
    Clean document while preserving semantic structure.
    """

    # Remove OCR garbage
    text = remove_ocr_noise(text)

    # Normalize formatting
    text = normalize_text(text)

    return text


# =========================================================
# CONTENT CLASSIFICATION
# =========================================================

def classify_content(text: str) -> str:
    """
    Detect technical/API-related content types.
    """

    text_lower = text.lower()

    # API endpoint detection
    if (
        "/api/" in text_lower or
        "endpoint" in text_lower or
        "http://" in text_lower or
        "https://" in text_lower
    ):
        return "api_endpoint"

    # JSON examples
    if (
        "{" in text and
        "}" in text and
        ":" in text
    ):
        return "json_example"

    # Parameter tables
    if any(
        kw in text_lower
        for kw in [
            "parameter",
            "sample value",
            "description",
            "type"
        ]
    ):
        return "parameter_definition"

    # Authentication docs
    if any(
        kw in text_lower
        for kw in [
            "token",
            "authorization",
            "bearer",
            "api key"
        ]
    ):
        return "authentication"

    # Structural sections
    if any(
        kw in text_lower
        for kw in [
            "chapter",
            "section",
            "introduction"
        ]
    ):
        return "structural"

    return "prose"


# =========================================================
# SECTION TITLE EXTRACTION
# =========================================================

def extract_section_title(text: str) -> str:
    """
    Try extracting heading/section title from chunk.
    """

    lines = text.split('\n')

    for line in lines[:10]:

        cleaned = line.strip()

        # Skip very short lines
        if len(cleaned) < 5:
            continue

        # Skip overly long lines
        if len(cleaned) > 120:
            continue

        # Detect likely heading
        if (
            cleaned.lower().startswith("section") or
            cleaned.lower().startswith("chapter") or
            "api" in cleaned.lower() or
            cleaned.endswith(":")
        ):
            return cleaned

    return "Unknown Section"


# =========================================================
# SEMANTIC FEATURE DETECTION
# =========================================================

def detect_semantic_features(text: str) -> dict:

    text_lower = text.lower()

    return {
        "contains_json": (
            "{" in text and
            "}" in text
        ),

        "contains_url": (
            "http://" in text_lower or
            "https://" in text_lower
        ),

        "contains_api": (
            "/api/" in text_lower or
            "api" in text_lower
        ),

        "contains_table_like_data": (
            "sample value" in text_lower or
            "description" in text_lower
        )
    }


# =========================================================
# DEDUPLICATION
# =========================================================

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


# =========================================================
# METADATA ENRICHMENT
# =========================================================

def add_chunk_metadata(
    docs: List[Document],
    source_file: str,
    doc_type: str = "api_documentation"
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

            # Source tracking
            "source_file": source_file,
            "doc_type": doc_type,

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


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_./:-]+", text.lower())


def normalize_table_cell(value) -> str:
    return str(value or "").strip()


def normalize_table_row(row) -> List[str]:
    return [normalize_table_cell(value).lower() for value in row]


def is_blank_table(table) -> bool:
    return all(not normalize_table_cell(cell) for row in table for cell in row)


def should_merge_table(previous_df, previous_metadata, current_df, page_num) -> bool:
    if previous_df is None or previous_metadata is None:
        return False

    # Must have the same number of columns to be merged
    if len(current_df.columns) != len(previous_df.columns):
        return False

    # Must be on the same page or the immediately following page
    if page_num > previous_metadata["end_page"] + 1:
        return False

    previous_header = normalize_table_row(previous_df.iloc[0].tolist())
    current_header = normalize_table_row(current_df.iloc[0].tolist())

    # 1. If the header is exactly repeated, it's a continuation of the same table
    if previous_header == current_header:
        return True

    # 2. If it's on the exact same page, pdfplumber likely split one logical table
    if page_num == previous_metadata["end_page"]:
        return True

    # 3. If it's on the next page and columns match perfectly, it's highly likely 
    # to be a continuation of the same table (even without a repeated header).
    # If it happens to be a completely new table with the exact same column count, 
    # merging them into one Markdown block is still harmless for LLM comprehension.
    return True


def drop_duplicate_header(previous_df, current_df):
    if current_df.empty:
        return current_df

    previous_header = normalize_table_row(previous_df.iloc[0].tolist())
    current_header = normalize_table_row(current_df.iloc[0].tolist())

    if previous_header == current_header:
        return current_df.iloc[1:].reset_index(drop=True)

    return current_df


def get_doc_identity(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or hashlib.md5(
        doc.page_content.encode()
    ).hexdigest()


def get_doc_page(doc: Document):
    return doc.metadata.get("source_page", doc.metadata.get("page"))


def get_doc_end_page(doc: Document):
    return doc.metadata.get("end_page", get_doc_page(doc))


def expand_with_related_tables(results: List[Document], all_docs: List[Document]) -> List[Document]:
    table_docs = [
        doc
        for doc in all_docs
        if doc.metadata.get("content_type") == "table"
    ]

    expanded = []
    seen = set()

    def add_doc(doc: Document):
        doc_id = get_doc_identity(doc)
        if doc_id not in seen:
            seen.add(doc_id)
            expanded.append(doc)

    for result in results:
        add_doc(result)

        result_page = get_doc_page(result)
        if result_page is None:
            continue

        for table_doc in table_docs:
            start_page = get_doc_page(table_doc)
            end_page = get_doc_end_page(table_doc)

            if start_page is None or end_page is None:
                continue

            if start_page <= result_page <= end_page:
                add_doc(table_doc)

    return expanded


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


# end of new cleaning
# ========== Pipeline ==========
DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

# loader = PyPDFLoader("documents/sample.pdf")
# documents = loader.load()
# =============Extract Tables Separately if any======

documents = []

# ==========================================
# TABLE MERGE STATE
# ==========================================

previous_table_df = None
previous_table_metadata = None

with pdfplumber.open("documents/sample.pdf") as pdf:

    for page_num, page in enumerate(pdf.pages):

        # ==========================================
        # NORMAL TEXT
        # ==========================================

        text = page.extract_text()

        if text:

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_num,
                        "is_table": False,
                        "content_type": "text"
                    }
                )
            )

        # ==========================================
        # TABLE EXTRACTION
        # ==========================================

        tables = page.extract_tables()

        for table_index, table in enumerate(tables):

            # Skip empty tables
            if not table or is_blank_table(table):
                continue

            df = pd.DataFrame(table).fillna("")

            # ==========================================
            # FIRST TABLE
            # ==========================================

            if previous_table_df is None:

                previous_table_df = df

                previous_table_metadata = {
                    "page": page_num,
                    "end_page": page_num,
                    "is_table": True,
                    "content_type": "table",
                    "table_rows": len(previous_table_df),
                    "table_columns": len(previous_table_df.columns),
                    "table_index": table_index
                }

                continue

            # ==========================================
            # CHECK TABLE CONTINUATION
            # ==========================================

            # ==========================================
            # MERGE TABLES
            # ==========================================

            should_merge = should_merge_table(
                previous_table_df,
                previous_table_metadata,
                df,
                page_num
            )

            if should_merge:

                df = drop_duplicate_header(previous_table_df, df)

                previous_table_df = pd.concat(
                    [previous_table_df, df],
                    ignore_index=True
                )

                previous_table_metadata["table_rows"] = len(previous_table_df)
                previous_table_metadata["table_columns"] = len(previous_table_df.columns)
                previous_table_metadata["end_page"] = page_num

            else:

                # ==========================================
                # SAVE OLD TABLE
                # ==========================================

                table_text = previous_table_df.to_markdown(index=False)

                documents.append(
                    Document(
                        page_content=table_text,
                        metadata=previous_table_metadata
                    )
                )

                # Start new table
                previous_table_df = df

                previous_table_metadata = {
                    "page": page_num,
                    "end_page": page_num,
                    "is_table": True,
                    "content_type": "table",
                    "table_rows": len(previous_table_df),
                    "table_columns": len(previous_table_df.columns),
                    "table_index": table_index
                }

# ==========================================
# SAVE LAST TABLE
# ==========================================

if previous_table_df is not None:

    table_text = previous_table_df.to_markdown(index=False)

    documents.append(
        Document(
            page_content=table_text,
            metadata=previous_table_metadata
        )
    )

# for i, doc in enumerate(documents):
#     if doc.metadata.get("content_type") == "table":
#         print(f"\n=== Table Document {i} ===")
#         print(
#             f"Page: {doc.metadata.get('page')}, "
#             f"End page: {doc.metadata.get('end_page')}, "
#             f"Rows: {doc.metadata.get('table_rows')}"
#         )
#         print(doc.page_content)
#         print("---")
# sys.exit("Done")
documents = clean_documents(documents)
documents = deduplicate_documents(documents)

# Export tables to a markdown file
with open("extracted_tables.md", "w", encoding="utf-8") as f:
    for doc in documents:
        if doc.metadata.get("content_type") == "table":
            f.write(f"## Table (Pages {doc.metadata.get('page')} to {doc.metadata.get('end_page')})\n\n")
            f.write(doc.page_content)
            f.write("\n\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# docs = splitter.split_documents(documents)

final_docs = []

for doc in documents:

    # DO NOT SPLIT TABLES
    if doc.metadata.get("content_type") == "table":
        final_docs.append(doc)

    else:
        split_docs = splitter.split_documents([doc])
        final_docs.extend(split_docs)

docs = final_docs


docs = add_chunk_metadata(docs, "sample.pdf")
ingestion_signature = build_ingestion_signature(docs)
collection_metadata = {
    "source_file": "sample.pdf",
    "ingestion_signature": ingestion_signature,
}

bm25_corpus = [
    tokenize_for_bm25(doc.page_content)
    for doc in docs
]
bm25 = BM25Okapi(bm25_corpus)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

# Cross-encoder for reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Check if existing collection exists
if os.path.exists(DB_PATH):
    # Load existing vectorstore instead of recreating
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata=collection_metadata
    )
    existing_count = vectorstore._collection.count()
    existing_metadata = vectorstore._collection.metadata or {}
    existing_signature = existing_metadata.get("ingestion_signature")

    if existing_count == len(docs) and existing_signature == ingestion_signature:
        print(f"Loaded existing vector DB with {existing_count} chunks")
    elif existing_count > 0:
        print(
            f"Rebuilding vector DB: stored chunks/signature do not match "
            f"current extraction ({existing_count} stored, {len(docs)} current)"
        )
        vectorstore.delete_collection()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME,
            collection_metadata=collection_metadata
        )
    else:
        print("Empty DB, creating new...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME,
            collection_metadata=collection_metadata
        )
else:
    # Fresh creation
    print("Creating new vector DB...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        collection_metadata=collection_metadata
    )

# ========== Retrieval with Metadata ==========
print("Chatbot started. Type 'exit' to quit.\n")

while True:
    query = input("Ask question: ")
    if query.lower() == 'exit':
        print("Chatbot closed.")
        break

    # Hybrid retrieval: vector search + BM25 keyword search
    vector_results = vectorstore.similarity_search(
        query,
        k=20,
        filter={"source_file": "sample.pdf"}
    )

    tokenized_query = tokenize_for_bm25(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:20]
    bm25_results = [
        docs[i]
        for i in top_bm25_indices
        if bm25_scores[i] > 0
    ]

    seen_candidates = set()
    initial_results = []

    for doc in vector_results + bm25_results:
        candidate_id = get_doc_identity(doc)

        if candidate_id in seen_candidates:
            continue

        seen_candidates.add(candidate_id)
        initial_results.append(doc)

    initial_results = expand_with_related_tables(initial_results, docs)

    # Rerank using cross-encoder
    doc_texts = [doc.page_content for doc in initial_results]
    query_doc_pairs = [[query, doc] for doc in doc_texts]
    rerank_scores = cross_encoder.predict(query_doc_pairs)

    # Sort by rerank scores and get top 3
    scored_docs = list(zip(initial_results, rerank_scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    results = [doc for doc, score in scored_docs[:5]]

    # Expand context with full tables after reranking to ensure they are not truncated
    results = expand_with_related_tables(results, docs)

    # Save exact context to a markdown file as requested
    with open("retrieved_context.md", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"### Source: Page {r.metadata.get('source_page', '?')}\n\n")
            f.write(r.page_content)
            f.write("\n\n")

    context = "\n".join([
        f"[Source: Page {r.metadata.get('source_page', '?')}] {r.page_content}"
        for r in results
    ])

    prompt = f"""You are given document context.

Your task:
- Read the context carefully
- Ignore noisy or corrupted text
- Answer only using relevant information
- If answer is missing, say "Not found"

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}
"""

    print(f"PROMP: {prompt}")
    response = client.chat.completions.create(
        model="minimax-m2.5-free",
        messages=[
    {
        "role": "system",
        "content": """
You are a helpful and polite document question-answering assistant.

Rules:
1. For questions about the document, answer ONLY from the provided context.
2. If the user asks a question and the answer is not found in the context, politely say that you don't have that information in the provided document.
3. If the user sends a basic greeting (e.g., "hi", "hello", "how are you") or expresses gratitude (e.g., "thanks"), respond politely and naturally without complaining about missing context. Offer your assistance to answer questions about the document.
4. Be concise, accurate, and professional.
5. Prefer exact values and definitions from context.
6. Ignore corrupted or unrelated text in the context.
"""
    },
    {
        "role": "user",
        "content": prompt
    }
]
    )

    print("\nAI Answer:\n")
    print(response.choices[0].message.content)
