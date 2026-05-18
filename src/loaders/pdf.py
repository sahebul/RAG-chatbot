import os
import pdfplumber
import pandas as pd
from typing import List
from langchain_core.documents import Document
from src.preprocessing.metadata import get_doc_identity, get_doc_page, get_doc_end_page

def normalize_table_cell(value) -> str:
    return str(value or "").strip()

def normalize_table_row(row) -> List[str]:
    return [normalize_table_cell(value).lower() for value in row]

def is_blank_table(table) -> bool:
    return all(not normalize_table_cell(cell) for row in table for cell in row)

def should_merge_table(previous_df, previous_metadata, current_df, page_num) -> bool:
    if previous_df is None or previous_metadata is None:
        return False
    if len(current_df.columns) != len(previous_df.columns):
        return False
    if page_num > previous_metadata["end_page"] + 1:
        return False
    previous_header = normalize_table_row(previous_df.iloc[0].tolist())
    current_header = normalize_table_row(current_df.iloc[0].tolist())
    
    if previous_header == current_header:
        return True
    if page_num == previous_metadata["end_page"]:
        return True
    return True

def drop_duplicate_header(previous_df, current_df):
    if current_df.empty:
        return current_df
    previous_header = normalize_table_row(previous_df.iloc[0].tolist())
    current_header = normalize_table_row(current_df.iloc[0].tolist())
    if previous_header == current_header:
        return current_df.iloc[1:].reset_index(drop=True)
    return current_df

def expand_with_related_tables(results: List[Document], all_docs: List[Document]) -> List[Document]:
    table_docs = [doc for doc in all_docs if doc.metadata.get("content_type") == "table"]
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

def load_pdf(file_path: str) -> List[Document]:
    docs = []
    previous_table_df = None
    previous_table_metadata = None
    file_name = os.path.basename(file_path)

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "page": page_num,
                        "is_table": False,
                        "content_type": "text",
                        "source_file": file_name
                    }
                ))
            tables = page.extract_tables()
            for table_index, table in enumerate(tables):
                if not table or is_blank_table(table):
                    continue
                df = pd.DataFrame(table).fillna("")
                
                if previous_table_df is None:
                    previous_table_df = df
                    previous_table_metadata = {
                        "page": page_num,
                        "end_page": page_num,
                        "is_table": True,
                        "content_type": "table",
                        "table_rows": len(previous_table_df),
                        "table_columns": len(previous_table_df.columns),
                        "table_index": table_index,
                        "source_file": file_name
                    }
                    continue

                if should_merge_table(previous_table_df, previous_table_metadata, df, page_num):
                    df = drop_duplicate_header(previous_table_df, df)
                    previous_table_df = pd.concat([previous_table_df, df], ignore_index=True)
                    previous_table_metadata["table_rows"] = len(previous_table_df)
                    previous_table_metadata["table_columns"] = len(previous_table_df.columns)
                    previous_table_metadata["end_page"] = page_num
                else:
                    docs.append(Document(
                        page_content=previous_table_df.to_markdown(index=False),
                        metadata=previous_table_metadata
                    ))
                    previous_table_df = df
                    previous_table_metadata = {
                        "page": page_num,
                        "end_page": page_num,
                        "is_table": True,
                        "content_type": "table",
                        "table_rows": len(previous_table_df),
                        "table_columns": len(previous_table_df.columns),
                        "table_index": table_index,
                        "source_file": file_name
                    }

    if previous_table_df is not None:
        docs.append(Document(
            page_content=previous_table_df.to_markdown(index=False),
            metadata=previous_table_metadata
        ))
    return docs
