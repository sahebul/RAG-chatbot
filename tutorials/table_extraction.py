import pandas as pd
import pdfplumber
from langchain_core.documents import Document
documents = []


HEADER_TERMS = {"sl", "parameters", "type", "description", "sample value"}


def normalize_cell(value):
    return str(value or "").strip()


def normalize_row(row):
    return [normalize_cell(value).lower() for value in row]


def is_blank_table(table):
    return all(not normalize_cell(cell) for row in table for cell in row)


def is_header_row(row):
    row_text = " ".join(normalize_row(row))
    return sum(term in row_text for term in HEADER_TERMS) >= 2


def starts_with_serial(row):
    first_cell = normalize_cell(row[0]) if row else ""
    return first_cell.isdigit() and int(first_cell) > 0


def should_merge_table(previous_df, previous_metadata, current_df, page_num):
    if previous_df is None or previous_metadata is None:
        return False

    if len(current_df.columns) != len(previous_df.columns):
        return False

    if page_num != previous_metadata["end_page"] + 1:
        return False

    first_row = current_df.iloc[0].tolist()
    return not is_header_row(first_row) or starts_with_serial(first_row)


def drop_duplicate_header(previous_df, current_df):
    if current_df.empty:
        return current_df

    previous_header = normalize_row(previous_df.iloc[0].tolist())
    current_header = normalize_row(current_df.iloc[0].tolist())

    if previous_header == current_header:
        print("DEBUG: Dropped duplicate header row")
        return current_df.iloc[1:].reset_index(drop=True)

    return current_df


previous_table_df = None
previous_table_metadata = None

with pdfplumber.open("../documents/sample.pdf") as pdf:
    for page_num, page in enumerate(pdf.pages):
        # Normal text (unchanged)
        text = page.extract_text()
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"page": page_num, "is_table": False, "content_type": "text"}
            ))
        
        tables = page.extract_tables()
        for table_index, table in enumerate(tables):
            if not table or is_blank_table(table):
                continue
            
            df = pd.DataFrame(table).fillna('')
            print(f"DEBUG: Page {page_num}, table {table_index}, shape {df.shape}")
            
            # ----- DECIDE TO MERGE -----
            should_merge = should_merge_table(
                previous_table_df,
                previous_table_metadata,
                df,
                page_num
            )
            
            if should_merge:
                print(
                    f"DEBUG: Merging page {page_num} onto table "
                    f"started on page {previous_table_metadata['page']}"
                )
                df = drop_duplicate_header(previous_table_df, df)
                
                # Append
                previous_table_df = pd.concat([previous_table_df, df], ignore_index=True)
                previous_table_metadata["table_rows"] = len(previous_table_df)
                previous_table_metadata["table_columns"] = len(previous_table_df.columns)
                previous_table_metadata["end_page"] = page_num
            else:
                # Save previous table (if any)
                if previous_table_df is not None:
                    table_text = previous_table_df.to_markdown(index=False)
                    documents.append(Document(page_content=table_text, metadata=previous_table_metadata))
                # Start new table
                previous_table_df = df
                previous_table_metadata = {
                    "page": page_num,
                    "end_page": page_num,
                    "is_table": True,
                    "content_type": "table",
                    "table_rows": len(df),
                    "table_columns": len(df.columns),
                    "table_index": table_index
                }

# Save last table
if previous_table_df is not None:
    table_text = previous_table_df.to_markdown(index=False)
    documents.append(Document(page_content=table_text, metadata=previous_table_metadata))

for i, doc in enumerate(documents):
    if doc.metadata.get("content_type") == "table":
        print(f"\n=== Table Document {i} ===")
        print(
            f"Page: {doc.metadata.get('page')}, "
            f"End page: {doc.metadata.get('end_page')}, "
            f"Rows: {doc.metadata.get('table_rows')}"
        )
        print(doc.page_content)
        print("---")
