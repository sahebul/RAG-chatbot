import os
from typing import List
from langchain_core.documents import Document

def load_txt(file_path: str) -> List[Document]:
    file_name = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(
        page_content=text,
        metadata={
            "source_file": file_name,
            "is_table": False,
            "content_type": "text",
            "page": 0
        }
    )]
