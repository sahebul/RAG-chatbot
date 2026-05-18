import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Client setup
client = OpenAI(
    api_key=os.getenv("OPENCODE_API_KEY"),
    base_url="https://opencode.ai/zen/v1"
)

# Constants & Paths
DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
DOCS_DIR = "documents"
