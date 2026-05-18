
### Architecture

Documents
   ↓
Cleaning
   ↓
Smart Chunking
   ↓    
Embeddings
   ↓
Vector DB
   ↓
Hybrid Search
   ↓
Reranker
   ↓
Prompt Engineering
   ↓
LLM

### Create Virtual Environment

This keeps project packages isolated.

```bash
python -m venv venv
#to activate venv run below command
venv\Scripts\activate
```
### Install OpenAI Python Library
pip install openai
pip install langchain langchain-community langchain-openai  pypdf python-doten
pip install chromadb

### Store API Key Securely (.env)
```py
pip install python-dotenv
OPENAI_API_KEY=sk-xxxxxxxx

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
```
### OPENCODE endpoint

https://opencode.ai/zen/v1/chat/completions

### LangChain

LangChain is an open-source framework for building applications powered by large language models (LLMs). Instead of writing raw API calls for every interaction, LangChain provides reusable components, chains, and pre-built logic that let you connect an LLM to external data sources, tools, memory, and other systems.

Think of it as the “wiring” that turns a single LLM call into a real application—handling prompt engineering, context management, retrieval, and multi-step reasoning.


### Core concepts (why you’d use it)

    1. LLM Wrapper
    A unified interface for any LLM (OpenAI, OpenCode AI, local models, Hugging Face). You just change a few parameters, and the rest of your code stays the same.

    2. Chains
    The core idea: link multiple calls or operations together. For example:
    User question → Retrieve documents → Prompt template → LLM → Answer
    This whole sequence is a chain, and LangChain lets you define it declaratively.

    3. Retrievers & Vector Stores
    Built-in integrations with vector databases (Chroma, FAISS, Pinecone, etc.) and embedding models. The RetrievalQA chain we talked about earlier is a ready-made chain that does RAG out of the box.

    4. Prompts & Templates
    Manage prompt structure neatly, with templating and few-shot examples. You can dynamically inject retrieved context, chat history, or system instructions.

    5. Memory
    Adding conversation history to a chain so the bot remembers previous messages in a session.

    6. Agents
    More advanced: the LLM decides which chain or tool to use next (like calling a calculator, searching the web, or querying a database). The agent loops until it has a final answer.

    7. Tools
    Functions that the LLM can call (APIs, databases, Python functions, etc.). Agents use tools to interact with the world.

### Main Components of RAG

| Component       | Purpose                     |
| --------------- | --------------------------- |
| LLM             | Generates answer            |
| Embeddings      | Convert text into vectors   |
| Vector Database | Stores vectors              |
| Retriever       | Finds similar content       |
| Chunking        | Splits documents            |
| Prompt          | Combines context + question |


### Load PDF 

load the pdf using PyPDFLoader from langchain_community document_loaders

```py
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documents/sample.pdf")

documents = loader.load()

print(documents)
```

### split the document
Split the document in chunk using RecursiveCharacterTextSplitter from langchain_text_splitters

### Create Embeddings


Accuracy depends heavily on:

Chunking strategy
Retrieval quality
Prompt engineering
Reranking
Metadata
Query understanding
Hybrid search
Context filtering


### Recommended Chunking

For documents:
    chunk_size=800
    chunk_overlap=100

For technical docs:
    chunk_size=1200

For FAQs:
    chunk_size=300

### upgrade embeddings
using defual embddeing answer is not proper , upgrade the embedding model  BAAI/bge-large-en-v1.5

system role is important, its give the proper instruction about the chatbot is all about and how to responsed, what measure should take while answering the question 

## Recommended Reranker 
cross-encoder/ms-marco-MiniLM-L-6-v2

## data cleaning important before embedding 
should not clean all special charaters as in integration document there must be some charaters 
/api/v1/user?id=5
{"action":"Q"}
user_name":"Admin"
base64==
better to remove everything except safe chars remove only obviously broken OCR garbage

Repeated Headers/Footers
Broken Multiple Spaces
Duplicate Lines

metadata should be in semantic hierarchy

```py

{

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

            # Debugging / observability
            "char_count": len(doc.page_content),
            "word_count": len(doc.page_content.split()),

            # Timestamp
            "ingestion_timestamp": datetime.now().isoformat(),

            # Semantic features
            **semantic_features
        }

```

classify content in metadata should be document specific as this integration doc so it may contain

api_endpoint
json_example
parameter
sample value
description
type
token
authorization
bearer
api key

along with Structural sections like chapter,section,introduction





### Hybrid Search
This improves retrieval quality because:

Vector search understands meaning
BM25 catches exact words, IDs, error codes, function names, etc.


with python a common approach is:

Use ChromaDB for vector similarity
Use BM25 locally for keyword ranking
Merge both scores → Hybrid Search

```sh 
pip install chromadb sentence-transformers rank-bm25 numpy
``` 


### What is Table-Aware Extraction?

Table-aware extraction means:

✅ Detect tables separately from normal text
✅ Preserve rows/columns structure
✅ Store tables as standalone chunks
✅ Avoid splitting tables mid-way

Before chunk extract the text and table seperately using  pdfplumber pandas tabulate and Prevent Tables From Splitting

### What if table itself break in two page how to handle that?(Multi-page Table Continuation)
Solution : Table Stitching - detect table continuation and  merge pages together

### Parent–Child Retrieval



### stop word techniques
Stop word techniques refer to the process of filtering out common words (like "the", "is", "at", "which", "on") from text before it is processed by an NLP model or search engine. These words are called "stop words" because they appear so frequently that they carry very little unique semantic meaning.

Noise Reduction: It prevents search algorithms (like the BM25 you are using) from giving too much weight to common words. For example, in the query "What is the POST URL for RTPS?", the words "What", "is", "the", and "for" are stop words. Removing them leaves "POST URL RTPS", which contains the core intent.

Efficiency: Smaller vocabulary means faster processing and less memory usage for your vector database and BM25 index.

Improved Accuracy: It helps the system focus on the specific technical terms (API, Endpoint, Parameter) that actually distinguish one document from another.
Standard Stop Word Lists: Using libraries like NLTK, SpaCy, or scikit-learn which provide predefined lists of hundreds of common English words.

Custom Stop Word Lists: Modifying the standard lists to include domain-specific words that are noise in your context (e.g., "document", "page", "user" in a legal RAG).

Smart Stop Word Removal (Context-Aware): Analyzing query patterns to decide when to remove words. For example, in "What is the user_name field?", "user_name" is a keyword, not a stop word.


### Semantic Routing / Intent Classification for greeting messages
TODO
### Simple memory store(manual)
```py
chat_history = [] 
 # 1. Prepare the messages list with System prompt
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant... (your existing system prompt)"""
        }
    ]
     # 2. Add previous chat history (to provide context)
    # We only take the last 5-6 messages to avoid hitting token limits
    for msg in chat_history[-6:]: 
        messages.append(msg)
    # 3. Add the CURRENT prompt (which includes the retrieved context)
    messages.append({
        "role": "user",
        "content": prompt
    })
    # 4. Call the LLM
    response = client.chat.completions.create(
        model="minimax-m2.5-free",
        messages=messages
    )
    ai_answer = response.choices[0].message.content
    print("\nAI Answer:\n", ai_answer)
    # 5. SAVE the interaction to history for the NEXT turn
    # IMPORTANT: Save the clean query and answer, NOT the huge prompt with context
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": ai_answer})

```


### Condense Question 
Condense Question (also known as a Standalone Question) is a critical step in professional RAG systems.

Imagine this conversation:

User: "How do I register for the RTPS Portal?"
AI: (Gives a long explanation)
User: "What are the fees?"
If you search your PDF for just "What are the fees?", the search engine might find fees for "Login updates" or "Certificate downloads" instead of "Registration".

The Condense Question step uses an LLM to rewrite the user's short question into a full, detailed sentence using the chat history. In the example above, it would turn "What are the fees?" into:

"What are the registration fees for the RTPS Portal?"

This ensures your vector search finds the exact right page in your document.

### chat history using LangChain memory

| Memory Type                      | Purpose                              |
| -------------------------------- | ------------------------------------ |
| `ConversationBufferMemory`       | Stores full history                  |
| `ConversationBufferWindowMemory` | Stores last N messages               |
| `ConversationSummaryMemory`      | Summarizes old chats                 |
| `ConversationTokenBufferMemory`  | Keeps history within token limit     |
| `VectorStoreRetrieverMemory`     | Stores long-term memory in vector DB |


### Short-term memory
Last 5–10 chats

### Long-term memory
Store embeddings in vector DB

### User profile memory
Store in MongoDB/PostgreSQL




gradio : Build machine learning apps in Python. Create web interfaces for your ML models in minutes

### TODOO : 
chainlit  :  tools for developers and enterprises that want to ship ambitious and reliable AI applications

Save history on local sqlite DB, Give answer for exitsing question from it instead of search again