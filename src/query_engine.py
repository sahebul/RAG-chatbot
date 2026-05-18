from typing import Iterator
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from config import client
from src.preprocessing.metadata import get_doc_identity
from src.loaders.pdf import expand_with_related_tables
from src.indexer.bm25_index import tokenize_for_bm25

# Load models once when query engine imports
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def get_query_engine_embeddings():
    return embeddings

def chat_function_engine(message: str, history: list, vectorstore, bm25, docs) -> Iterator[str]:
    chat_history = []
    
    # Handle Gradio 5 vs older formats
    if len(history) > 0 and isinstance(history[0], dict):
        chat_history = history[-10:]
    else:
        for turn in history[-5:]:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                chat_history.append({"role": "user", "content": turn[0]})
                chat_history.append({"role": "assistant", "content": turn[1]})
                
    # 1. CONDENSE QUESTION STEP
    search_query = message
    if len(chat_history) > 0:
        history_context = ""
        for msg in chat_history[-6:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                history_context += f"{role}: {content}\n"
                
        condense_prompt = f"""Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question (in its original language). 
Do NOT answer the question. Just return the rephrased question.
Chat History:
{history_context}
Follow-up Input: {message}
Standalone Question:"""

        try:
            condense_response = client.chat.completions.create(
                model="minimax-m2.5-free",
                messages=[{"role": "user", "content": condense_prompt}]
            )
            search_query = condense_response.choices[0].message.content.strip()
            print(f"\n[Search refined to: {search_query}]") 
        except Exception as e:
            print(f"Error condensing question: {e}")
            search_query = message

    # 2. HYBRID RETRIEVAL
    vector_results = vectorstore.similarity_search(search_query, k=20)
    tokenized_query = tokenize_for_bm25(search_query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    bm25_results = [docs[i] for i in top_bm25_indices if bm25_scores[i] > 0]

    seen_candidates = set()
    initial_results = []
    for doc in vector_results + bm25_results:
        candidate_id = get_doc_identity(doc)
        if candidate_id not in seen_candidates:
            seen_candidates.add(candidate_id)
            initial_results.append(doc)

    initial_results = expand_with_related_tables(initial_results, docs)

    # 3. RERANK
    doc_texts = [doc.page_content for doc in initial_results]
    if not doc_texts:
        results = []
    else:
        query_doc_pairs = [[search_query, doc] for doc in doc_texts]
        rerank_scores = cross_encoder.predict(query_doc_pairs)
        scored_docs = list(zip(initial_results, rerank_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        results = [doc for doc, score in scored_docs[:5]]
        results = expand_with_related_tables(results, docs)

    # Save debugging context
    with open("retrieved_context.md", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"### Source: Page {r.metadata.get('source_page', '?')}\n\n")
            f.write(r.page_content)
            f.write("\n\n")

    context = "\n".join([
        f"[Source: {r.metadata.get('source_file', 'Unknown')} | Page: {r.metadata.get('source_page', '?')}] {r.page_content}"
        for r in results
    ])

    # 4. FINAL PROMPT GENERATION
    prompt = f"""You are provided with technical details from various integration documents and knowledge bases.
    
Your task:
- Analyze the technical details carefully
- Answer the user's question accurately using only these details
- If the information is not available in the details, say "I don't have that information"

TECHNICAL DETAILS:
{context}

QUESTION:
{message}
"""
    messages = [
        {
            "role": "system",
            "content": """
You are a helpful, professional, and polite Technical Support Assistant specializing in system integrations and API documentation.

Rules:
1. Answer questions based on the technical details provided to you. Do NOT mention that you are reading from a document, file, or provided context.
2. If the user asks a question and the answer is not found in the technical details, politely state that you don't have that information at the moment.
3. If the user sends a basic greeting (e.g., "hi", "hello", "how are you") or asks how you can help, respond naturally as a Technical Assistant. Mention that you can assist with technical integration details, API requirements, and system specifications found in your documentation.
4. Be concise, accurate, and professional.
5. Prefer exact values and definitions from the provided details.
6. Ignore any corrupted or unrelated text in the context.
7. Always sound like a native part of the support platform, not an AI reading a file.
8. If the information pertains to the "RTPS Portal" or "Sewasetu Portal", treat them as the same entity.
"""
        }
    ]
    for msg in chat_history:
        if msg.get("content"):
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content")})
    messages.append({"role": "user", "content": prompt})

    # 5. STREAMING LLM RESPONSE
    response = client.chat.completions.create(
        model="minimax-m2.5-free",
        messages=messages,
        stream=True
    )

    ai_answer = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            ai_answer += chunk.choices[0].delta.content
            yield ai_answer
