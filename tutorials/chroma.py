import chromadb
from sentence_transformers import SentenceTransformer
chroma_client=chromadb.Client()
collection=chroma_client.create_collection(name="test_collection")

# add data

collection.add(
    documents=[
        "This is a document about machine learning",
        "This is another document about data science",
        "A third document about artificial intelligence"
    ],
     metadatas=[
        {"source": "test1","chapter": 3,
        "tags": ["fiction", "adventure"],
        "scores": [1, 2, 3]},
        {"source": "test2"},
        {"source": "test3"}
    ],
    ids=[
        "id1",
        "id2",
        "id3"
    ]
)
# query the doc using text
# results=collection.query(
#     query_texts=[
#         'doc1'
#     ],
#     n_results=2
# )

# when we do embedding search that should be same as insert 

# using embedding 
model=SentenceTransformer('all-MiniLM-L6-v2')
embedded_collection=chroma_client.create_collection(name="em_collection")
docs=["doc1","doc2","doc3"]
embeddings=model.encode(docs).tolist()
embedded_collection.add(
    ids=['id1','id2','id3'],
    embeddings=embeddings,
    documents=docs
)

# query using embeddings
query_embedding=model.encode(['doc1']).tolist()
results=embedded_collection.query(
    query_embeddings=query_embedding,
    n_results=1
)

distance=results['distances'][0][0]
if distance  > 1.5 :
    print("No result found")
else:
    
    print(f"Result: \n\n")
    print(results['documents'])