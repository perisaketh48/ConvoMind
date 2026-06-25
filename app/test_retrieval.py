import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path=r"C:\GenAi\Chat_square_Bot\chroma_db"
)

collection = client.get_collection(
    "topic_summaries"
)

query = "cooking food recipes"

embedding = model.encode(
    query
).tolist()

results = collection.query(
    query_embeddings=[embedding],
    n_results=3
)

print(results)