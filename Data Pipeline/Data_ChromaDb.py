import os
import json
import chromadb
from sentence_transformers import SentenceTransformer


TOPIC_DIR      = r"C:\GenAi\Chat_square_Bot\topic_output"
CHECKPOINT_DIR = r"C:\GenAi\Chat_square_Bot\checkpoint_output"
CHROMA_DIR     = r"C:\GenAi\Chat_square_Bot\chroma_db"

client = chromadb.PersistentClient(path=CHROMA_DIR)
model  = SentenceTransformer('all-MiniLM-L6-v2')

topic_collection = client.get_or_create_collection(
    name="topic_summaries"
)

checkpoint_collection = client.get_or_create_collection(
    name="checkpoint_summaries"
)

print("✅ Collections created!")
print(f"Topic collection: {topic_collection.name}")
print(f"Checkpoint collection: {checkpoint_collection.name}")


total_topics = len(os.listdir(TOPIC_DIR))
print(f"Total topic files: {total_topics}")

for idx in range(total_topics):
    topic_path = os.path.join(TOPIC_DIR, f"topic_{idx}.json")

    if not os.path.exists(topic_path):
        print(f"⚠️ topic_{idx}.json not found, skipping.")
        continue

    with open(topic_path, "r") as f:
        topics = json.load(f)

    for topic in topics:
        summary    = topic["summary"]
        topic_idx  = topic["topic_index"]
        start_msg  = topic["start_message"]

        embedding  = model.encode(summary).tolist()

        topic_collection.add(
            ids          = [f"conv_{idx}_topic_{topic_idx}"],
            embeddings   = [embedding],
            documents    = [summary],
            metadatas    = [{
                "type"            : "topic",
                "conversation_id" : idx,
                "topic_index"     : topic_idx,
                "start_message"   : start_msg
            }]
        )

    if (idx + 1) % 500 == 0:
        print(f"✅ Processed {idx + 1} / {total_topics} topic files")

print("✅ Topic summaries stored in ChromaDB!")

total_checkpoints = len(os.listdir(CHECKPOINT_DIR))
print(f"Total checkpoint files: {total_checkpoints}")

for idx in range(1, total_checkpoints + 1):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{idx}.json")

    if not os.path.exists(checkpoint_path):
        print(f"⚠️ checkpoint_{idx}.json not found, skipping.")
        continue

    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)

    summary            = checkpoint["summary"]
    checkpoint_number  = checkpoint["checkpoint_number"]
    start_message      = checkpoint["start_message"]
    end_message        = checkpoint["end_message"]

    embedding = model.encode(summary).tolist()

    checkpoint_collection.add(
        ids        = [f"checkpoint_{checkpoint_number}"],
        embeddings = [embedding],
        documents  = [summary],
        metadatas  = [{
            "type"              : "checkpoint",
            "checkpoint_number" : checkpoint_number,
            "start_message"     : start_message,
            "end_message"       : end_message
        }]
    )

    if idx % 200 == 0:
        print(f"✅ Processed {idx} / {total_checkpoints} checkpoints")

print("✅ Checkpoint summaries stored in ChromaDB!")
print(f"Total topics stored     : {topic_collection.count()}")
print(f"Total checkpoints stored: {checkpoint_collection.count()}")


def search(query, n_results=3):
    query_embedding = model.encode(query).tolist()

    topic_results = topic_collection.query(
        query_embeddings = [query_embedding],
        n_results        = n_results
    )

    checkpoint_results = checkpoint_collection.query(
        query_embeddings = [query_embedding],
        n_results        = n_results
    )

    print(f"\n🔍 Query: {query}")
    print("\n📌 Topic Results:")
    for i in range(len(topic_results['documents'][0])):
        print(f"  Result {i+1}: {topic_results['documents'][0][i]}")
        print(f"  Metadata: {topic_results['metadatas'][0][i]}")

    print("\n📌 Checkpoint Results:")
    for i in range(len(checkpoint_results['documents'][0])):
        print(f"  Result {i+1}: {checkpoint_results['documents'][0][i]}")
        print(f"  Metadata: {checkpoint_results['metadatas'][0][i]}")

search("Did anyone talk about yoga?")