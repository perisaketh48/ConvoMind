import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

INPUT_DIR = r"C:\Users\NEXII CONSULTING\Desktop\saketh\embeddings_output"
OUTPUT_DIR = r"C:\GenAi\Chat_square_Bot\topic_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_topics(embeddings, threshold=0.3):
    topic_boundaries = [0]
    for i in range(1, len(embeddings)):
        similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if similarity < threshold:
            topic_boundaries.append(i)
    return topic_boundaries

def split_into_topics(messages, topic_boundaries):
    topics = []
    for i in range(len(topic_boundaries)):
        start = topic_boundaries[i]
        end = topic_boundaries[i + 1] if i + 1 < len(topic_boundaries) else len(messages)
        topic_messages = messages[start:end]
        topics.append(topic_messages)
    return topics

def summarize_topic(messages):
    total = len(messages)

    if total <= 3:
        return " ".join(messages)
    elif total <= 6:
        top_n = 3
    elif total <= 10:
        top_n = 4
    else:
        top_n = 6

    scores = [len(msg.split()) for msg in messages]

    top_indices = []
    scores_copy = scores.copy()
    for i in range(top_n):
        max_idx = scores_copy.index(max(scores_copy))
        top_indices.append(max_idx)
        scores_copy[max_idx] = -1  
    top_indices = sorted(top_indices)

    summary = " ".join([messages[i] for i in top_indices])
    return summary

total_files = len(os.listdir(INPUT_DIR))
print(f"Total conversations to process: {total_files}")

for idx in range(total_files):
    npz_path = os.path.join(INPUT_DIR, f"conv_{idx}.npz")
    
    if not os.path.exists(npz_path):
        print(f"⚠️ conv_{idx}.npz not found, skipping.")
        continue

    data = np.load(npz_path, allow_pickle=True)
    messages = list(data['messages'])
    embeddings = data['embeddings']

    topic_boundaries = detect_topics(embeddings)
    topics = split_into_topics(messages, topic_boundaries)

    results = []
    for topic_idx, topic_messages in enumerate(topics):
        summary = summarize_topic(topic_messages)
        results.append({
            "topic_index": topic_idx,
            "start_message": topic_boundaries[topic_idx],
            "messages": topic_messages,
            "summary": summary
        })

    output_path = os.path.join(OUTPUT_DIR, f"topic_{idx}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1} / {total_files}")

print("✅ Topic detection complete!")
print(f"Total topic files saved: {len(os.listdir(OUTPUT_DIR))}")