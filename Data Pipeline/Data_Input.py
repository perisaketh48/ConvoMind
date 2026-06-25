import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

INPUT_FILE = r"C:\GenAi\Chat_square_Bot\conversations.xlsx"

OUTPUT_DIR = r"C:\GenAi\Chat_square_Bot\embeddings_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_into_messages(conversation_text):
    lines = [line.strip() for line in str(conversation_text).strip().split('\n') if line.strip()]
    return lines

data = pd.read_excel(INPUT_FILE)
data.columns = ["message"]

model = SentenceTransformer('all-MiniLM-L6-v2')
total_rows = len(data)
print(f"Total conversations: {total_rows}")

for index in range(total_rows):
    conversation_text = data.iloc[index]['message']
    messages = split_into_messages(conversation_text)

    if not messages:
        print(f"⚠️ Row {index} empty, skipping.")
        continue

    embeddings = model.encode(messages)

    np.savez(
        os.path.join(OUTPUT_DIR, f"conv_{index}.npz"),
        messages=np.array(messages, dtype=object),
        embeddings=embeddings
    )

    if (index + 1) % 50 == 0:
        print(f"Processed {index + 1} / {total_rows}")

print("✅ Embedding process complete!")