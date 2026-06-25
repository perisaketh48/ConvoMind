import os
import json

INPUT_DIR = r"C:\GenAi\Chat_square_Bot\topic_output"
OUTPUT_DIR = r"C:\GenAi\Chat_square_Bot\checkpoint_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def summarize_messages(messages):
    total = len(messages)

    if total <= 25:
        return " ".join(messages)

    top_n = 25

    scores = [len(msg.split()) for msg in messages]
    scores_copy = scores.copy()

    top_indices = []
    for i in range(top_n):
        max_idx = scores_copy.index(max(scores_copy))
        top_indices.append(max_idx)
        scores_copy[max_idx] = -1

    top_indices = sorted(top_indices)

    summary = " ".join([messages[i] for i in top_indices])
    return summary

current_messages = []
checkpoint_number = 1
global_message_count = 0
total_files = len(os.listdir(INPUT_DIR))
print(f"Total topic files to process: {total_files}")

for file_idx in range(total_files):
    topic_path = os.path.join(INPUT_DIR, f"topic_{file_idx}.json")

    if not os.path.exists(topic_path):
        print(f"⚠️ topic_{file_idx}.json not found, skipping.")
        continue

    with open(topic_path, "r") as f:
        topics = json.load(f)

    for topic in topics:
        for message in topic["messages"]:
            current_messages.append(message)
            global_message_count += 1

            if len(current_messages) == 100:
                summary = summarize_messages(current_messages)
                checkpoint = {
                    "checkpoint_number": checkpoint_number,
                    "start_message": global_message_count - 99,
                    "end_message": global_message_count,
                    "messages": current_messages,
                    "summary": summary
                }
                output_path = os.path.join(OUTPUT_DIR, f"checkpoint_{checkpoint_number}.json")
                with open(output_path, "w") as f:
                    json.dump(checkpoint, f, indent=4)
                print(f"✅ Checkpoint {checkpoint_number} saved!")
                checkpoint_number += 1
                current_messages = []
if len(current_messages) > 0:
    summary = summarize_messages(current_messages)
    checkpoint = {
        "checkpoint_number": checkpoint_number,
        "start_message": global_message_count - len(current_messages) + 1,
        "end_message": global_message_count,
        "messages": current_messages,
        "summary": summary
    }
    output_path = os.path.join(OUTPUT_DIR, f"checkpoint_{checkpoint_number}.json")
    with open(output_path, "w") as f:
        json.dump(checkpoint, f, indent=4)
    print(f"✅ Last checkpoint {checkpoint_number} saved!")

print(f"✅ Checkpoint process complete!")
print(f"Total checkpoints saved: {checkpoint_number}")