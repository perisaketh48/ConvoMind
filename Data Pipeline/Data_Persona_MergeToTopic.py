import os
import json
from collections import Counter

TOPIC_DIR    = r"C:\GenAi\Chat_square_Bot\topic_output"
OUTPUT_FILE  = r"C:\GenAi\Chat_square_Bot\persona.json"

total_files  = len(os.listdir(TOPIC_DIR))

all_habits       = []
all_facts        = []
all_personality  = []
all_comm_style   = []
llm_insights     = []

print("Combining all personas...")

for file_idx in range(total_files):
    topic_path = os.path.join(TOPIC_DIR, f"topic_{file_idx}.json")

    if not os.path.exists(topic_path):
        continue

    with open(topic_path, "r") as f:
        topics = json.load(f)

    for topic in topics:
        if "persona" not in topic:
            continue

        persona = topic["persona"]
        source  = persona.get("source", "rule_based")

        all_habits      += persona.get("habits", [])
        all_facts       += persona.get("personal_facts", [])
        all_personality += persona.get("personality_traits", [])
        all_comm_style  += persona.get("communication_style", [])

# Count most common
def top_items(items, n=10):
    counter = Counter(items)
    return [item for item, count in counter.most_common(n)]

final_persona = {
    "total_conversations" : total_files,
    "habits"              : top_items(all_habits),
    "personal_facts"      : top_items(all_facts),
    "personality_traits"  : top_items(all_personality),
    "communication_style" : top_items(all_comm_style)
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(final_persona, f, indent=4)

print(f"✅ Final persona saved to persona.json!")
print(json.dumps(final_persona, indent=4))