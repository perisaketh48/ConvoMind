import os
import json
import random
from groq import Groq

TOPIC_DIR  = r"C:\GenAi\Chat_square_Bot\topic_output"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"


client      = Groq(api_key=GROQ_API_KEY)
total_files = len(os.listdir(TOPIC_DIR))

random.seed(42)
random_500  = random.sample(range(total_files), 500)
print(f"Total files: {total_files}")
print(f"Processing 500 random files with Groq...")

def extract_persona_groq(messages, retries=3):
    conversation_text = "\n".join(messages)
    prompt = f"""
From the following conversation extract persona information.
Return ONLY a JSON object with no extra text.
No thinking, no explanation, just JSON.

Conversation:
{conversation_text}

Return this exact JSON format:
{{
    "habits": [],
    "personal_facts": [],
    "personality_traits": [],
    "communication_style": []
}}

Rules:
- Only include what is clearly mentioned
- No guesses
- Keep each item short one line
- If nothing found leave list empty
"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0,
            )
            result = response.choices[0].message.content
            start  = result.index("{")
            end    = result.rindex("}") + 1
            persona = json.loads(result[start:end])
            persona["source"] = "groq_llm"
            return persona
        except Exception as e:
            print(f"⚠️ Retry {attempt+1} failed: {e}")

    return None

processed  = 0
failed     = 0

for count, file_idx in enumerate(random_500):
    topic_path = os.path.join(TOPIC_DIR, f"topic_{file_idx}.json")

    if not os.path.exists(topic_path):
        continue

    with open(topic_path, "r") as f:
        topics = json.load(f)

    updated = False

    for topic in topics:
        messages = topic["messages"]

        if not messages:
            continue

        persona = extract_persona_groq(messages)

        if persona:
            topic["persona"] = persona
            updated          = True
            processed       += 1
        else:
            failed += 1

    if updated:
        with open(topic_path, "w") as f:
            json.dump(topics, f, indent=4)

    if (count + 1) % 50 == 0:
        print(f"✅ Processed {count + 1} / 500 files")
        print(f"   Success: {processed} | Failed: {failed}")

print(f"\n✅ Groq persona extraction complete!")
print(f"   Total processed : {processed}")
print(f"   Total failed    : {failed}")