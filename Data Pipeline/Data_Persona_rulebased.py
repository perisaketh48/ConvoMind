import os
import json
import re
import random

TOPIC_DIR = r"C:\GenAi\Chat_square_Bot\topic_output"

habit_patterns = {
    "late sleeper"      : ["sleep late", "wake up late", "3am", "2am", "midnight"],
    "food lover"        : ["love food", "ordered", "eating out", "cooking", "baking", "pizza"],
    "fitness enthusiast": ["gym", "yoga", "running", "workout", "exercise", "hiking"],
    "reader"            : ["reading", "book", "novel", "library", "kindle"],
    "gamer"             : ["gaming", "video games", "xbox", "playstation", "nintendo"],
    "music lover"       : ["music", "guitar", "singing", "concert", "band"],
    "traveler"          : ["travel", "trip", "vacation", "flight", "exploring"]
}

fact_patterns = {
    "has pet"      : ["my dog", "my cat", "my pet", "my puppy"],
    "is student"   : ["studying", "college", "university", "student", "school"],
    "is working"   : ["my job", "at work", "my boss", "office", "career"],
    "has siblings" : ["my sister", "my brother", "my sibling"],
    "has family"   : ["my mom", "my dad", "my wife", "my husband", "my kids"],
    "has car"      : ["my car", "my vehicle", "driving", "my truck"]
}

personality_patterns = {
    "humorous"    : ["lol", "haha", "funny", "😂", "joke", "hilarious"],
    "emotional"   : ["sad", "crying", "happy", "excited", "stressed", "😢"],
    "caring"      : ["hope you're ok", "take care", "worried about"],
    "adventurous" : ["adventure", "explore", "new places", "spontaneous"],
    "ambitious"   : ["goals", "career", "success", "achieve", "working hard"],
    "creative"    : ["art", "design", "creative", "paint", "draw", "photography"]
}

def check_patterns(text, patterns):
    found      = []
    text_lower = text.lower()
    for label, keywords in patterns.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                if label not in found:
                    found.append(label)
                break
    return found

def get_communication_style(messages):
    style        = []
    total_words  = 0
    emoji_count  = 0
    casual_count = 0
    formal_count = 0

    casual_words  = ["gonna", "wanna", "lol", "omg", "yeah", "nah", "tbh", "bruh"]
    formal_words  = ["however", "therefore", "regarding", "furthermore", "indeed"]
    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE
    )

    for msg in messages:
        words        = msg.split()
        total_words += len(words)
        msg_lower    = msg.lower()
        emoji_count += len(emoji_pattern.findall(msg))

        for w in casual_words:
            if w in msg_lower:
                casual_count += 1

        for w in formal_words:
            if w in msg_lower:
                formal_count += 1

    avg_words = total_words / len(messages) if messages else 0

    if avg_words < 8:
        style.append("very short messages")
    elif avg_words < 15:
        style.append("short messages")
    elif avg_words < 25:
        style.append("medium length messages")
    else:
        style.append("long detailed messages")

    if emoji_count > 5:
        style.append("frequent emoji user")
    elif emoji_count > 0:
        style.append("occasional emoji user")
    else:
        style.append("no emoji usage")

    if casual_count > formal_count:
        style.append("casual tone")
    elif formal_count > casual_count:
        style.append("formal tone")
    else:
        style.append("neutral tone")

    return style

def extract_persona_rules(messages):
    full_text   = " ".join(messages)
    habits      = check_patterns(full_text, habit_patterns)
    facts       = check_patterns(full_text, fact_patterns)
    personality = check_patterns(full_text, personality_patterns)
    comm_style  = get_communication_style(messages)

    return {
        "habits"             : habits,
        "personal_facts"     : facts,
        "personality_traits" : personality,
        "communication_style": comm_style,
        "source"             : "rule_based"
    }

total_files = len(os.listdir(TOPIC_DIR))
print(f"Total topic files: {total_files}")

for file_idx in range(total_files):
    topic_path = os.path.join(TOPIC_DIR, f"topic_{file_idx}.json")

    if not os.path.exists(topic_path):
        print(f"⚠️ topic_{file_idx}.json not found, skipping.")
        continue

    with open(topic_path, "r") as f:
        topics = json.load(f)

    updated = False

    for topic in topics:
        if "persona" in topic:
            continue

        messages = topic["messages"]

        if not messages:
            continue

        topic["persona"] = extract_persona_rules(messages)
        updated          = True

    if updated:
        with open(topic_path, "w") as f:
            json.dump(topics, f, indent=4)

    if (file_idx + 1) % 500 == 0:
        print(f"✅ Processed {file_idx + 1} / {total_files}")

print("✅ Rule based persona extraction complete!")