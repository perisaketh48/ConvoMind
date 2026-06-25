import streamlit as st
import chromadb
import json
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

# =====================================================
# CONFIG
# =====================================================

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOPIC_DIR = os.path.join(BASE_DIR, "topic_output")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoint_output")

PERSONA_FILE = os.path.join(BASE_DIR, "persona.json")

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

TOPIC_COLLECTION_NAME = "topic_summaries"
CHECKPOINT_COLLECTION_NAME = "checkpoint_summaries"


# =====================================================
# GEMINI
# =====================================================

genai.configure(api_key=GEMINI_API_KEY)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# =====================================================
# LOAD RESOURCES
# =====================================================

st.set_page_config(
    page_title="ConvoMind",
    layout="centered"
)

@st.cache_resource
def load_resources():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    topic_collection = client.get_collection(
        TOPIC_COLLECTION_NAME
    )

    checkpoint_collection = client.get_collection(
        CHECKPOINT_COLLECTION_NAME
    )

    return (
        model,
        topic_collection,
        checkpoint_collection
    )


(
    embedding_model,
    topic_collection,
    checkpoint_collection
) = load_resources()


# =====================================================
# PERSONA
# =====================================================

def load_global_persona():

    if not os.path.exists(PERSONA_FILE):
        return {}

    with open(
        PERSONA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =====================================================
# TOPIC LOADER
# =====================================================

def load_topic(
    conversation_id,
    topic_index
):

    path = os.path.join(
        TOPIC_DIR,
        f"topic_{conversation_id}.json"
    )

    if not os.path.exists(path):
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            topics = json.load(f)

        for topic in topics:

            if topic["topic_index"] == topic_index:
                return topic

    except Exception:
        return None

    return None


# =====================================================
# TOPIC RETRIEVAL
# =====================================================

def retrieve_topics(
    query,
    top_k=3
):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = topic_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_topics = []

    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for metadata, distance in zip(
        metadatas,
        distances
    ):

        # Skip weak matches
        if distance > 0.8:
            continue

        conversation_id = metadata["conversation_id"]
        topic_index = metadata["topic_index"]

        topic = load_topic(
            conversation_id,
            topic_index
        )

        if topic:
            retrieved_topics.append(
                topic
            )

    return retrieved_topics


# =====================================================
# CHECKPOINT RETRIEVAL
# =====================================================

def retrieve_checkpoints(query, top_k=2):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = checkpoint_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    checkpoints = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for doc, meta in zip(documents, metadatas):

        checkpoints.append(
            {
                "summary": doc,
                "checkpoint_number": meta.get(
                    "checkpoint_number",
                    "Unknown"
                ),
                "start_message": meta.get(
                    "start_message"
                ),
                "end_message": meta.get(
                    "end_message"
                )
            }
        )

    return checkpoints
# =====================================================
# CONTEXT
# =====================================================

def build_context(
    query,
    persona,
    topics,
    checkpoints
):

    context = ""
    
    context += "\n\n===== RELEVANT TOPICS =====\n\n"

    context += "Habits:\n"
    context += ", ".join(
        persona.get("habits", [])
    )

    context += "\n\nPersonal Facts:\n"
    context += ", ".join(
        persona.get("personal_facts", [])
    )

    context += "\n\nPersonality Traits:\n"
    context += ", ".join(
        persona.get("personality_traits", [])
    )

    context += "\n\nCommunication Style:\n"
    context += ", ".join(
        persona.get("communication_style", [])
    )

    for topic in topics:

        context += (
            f"\nTopic {topic['topic_index']}\n"
        )

        context += (
            f"\nSummary:\n"
            f"{topic['summary']}\n"
        )

        context += (
            f"\nPersona:\n"
            f"{json.dumps(topic['persona'], indent=2)}\n"
        )

        context += "\nMessages:\n"

        context += "\n".join(
            topic["messages"][:5]
        )

        context += "\n\n"

    context += "\n===== RELEVANT CHECKPOINTS =====\n\n"

    for checkpoint in checkpoints:

        context += (
            f"\nCheckpoint "
            f"{checkpoint.get('checkpoint_number')}\n"
        )

        context += (
            f"\nSummary:\n"
            f"{checkpoint.get('summary')}\n"
        )

        context += "\n"

    return context


# =====================================================
# GEMINI
# =====================================================

def ask_gemini(query, context):

    prompt = f"""
You are an expert conversation analyst.

You must answer ONLY using the supplied context.

Rules:

1. Never invent facts.
2. Never guess.
3. Never use external knowledge.
4. Use only:
   - Global persona
   - Topic summaries
   - Topic personas
   - Topic messages
   - Checkpoint summaries

5. If the answer cannot be found in the supplied context,
respond with:
"I could not find sufficient information in the conversation data to answer that question."
Do not guess or infer missing information.

6. If evidence is limited, clearly mention that the answer
is based on limited evidence.

7. Prefer persona information for:
   - habits
   - personality traits
   - communication style
   - personal facts

8. Prefer topic summaries and topic messages for:
   - hobbies
   - interests
   - events
   - activities

Context:

{context}

Question:

{query}

Answer:
"""

    try:

        response = llm.generate_content(
            prompt
        )

        return response.text

    except Exception as e:

        return f"Error: {str(e)}"
# =====================================================
# STREAMLIT UI
# =====================================================



if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found.")
    st.stop()

st.title("🧠 ConvoMind")

with st.sidebar:

    st.header("RAG System")

    st.write(
        "Embedding: all-MiniLM-L6-v2"
    )

    st.write(
        "Vector DB: ChromaDB"
    )

    st.write(
        "Topic Retrieval: Enabled"
    )

    st.write(
        "Checkpoint Retrieval: Enabled"
    )

    st.write(
        "Persona Retrieval: Enabled"
    )

st.markdown(
    "Ask questions about the user Chats."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):
        st.markdown(
            msg["content"]
        )

query = st.chat_input(
    "Ask a question..."
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Searching memories..."):

        persona = load_global_persona()

        topics = retrieve_topics(query)

        checkpoints = retrieve_checkpoints(query)

        if len(topics) == 0 and len(checkpoints) == 0:

            answer = """
I could not find sufficient information in the conversation data to answer that question.

Please ask something related to:

• User habits
• Personality traits
• Communication style
• Personal facts
• Interests
• Topics discussed in the conversations
"""

        else:

            context = build_context(
                query,
                persona,
                topics,
                checkpoints
            )

            if len(context.strip()) < 100:

                answer = """
I could not find sufficient information in the conversation data to answer that question.

Please ask something related to the available conversation history.
"""

            else:

                answer = ask_gemini(
                    query,
                    context
                )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.expander("Retrieved Evidence"):


        st.subheader("Topics")

        for i, topic in enumerate(topics):

            st.write(
                f"Topic {i+1}"
            )

            st.info(topic["summary"])

        
        st.subheader("Topic Personas")

        for topic in topics:
            st.json(topic["persona"])

        st.subheader("Checkpoints")

        for i, checkpoint in enumerate(checkpoints):

            st.success(checkpoint["summary"][:500])