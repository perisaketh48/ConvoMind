# ConvoMind – Persona-Based RAG Chatbot

## Overview

ConvoMind is a Retrieval-Augmented Generation (RAG) chatbot that retrieves relevant information from long conversation histories while maintaining user persona and contextual understanding.

Instead of sending the entire conversation history to the LLM, ConvoMind uses semantic search with ChromaDB and Sentence Transformers to retrieve only the most relevant context, improving both efficiency and response quality.

---

## Features

- Persona Extraction from conversation history
- Topic-based conversation segmentation
- Checkpoint summarization
- Semantic search using ChromaDB
- Sentence Transformer embeddings
- Gemini API integration
- Streamlit user interface
- Dockerized deployment

---

## Tech Stack

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Google Gemini API
- Docker

---

## Project Structure

```
app/
    chat_app.py
    chroma_db/
    topic_output/
    checkpoint_output/
    persona.json

Data Pipeline/
    Data_Input.py
    Data_Similarity.py
    Data_ChromaDb.py
    Data_checkpoint.py
    Data_Persona_llmbased.py

Dockerfile
requirements.txt
```

---

## How It Works

Conversation Dataset

↓

Sentence Embeddings

↓

Topic Detection

↓

Topic Summaries

↓

Checkpoint Summaries

↓

ChromaDB

↓

Semantic Retrieval

↓

Gemini LLM

↓

Final Response

---

## Running Locally

Install dependencies

```bash
pip install -r requirements.txt
```

Set environment variables

```
GOOGLE_API_KEY=YOUR_KEY
```

Run

```bash
streamlit run app/chat_app.py
```

---

## Docker

Build

```bash
docker build -t convomind .
```

Run

```bash
docker run -p 8501:8501 \
-e GOOGLE_API_KEY=YOUR_KEY \
convomind
```

---

## Demo

🎥 Demo Video

(Add your YouTube or LinkedIn demo link here.)

---

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases
- Embedding Models
- LLM Integration
- Docker
- Streamlit
- Python
- Information Retrieval
