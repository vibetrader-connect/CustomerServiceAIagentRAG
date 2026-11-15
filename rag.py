from typing import Any
import requests
import numpy as np
from pathlib import Path
#import time

#If you want fast offline counting in Python without making an HTTP call to Ollama:
# pyrefly: ignore[missing-import]
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"

# Lowered from 0.55 to 0.35 because short queries ("Where is the shop?")
# have lower cosine similarity against longer paragraphs in nomic-embed-text.
#SIMILARITY_THRESHOLD = input("Enter similarity threshold (0.0-1.0, default 0.35): ").strip()
SIMILARITY_THRESHOLD = 0.45
if SIMILARITY_THRESHOLD:
    SIMILARITY_THRESHOLD = float(SIMILARITY_THRESHOLD) if float(SIMILARITY_THRESHOLD) < 0.6 else 0.45
    
else:
    SIMILARITY_THRESHOLD = 0.35

print(f"Using similarity threshold: {SIMILARITY_THRESHOLD}\n")
print("Customer care: Please note you quit by typing 'exit' during our conversation to quit")

def embed(texts):
    """Generates vector embeddings for a list of strings using Ollama."""
    r = requests.post(
        f"{OLLAMA}/api/embed",
        json={"model": EMBED_MODEL, "input": texts}
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


def ask_llm(prompt):
    """Sends a prompt to the Ollama chat model and returns the response text."""
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


# 1. Load and split markdown files into chunks
chunks = []
for file in Path("docs").glob("*.md"):
    chunks += [p.strip() for p in file.read_text(encoding="utf-8").split("\n\n") if p.strip()]

if not chunks:
    print("Warning: No markdown chunks found in 'docs/'. Running in pure LLM mode.\n")
    chunk_vectors = np.array([])
else:
    chunk_vectors = embed(chunks)
    print(f"Indexed {len(chunks)} chunks\n")


def retrieve(question, k=3, threshold=SIMILARITY_THRESHOLD):
    """
    Retrieves up to 'k' chunks whose cosine similarity meets or exceeds 'threshold'.
    """
    if len(chunks) == 0:
        return []

    # [0] is required so 'q' is a 1D vector (768,) matching chunk_vectors (N, 768)
    q = embed([question])[0]

    # Compute cosine similarity between question vector and all chunk vectors
    norms = np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q)
    scores = np.divide(
        chunk_vectors @ q,
        norms,
        out=np.zeros_like(norms, dtype=float),
        where=norms != 0
    )

    # Sort indices from highest score to lowest
    top_indices = np.argsort(scores)[::-1][:k]

    # Debug output: Shows the top similarity scores so you can see why chunks matched or missed
    #print(f"==> [Debug] Top similarity scores: {[round(float(scores[i]), 3) for i in top_indices]}")
    #print(top_indices)
    #print(f"==> [Debug] Top similarity chunks: {[(chunks[i], round(float(scores[i]), 3)) for i in top_indices]}")

    # Keep only chunks that pass the threshold
    return [chunks[i] for i in top_indices if scores[i] >= threshold]


last_responses= ["NA", "NA", "NA"]  # Store the last 3 responses for context in future answers

# 2. Interactive Q&A Loop
while True:
    if len(last_responses) > 3: last_responses = last_responses[-3:]  # Keep only the last 3 responses

    question = input("Ask: ").strip()
    
    if question.lower() in {"exit", "quit"}:
        print("Customer care: Thank you for using our service. Have a great day! 😊")
        break
    if not question:
        continue

    matched_chunks = retrieve(question)

    if matched_chunks:
        # CASE A: Found matching chunks in your docs -> Answer from docs
        context = "\n\n".join(matched_chunks)
        prompt = f"""You are a helpful customer care assistant. If customer shares his/her name or any input, keep them in mind when answering. Answer the question using the context below and reference the last 3 responses if relevant.
If the context does not contain the answer, answer using your general knowledge. Ask one or two questions to get more information from customer and keep the conversation flowing. Your all responses always include related emojies at end of sentence for better emotional connect with customer.

Context:
{context}

Your Last 3 Responses (if any): 
 Last response: {last_responses[-1]}

 2nd Last response: {last_responses[-2]}

 3rd Last response: {last_responses[-3]}

Question: {question}"""
    else:
        # CASE B: No matching chunks found -> Pure LLM fallback
        prompt = f"""You are a helpful customer care assistant. If customer shares his/her name or any input, keep them in mind when answering. Answer the question using the last 3 responses if relevant.
If the context does not contain the answer, answer using your general knowledge. Ask one or two questions to get more information from customer and keep the conversation flowing. Your all responses always include related emojies at end of sentence for better emotional connect with customer.


Your Last 3 Responses (if any):
 Last response: {last_responses[-1]}

 2nd Last response: {last_responses[-2]}

 3rd Last response: {last_responses[-3]}

Question: {question}"""
    #print(f"\n[Debug] Prompt sent to LLM with {len(enc.encode(prompt))} tokens:\n{prompt}\n")    
    #print("="*80)
    ask_llm_response = ask_llm(prompt)
    last_responses.append(ask_llm_response)
    print("\nCustomer Care: " + ask_llm_response + "\n")
    #print("="*80)
    #print(f" Tokens sent: {len(enc.encode(prompt))} || Tokens received: {len(enc.encode(ask_llm_response))}\n")
