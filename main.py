from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm_utils import extract_search_query, extract_subject
import requests
import re

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Utility: search books from Open Library

def clean_query_for_openlibrary(query: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', query)  # remove punctuation
    return "+".join(cleaned.strip().lower().split())

def search_books(search_string, max_results=100):
    url = f"https://openlibrary.org/search.json?q={search_string}&limit={max_results}"
    response = requests.get(url)
    data = response.json()
    return data.get("docs", [])

def subject_search(prompt: str, max_results=50):
    subject = extract_subject(prompt)
    print(f"LLM-extracted subject: {subject}")  # optional debug
    subject_slug = subject.lower().replace(" ", "_")  
    url = f"https://openlibrary.org/subjects/{subject_slug}.json?limit={max_results}"
    response = requests.get(url)
    data = response.json()
    return data.get("works", [])


# Utility: build embedding-friendly string
def describe_book(book):
    title = book.get("title", "")
    author = ", ".join(book.get("author_name", []))
    desc = book.get("first_sentence", {}).get("value", "") if isinstance(book.get("first_sentence"), dict) else ""
    subjects = ", ".join(book.get("subject", [])[:5]) if "subject" in book else ""
    return f"{title} by {author}. {desc} Subjects: {subjects}"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def is_relevant(books, prompt, threshold=0.2, top_n=10):
    if not books:
        print("No books found for relevance check.")
        return False

    prompt_emb = model.encode([prompt])
    book_titles = [book.get("title", "") for book in books]
    book_embs = model.encode(book_titles)

    sims = cosine_similarity(prompt_emb, book_embs)[0]
    # Sort similarities in descending order and take the top N
    top_sims = sorted(sims, reverse=True)[:top_n]
    avg_sim = sum(top_sims) / len(top_sims) if top_sims else 0
    print(f"Average similarity of top {top_n}: {avg_sim}, Threshold: {threshold}")  # optional debug
    for i, idx in enumerate(sorted(range(len(sims)), key=lambda x: sims[x], reverse=True)[:10]):
        print(f"Top {i+1}: {book_titles[idx], sims[idx]}")
    print(avg_sim > threshold)
    return avg_sim > threshold

@app.get("/recommend")
def recommend(prompt: str = Query(..., description="Describe the kind of book you're looking for")):
    query = extract_search_query(prompt)
    print(f"LLM-extracted query: {query}")  # optional debug
    search_string = clean_query_for_openlibrary(query)
    print(f"cleaned up search query: {search_string}")  

    books = search_books(search_string)

    # If OpenLibrary could not find relevant results, try subject-based fallback
    if not is_relevant(books, search_string):
        subject_books = subject_search(prompt)
        books = subject_books if subject_books else books

    if not books:
        return {"results": [], "message": "No books found."}

    book_texts = [describe_book(book) for book in books]
    book_embeddings = model.encode(book_texts, normalize_embeddings=True)
    user_embedding = model.encode(prompt, normalize_embeddings=True)

    scores = cosine_similarity([user_embedding], book_embeddings)[0]
    top_indices = scores.argsort()[::-1][:5]
    results = []

    for idx in top_indices:
        book = books[idx]
        results.append({
            "title": book.get("title"),
            "author": ", ".join(book.get("author_name", [])),
            "subjects": book.get("subject", []),
            "score": round(float(scores[idx]), 3),
            "link": f"https://openlibrary.org{book.get('key', '')}"
        })

    return {"results": results}
