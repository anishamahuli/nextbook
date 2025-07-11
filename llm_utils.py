from openai import OpenAI
import os

from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_search_query(prompt: str) -> str:
    system_prompt = (
    "You are a helpful assistant. The user wants a book recommendation. "
    "Extract a clean search query using only keywords, genres, or book titles. "
    "Do not include labels like 'Search query:' or full sentences. "
    "Just return a comma-separated list or short phrase. "
    "Do not include words like 'book', 'novel', or 'story'. "
    "Focus on the main themes, genres, or specific titles mentioned by the user. "
    "If the user provides a specific book title, return the genre, setting, or other relevant keywords. "
    "If the user asks for books similar to a those of a specific author, return the author's name and the genre they write in. "
    "Avoid filler words or phrases. "
    "Keep it concise and relevant to the user's request. "
    "Examples:\n"
    "User: 'novels like Pride and Prejudice'\n"
    "You: classic romance\n"
    "User: 'I want something slow and sad about memory and love'\n"
    "You: grief, memory, love, slow fiction\n"
    "User: 'I want a gritty war novel set in the trenches of World War I'\n"
    "You: World War I, war, trenches\n"
    "User: 'angsty coming-of-age novels\n"
    "You: coming-of-age, angst, young adult\n"

)
    

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

def extract_subject(prompt: str) -> str:
    system_prompt = (
        "You are a helpful assistant. The user wants a book recommendation. "
        "Extract just the main subject, genre, or theme from the user's prompt. "
        "Return it as a single word or short phrase, without any additional text. "
        "Try to go a step deeper than just the broad genre. For example, if the user says 'Books like It', "
        "you might return 'horror fiction', 'Stephen King', or 'coming-of-age."
        "Examples: 'contemporary romance', 'science fiction', 'war history', 'mystery', 'fantasy', 'self-help', 'biography', 'thriller', 'poetry', 'children\'s literature'"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
