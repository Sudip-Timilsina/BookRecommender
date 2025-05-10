import pandas as pd
from django.shortcuts import render

import random

# Import Chroma, TextLoader, Embeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load the dataset once
df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\book-recommender\recommender_llm\books_with_emotions.csv").fillna("")


# Pre-compute genres and emotions
GENRES = sorted(df['simple_categories'].dropna().unique())
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

# Prepare semantic index (once)
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
df["tagged_description"].to_csv("tagged_description.txt", sep="\n", index=False, header=False)
documents = text_splitter.split_documents(TextLoader("tagged_description.txt", encoding="utf-8").load())
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding_model)

# Semantic recommendation fallback
def semantic_retrieve(query: str, top_k: int = 10) -> pd.DataFrame:
    results = db_books.similarity_search(query, k=16)
    isbn_matches = [int(doc.page_content.strip('"').split()[0]) for doc in results]
    return df[df["isbn13"].isin(isbn_matches)].drop_duplicates(subset=["title"]).head(top_k)

def home(request):
    query = request.GET.get('q', '').strip()
    genre = request.GET.get('genre', '').strip()
    emotion = request.GET.get('emotion', '').strip()

    # Start with the full dataset
    filtered_df = df.copy()

    # Apply genre filter first
    if genre:
        filtered_df = filtered_df[filtered_df['simple_categories'].str.contains(genre, case=False, na=False)]

    # Apply emotion filter
    if emotion:
        filtered_df = filtered_df[filtered_df[emotion].astype(float) > 0.7]

    # Apply query filter
    if query:
        # First try normal text search
        text_filtered = filtered_df[
            filtered_df['description'].str.contains(query, case=False)
            | filtered_df['title'].str.contains(query, case=False)
            | filtered_df['tagged_description'].str.contains(query, case=False)
        ]

        # If no match from text, fallback to semantic
        if text_filtered.empty:
            semantic_matches = semantic_retrieve(query)
            # Filter semantic matches by genre and emotion again
            if genre:
                semantic_matches = semantic_matches[semantic_matches['simple_categories'].str.contains(genre, case=False, na=False)]
            if emotion:
                semantic_matches = semantic_matches[semantic_matches[emotion].astype(float) > 0.7]

            filtered_df = semantic_matches
        else:
            filtered_df = text_filtered

    # If no query, genre and emotion filters already applied

    # Drop duplicates and prepare display
    filtered_df = filtered_df.drop_duplicates(subset=['title'])
    filtered_df['short_description'] = filtered_df['description'].apply(lambda x: ' '.join(x.split()[:30]) + '...')

    books = filtered_df[['title', 'authors', 'average_rating', 'thumbnail', 'short_description','description']].head(10).to_dict(orient='records')

    return render(request, 'recommender/index.html', {
        'books': books,
        'genres': GENRES,
        'emotions': EMOTIONS,
        'selected_genre': genre,
        'selected_emotion': emotion,
        'query': query,
    })
