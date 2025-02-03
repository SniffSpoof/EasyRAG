from typing import List, Dict, Any
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import time

import asyncio
import nest_asyncio

import logging

import google.generativeai as genai
from google.api_core import exceptions

import threading
from queue import Queue, Empty

class DocumentProcessor:
    def __init__(self, chunk_size: int = 100, overlap: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, text: str) -> List[str]:
        words = re.split(r'\s+', text)
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - self.overlap if end < len(words) else end

        return chunks

class ContextGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def generate_context(self, full_doc: str, chunk: str) -> str:
        truncated_doc = ' '.join(full_doc.split()[:2000])
        prompt = f"""
        <document>
        {truncated_doc}
        </document>
        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>
        Respond ONLY with context under 50 tokens.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logging.error(f"Ошибка генерации контекста: {e}")
            return ""

def context_worker(api_key: str, task_queue: Queue, result_queue: Queue):
    generator = ContextGenerator(api_key)
    while True:
        try:
            doc, chunk = task_queue.get(timeout=5)
            context = generator.generate_context(doc, chunk)
            result_queue.put((context, chunk))
            task_queue.task_done()
        except Empty:
            break
        except Exception as e:
            logging.error(f"Ошибка в worker {api_key[-5:]}: {e}")
            task_queue.task_done()

class ContextualRetrievalSystem:
    def __init__(self, api_keys: List[str]):
        self.processor = DocumentProcessor()
        self.api_keys = api_keys
        self.vectorizer = TfidfVectorizer()
        self.embeddings = {}
        self.tfidf_vectors = None
        self.documents = []

    def process_corpus(self, documents: List[str]):
        self.documents = documents
        task_queue = Queue()
        result_queue = Queue()

        for doc in documents:
            chunks = self.processor.chunk_document(doc)
            for chunk in chunks:
                task_queue.put((doc, chunk))

        threads = []
        for key in self.api_keys:
            thread = threading.Thread(
                target=context_worker,
                args=(key, task_queue, result_queue),
                daemon=True
            )
            thread.start()
            threads.append(thread)

        task_queue.join()

        contextualized_chunks = []
        while not result_queue.empty():
            context, chunk = result_queue.get()
            contextualized_chunks.append(f"{context}\n\n{chunk}")

        self.tfidf_vectors = self.vectorizer.fit_transform(contextualized_chunks)

        self.embeddings = {
            chunk: self._get_embedding(chunk)
            for chunk in contextualized_chunks
        }

    def _get_embedding(self, text: str) -> List[float]:
        MAX_SIZE = 9500

        if len(text.encode('utf-8')) > MAX_SIZE:
            text = text[:int(MAX_SIZE * 0.8)]

        try:
            embedding = genai.embed_content(
              model='models/embedding-001',
              content=text,
              task_type="retrieval_document"
            )['embedding']
            return np.array(embedding)

        except exceptions.BadRequest as e:
            print(f"Error embedding text of size {len(text)}: {e}")
            return np.zeros(768)  # placeholde



    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        # Semantic search with embeddings
        query_embedding = np.array(genai.embed_content(
            model='models/embedding-001',
            content=query,
            task_type="retrieval_query"
        )['embedding'])

        semantic_scores = [
            cosine_similarity([query_embedding], [emb])[0][0]
            for emb in self.embeddings.values()
        ]

        # Lexical search with BM25
        query_tfidf = self.vectorizer.transform([query])
        lexical_scores = cosine_similarity(query_tfidf, self.tfidf_vectors).flatten()

        # Combine scores
        combined_scores = 0.7 * np.array(semantic_scores) + 0.3 * lexical_scores
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]

        return [{
            'score': combined_scores[i],
            'text': list(self.embeddings.keys())[i],
            'semantic_score': semantic_scores[i],
            'lexical_score': lexical_scores[i]
        } for i in top_indices]

class QASystem:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def ask(self, PROMPT: str, question: str, book: str, context: str) -> str:
        prompt = PROMPT

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Ошибка генерации: {e}")
            return "Не удалось сгенерировать ответ. Попробуйте переформулировать вопрос."