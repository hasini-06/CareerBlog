import fitz  # PyMuPDF
import textwrap
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Chunk text
def chunk_text(text: str, chunk_size: int = 300):
    return textwrap.wrap(text, width=chunk_size)

class SimpleRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2", gen_model="google/flan-t5-small"):
        # embeddings
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        # generator (you can swap with Groq API if you want later)
        self.generator = pipeline("text2text-generation", model=gen_model)

    def build_index(self, docs):
        # docs is a list of strings
        self.chunks = []
        for d in docs:
            self.chunks.extend(chunk_text(d))

        embeddings = self.embedder.encode(self.chunks)
        dim = embeddings[0].shape[0]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def retrieve_and_answer(self, query: str, top_k: int = 3):
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index first.")

        query_embedding = self.embedder.encode([query])
        _, indices = self.index.search(np.array(query_embedding), top_k)

        retrieved_texts = [self.chunks[i] for i in indices[0]]
        context = " ".join(retrieved_texts)

        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        result = self.generator(prompt, max_length=200)

        return result[0]["generated_text"]
