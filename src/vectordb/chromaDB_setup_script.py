import json
import os
import uuid
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Paths
INPUT_JSON = 'hybrid_chunked_schemes.json'
CHROMA_DB_DIR = 'chroma_db'
COLLECTION_NAME = 'government_schemes_chunks'

# Load chunked data
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Initialize ChromaDB client with persistent storage
client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Create or get collection
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare data for indexing
documents = []
metadatas = []
ids = []

for chunk in tqdm(chunks, desc="Processing chunks"):
    text = chunk['text']
    chunk_id = chunk['chunk_id']
    metadata = {
        'parent_doc_id': chunk['parent_doc_id'],
        'field': chunk['field'],
        'scheme_name': chunk['scheme_name']
    }

    documents.append(text)
    metadatas.append(metadata)
    ids.append(chunk_id)

# Generate embeddings
embeddings = model.encode(documents, show_progress_bar=True)



#Adding to ChromaDB
BATCH_SIZE = 500

for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Storing in ChromaDB"):
    collection.add(
        documents=documents[i:i+BATCH_SIZE],
        embeddings=embeddings[i:i+BATCH_SIZE],
        metadatas=metadatas[i:i+BATCH_SIZE],
        ids=ids[i:i+BATCH_SIZE]
    )


print(f"✅ Indexed {len(documents)} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
