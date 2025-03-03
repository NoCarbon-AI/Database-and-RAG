import os
import torch
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import PointStruct, Distance, VectorParams
from PyPDF2 import PdfReader

# Step 1: Configure settings
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL", "your-qdrant-cloud-url")  
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-api-key")  
COLLECTION_NAME = "data_embeddings"  

if "your-qdrant-cloud-url" in QDRANT_CLOUD_URL or "your-api-key" in QDRANT_API_KEY:
    raise ValueError("Missing Qdrant credentials. Set QDRANT_CLOUD_URL and QDRANT_API_KEY as environment variables.")

# Initialize Qdrant client
client = qdrant_client.QdrantClient(
    url=QDRANT_CLOUD_URL, 
    api_key=QDRANT_API_KEY, 
    timeout=60  
)

# Load embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(MODEL_NAME)

def get_embedding(text):
    """Generate embedding vector using all-MiniLM-L6-v2."""
    return embed_model.encode(text).tolist()

# Step 2: Create collection in Qdrant
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print(f"Collection '{COLLECTION_NAME}' created successfully.")

# Step 3: Extract text from PDF
pdf_path = "Germany-GHG.pdf"  
pdf_reader = PdfReader(pdf_path)
text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

# Step 4: Chunk text for embedding
def create_word_chunks(text, chunk_size=500, overlap=120):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap  
    return chunks

chunks = create_word_chunks(text, chunk_size=500, overlap=120)

# Step 5: Generate embeddings for each chunk
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

# Step 6: Upload data to Qdrant in batches
def batch_upsert(client, collection_name, points, batch_size=10):
    """Uploads data in smaller batches to prevent timeouts."""
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            client.upsert(collection_name=collection_name, points=batch)
            print(f"Uploaded batch {i // batch_size + 1}/{(len(points) // batch_size) + 1}")
        except ResponseHandlingException as e:
            print(f"Error uploading batch {i // batch_size + 1}: {e}")

# Prepare points for Qdrant
points = [
    PointStruct(id=i, vector=chunk_embeddings[i], payload={"text": chunks[i]})
    for i in range(len(chunks))
]

batch_upsert(client, COLLECTION_NAME, points, batch_size=10)

print(f"Successfully stored {len(chunks)} chunks in Qdrant Cloud under '{COLLECTION_NAME}'.")
