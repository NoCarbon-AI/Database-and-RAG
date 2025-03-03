import os
import json
import boto3
import qdrant_client
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

# Step 1: Connect to Qdrant
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL", "your-qdrant-cloud-url")  
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-api-key")  

if "your-qdrant-cloud-url" in QDRANT_CLOUD_URL or "your-api-key" in QDRANT_API_KEY:
    raise ValueError("Missing Qdrant credentials. Set QDRANT_CLOUD_URL and QDRANT_API_KEY as environment variables.")

qdrant = qdrant_client.QdrantClient(
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "data_embeddings"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Search Qdrant for context
def get_embedding(text):
    """Convert text into an embedding vector."""
    return embedding_model.encode(text).tolist()

def search_qdrant(query, top_k=3):
    """Search Qdrant for relevant context based on query embedding."""
    query_vector = get_embedding(query)

    search_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )

    retrieved_texts = [hit.payload["text"] for hit in search_results]
    return "\n".join(retrieved_texts)

# Step 3: Query Claude
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

def query_claude(prompt):
    """Send query to Claude LLM using AWS Bedrock."""
    payload = {
        "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        })
    }

    response = bedrock.invoke_model(
        body=payload["body"],
        modelId=payload["modelId"],
        contentType=payload["contentType"],
        accept=payload["accept"]
    )

    response_body = json.loads(response["body"].read().decode("utf-8"))
    return response_body["content"][0]["text"]

# Step 4: Main function
def main():
    """Main function to process user query, retrieve context from Qdrant, and get response from Claude."""
    user_query = "How did Germany's sources of hard coal imports change from 1990 to 2020?" 

    retrieved_context = search_qdrant(user_query)
    final_prompt = f"Use the following information to answer the question:\n{retrieved_context}\n\nQuestion: {user_query}"
    
    response = query_claude(final_prompt)
    print(response)

if __name__ == "__main__":
    main()
