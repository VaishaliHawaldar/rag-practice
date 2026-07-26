import os
import sys

import chromadb
import voyageai

VOYAGE_MODEL = "voyage-4-lite"
COLLECTION_NAME = "mongodb_docs"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 query.py \"<question>\"")
        sys.exit(1)

    question = sys.argv[1]

    api_key = os.environ["VOYAGE_API_KEY"]
    voyage_client = voyageai.Client(api_key=api_key)
    result = voyage_client.embed([question], model=VOYAGE_MODEL, input_type="query")
    query_embedding = result.embeddings[0]

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    chunks = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (metadata, distance) in enumerate(zip(chunks, distances), 1):
        print(f"Result {i} (distance: {distance:.4f}):")
        print(metadata["text"])
        print()


if __name__ == "__main__":
    main()
