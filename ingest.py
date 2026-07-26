import glob
import os

import chromadb
import voyageai

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
VOYAGE_MODEL = "voyage-4-lite"
COLLECTION_NAME = "mongodb_docs"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def embed_chunks(chunks, client):
    embeddings = []
    batch_size = 128
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        result = client.embed(batch, model=VOYAGE_MODEL, input_type="document")
        embeddings.extend(result.embeddings)
    return embeddings


def main():
    txt_files = sorted(glob.glob(os.path.join("docs", "*.txt")))

    all_chunks = []
    for path in txt_files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_text(text))

    print(f"Total chunks created: {len(all_chunks)}\n")
    for i, chunk in enumerate(all_chunks, 1):
        print(f"Chunk {i}: {chunk[:100]!r}")

    api_key = os.environ["VOYAGE_API_KEY"]
    voyage_client = voyageai.Client(api_key=api_key)
    embeddings = embed_chunks(all_chunks, voyage_client)

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [f"chunk-{i}" for i in range(len(all_chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=[{"text": chunk} for chunk in all_chunks],
    )

    print(f"\nVectors stored in '{COLLECTION_NAME}': {collection.count()}")


if __name__ == "__main__":
    main()
