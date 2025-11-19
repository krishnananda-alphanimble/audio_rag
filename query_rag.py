import os
from dotenv import load_dotenv
from mistralai import Mistral
import chromadb

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment")

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def query_rag(query: str, top_k: int = 3):
    print("\n" + "=" * 60)
    print(f"QUERYING: {query}")
    print("=" * 60)
    
    collection = chroma_client.get_collection(name="audio_rag")
    
    query_embedding = mistral_client.embeddings.create(
        model="mistral-embed",
        inputs=query
    ).data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    print(f"\n[RETRIEVED CHUNKS]\n")
    context_parts = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = 1 - distance
        
        print(f"{i+1}. {meta['timestamp']} (relevance: {similarity:.1%})")
        print(f"   {meta['transcript'][:80]}...")
        print()
        
        context_parts.append(f"[{meta['timestamp']}]\n{meta['transcript']}\n{meta['caption']}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Answer the question using ONLY the provided audio chunks.

AUDIO CHUNKS:
{context}

QUESTION: {query}

Answer concisely based on the chunks above. Mention about the context also"""
    
    print("[GENERATING ANSWER]\n")
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    print(f"{answer}\n")
    
    return answer


if __name__ == "__main__":
        query = input("Enter query (or 'quit' to exit): ").strip()
        if query:
            query_rag(query)
        else:
             print("No query found")
             
