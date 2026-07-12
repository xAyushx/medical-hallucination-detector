import chromadb
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="pubmed_corpus",
    metadata={"hnsw:space": "cosine"}
)
def retrieve(claims, k=3):
    embeddings = embed_model.encode(claims, normalize_embeddings=True).tolist()
    return collection.query(
        query_embeddings=embeddings,
        n_results=k
    )