import chromadb
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer

embedmodel = SentenceTransformer('NeuML/pubmedbert-base-embeddings',device="cuda")
datapath = 'data/pubmed_chunks.jsonl'

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="pubmed_corpus",
    metadata={"hnsw:space": "cosine"}
)
# because during ingestion time same chunk got inserted muliple times because a 
# an article can belong to multiple conditions ex we got an article which had both 
# cardiovas and diabeted in one so when i ran query for dibates same article got 
# pulled here similary it got pulled in cardio thats why we got duplicate chunks
#  so instad of dropping duplicate chunks and possibil loosing on info i just merged 
# them on conditions

chunk_conditions = defaultdict(set)
with open(datapath, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        chunk_conditions[data['chunk_id']].add(data['condition'])

ids = []
documents = []
metadata = []
seen = set()

with open(datapath, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        cid = data['chunk_id']
        if cid in seen:
            continue
        seen.add(cid)

        merged_condition = ",".join(sorted(chunk_conditions[cid]))

        ids.append(cid)
        documents.append(data['chunk_text'])
        metadata.append({
            "pmid": data["pmid"],
            "title": data["title"],
            "condition": merged_condition
        })

print(f"Total unique chunks after dedup: {len(ids)}")


BATCH_SIZE = 2000

for i in range(0, len(ids), BATCH_SIZE):
    batch_ids = ids[i:i + BATCH_SIZE]
    batch_documents = documents[i:i + BATCH_SIZE]
    batch_metadata = metadata[i:i + BATCH_SIZE]

    batch_embeddings = embedmodel.encode(
        batch_documents,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    collection.add(
        ids=batch_ids,
        documents=batch_documents,
        embeddings=batch_embeddings,
        metadatas=batch_metadata
    )

    print(f"Inserted {i + len(batch_ids)} / {len(ids)}")

print(f"\nFinal collection count: {collection.count()}")