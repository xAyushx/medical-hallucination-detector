import chromadb
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer

embedmodel = SentenceTransformer('NeuML/pubmedbert-base-embeddings')
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
with open(datapath, 'r',encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        chunk_conditions[data['chunk_id']].add(data['condition'])

ids = []
documents = []
text = []
metadata = []
seen = set()

with open(datapath, 'r',encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        cid = data['chunk_id']
        if cid in seen:
            continue 
        seen.add(cid)

        merged_condition = ",".join(sorted(chunk_conditions[cid]))

        ids.append(cid)
        documents.append(data['chunk_text'])
        text.append(data['chunk_text'])
        metadata.append({
            "pmid": data["pmid"],
            "title": data["title"],
            "condition": merged_condition
        })

print(f"Total unique chunks after dedup: {len(ids)}")

embedding = embedmodel.encode(text, normalize_embeddings=True).tolist()

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embedding,
    metadatas=metadata
)

print(collection.count())