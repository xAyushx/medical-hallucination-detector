import time

t0 = time.time()
from sentence_transformers import SentenceTransformer
import json
print(f"Import took {time.time() - t0:.2f}s")

t1 = time.time()
model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')
print(f"Model loading took {time.time() - t1:.2f}s")

datapath = 'data/pubmed_chunks.jsonl'
chunktext=[]

t2 = time.time() 
with open(datapath, 'r') as file:
    for i, line in enumerate(file):
        if i >= 100: 
            break
        chunktext.append(json.loads(line)['chunk_text'])
        
embeddings = model.encode(chunktext)

print(f"Encoding took {time.time() - t2:.2f}s") 
