from src.claim_extractor import extract_claims
from src.retriever import retrieve
from src.nli_verifier import verify_claim
from src.aggregate_result import getresult
sample="Although recent trials have raised questions about its long-term cardiovascular safety, ibuprofen remains one of the most widely used over-the-counter analgesics."
claims=extract_claims(sample)

results = retrieve(claims)
evidence_list = []

claim_evidence = []

for i, claim in enumerate(claims):

    docs = results["documents"][i]
    metas = results["metadatas"][i]
    dists = results["distances"][i]

    evidence = []

    for doc, meta, dist in zip(docs, metas, dists):

        evidence.append({
            "text": doc,
            "pmid": meta["pmid"],
            "distance": dist
        })

    claim_evidence.append({
        "claim": claim,
        "evidence": evidence
    })

# for each claim i will call verify claims from nli verofier and it will return for each evidence one entail,contra,neutral
for  item in claim_evidence:
    claim=item["claim"]
    evidence_list=item["evidence"]
    result=verify_claim(claim,evidence_list)
    output=getresult(result)
    print(f"\nClaim:{claim}")
    print(output)
