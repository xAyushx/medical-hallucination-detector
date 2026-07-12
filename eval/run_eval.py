# eval/run_eval.py
# Run from project root: python eval/run_eval.py
import sys, json
sys.path.append(".")

from src.retriever import retrieve
from src.nli_verifier import verify_claim

GOLD_PATH = "eval/evalution_set_final.jsonl"
OUTPUT_PATH = "eval/raw_eval_results.json"

def load_gold(path):
    claims = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    return claims

def build_evidence_list(documents, distances, metadatas):
    evidence_list = []
    for text, dist, meta in zip(documents, distances, metadatas):
        evidence_list.append({
            "text": text,
            "distance": dist,
            "pmid": meta["pmid"]
        })
    return evidence_list

if __name__ == "__main__":
    gold_claims = load_gold(GOLD_PATH)
    all_results = []

    for i, gold in enumerate(gold_claims):
        claim_text = gold["claim"]
        print(f"[{i+1}/{len(gold_claims)}] Processing claim {gold['id']}...")

        query_result = retrieve([claim_text], k=3)
        documents = query_result["documents"][0]
        distances = query_result["distances"][0]
        metadatas = query_result["metadatas"][0]

        evidence_list = build_evidence_list(documents, distances, metadatas)
        nli_results = verify_claim(claim_text, evidence_list)

        all_results.append({
            "id": gold["id"],
            "claim": claim_text,
            "gold_verdict": gold["eval_verdict"],
            "gold_source_pmid": gold["source_pmid"],
            "nli_results": nli_results  # list of {pmid, distance, text, probabilities}
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. Raw results saved to {OUTPUT_PATH}")