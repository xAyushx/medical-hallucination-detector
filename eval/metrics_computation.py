import json
from sklearn.metrics import classification_report, confusion_matrix

# --- paste your finalized getresult() here, or import it ---
DISTANCE_CUTOFF = 0.3
CONTRADICTION_THRESHOLD = 0.8
ENTAILMENT_THRESHOLD = 0.9

def getresult(nli_results):
    if not nli_results:
        return {"verdict": "NOT_ENOUGH_INFO", "pmid": None, "evidence_text": None, "reason": "no evidence retrieved"}

    top_distance = nli_results[0]["distance"]
    if top_distance > DISTANCE_CUTOFF:
        return {"verdict": "NOT_ENOUGH_INFO", "pmid": None, "evidence_text": None,
                "reason": f"top retrieval distance {top_distance:.4f} exceeds cutoff {DISTANCE_CUTOFF}"}

    refute = max(r["probabilities"]["contradiction"] for r in nli_results)
    support = max(r["probabilities"]["entailment"] for r in nli_results)

    if refute >= CONTRADICTION_THRESHOLD:
        decisive = max(nli_results, key=lambda r: r["probabilities"]["contradiction"])
        return {"verdict": "REJECTED", "pmid": decisive["pmid"], "evidence_text": decisive["text"],
                "contradiction_score": refute}
    elif support >= ENTAILMENT_THRESHOLD:
        decisive = max(nli_results, key=lambda r: r["probabilities"]["entailment"])
        return {"verdict": "SUPPORTED", "pmid": decisive["pmid"], "evidence_text": decisive["text"],
                "entailment_score": support}
    else:
        return {"verdict": "NOT_ENOUGH_INFO", "pmid": None, "evidence_text": None,
                "reason": f"refute={refute:.4f}, support={support:.4f} - inconclusive"}

# --- load raw eval results and apply getresult() ---
with open("eval/raw_eval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

y_true = []
y_pred = []
mismatches = []

for item in data:
    predicted = getresult(item["nli_results"])
    y_true.append(item["gold_verdict"])
    y_pred.append(predicted["verdict"])
    if predicted["verdict"] != item["gold_verdict"]:
        mismatches.append({
            "id": item["id"],
            "claim": item["claim"],
            "gold": item["gold_verdict"],
            "predicted": predicted["verdict"],
            "reason": predicted.get("reason", predicted.get("contradiction_score", predicted.get("entailment_score")))
        })

print(classification_report(y_true, y_pred, labels=["SUPPORTED", "REJECTED", "NOT_ENOUGH_INFO"]))

print("\nConfusion matrix (rows=true, cols=pred), order = SUPPORTED, REJECTED, NOT_ENOUGH_INFO:")
print(confusion_matrix(y_true, y_pred, labels=["SUPPORTED", "REJECTED", "NOT_ENOUGH_INFO"]))

print(f"\nMisclassified ({len(mismatches)}):")
for m in mismatches:
    print(f"  id={m['id']} gold={m['gold']} pred={m['predicted']} | {m['claim'][:70]}")