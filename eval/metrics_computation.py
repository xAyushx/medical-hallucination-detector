import json
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append(".")
from src.aggregate_result import getresult
DISTANCE_CUTOFF = 0.3
CONTRADICTION_THRESHOLD = 0.8
ENTAILMENT_THRESHOLD = 0.9

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