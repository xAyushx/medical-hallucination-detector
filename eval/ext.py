import json

path = "eval/raw_eval_results.json"
with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

result = []
for item in data:
    current = {}
    current['id'] = item['id']
    current['claim'] = item['claim']
    current['gold_verdict'] = item['gold_verdict']
    current['rank1_distance'] = item['nli_results'][0]['distance']

    best_contradiction_chunk = max(item['nli_results'], key=lambda r: r['probabilities']['contradiction'])
    best_entailment_chunk = max(item['nli_results'], key=lambda r: r['probabilities']['entailment'])

    current['best_contradiction'] = best_contradiction_chunk['probabilities']['contradiction']
    current['best_entailment'] = best_entailment_chunk['probabilities']['entailment']

    result.append(current)

with open("eval/claim_scores.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

for r in result:
    print(r['id'], r['gold_verdict'], round(r['rank1_distance'], 4), round(r['best_contradiction'], 4), round(r['best_entailment'], 4))