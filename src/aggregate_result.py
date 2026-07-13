# Thresholds calibrated via Phase 7 gold-set evaluation (50 claims), real data-derived,
# not guessed placeholders. See README for full reasoning.
#
# DISTANCE_CUTOFF: real gap in gold data between SUPPORTED/REJECTED (max ~0.28) and
# NOT_ENOUGH_INFO (min ~0.32) rank-1 distances.
#
# CONTRADICTION_THRESHOLD: chosen low (0.8) deliberately, prioritizing catching real
# hallucinations (REJECTED) over avoiding occasional false-rejection of true claims —
# a documented, deliberate safety-first tradeoff. Catches 19/20 gold REJECTED claims;
# misses 1 (claim 9, contradiction=0.099) — a genuine NLI limitation, not fixable by
# threshold tuning.
#
# ENTAILMENT_THRESHOLD: can sit high (0.9) since all real SUPPORTED claims clear it
# easily; 2 known exceptions (claims 1, 4) never reach this branch — both misclassify
# via the contradiction branch instead, due to a documented NLI weakness where
# evidence containing a contrast conjunction ("but"/"yet") sometimes triggers high
# contradiction despite the claim being factually supported.
#
# This project is not production-grade; known, understood, documented error cases
# remain rather than being chased indefinitely, consistent with project scope.

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
#................................#
# import json

# path = "eval/raw_eval_results.json"
# with open(path, "r", encoding="utf-8") as file:
#     data = json.load(file)
# print(data[0])

# result = []
# for item in data:
#     current = {}
#     current['id'] = item['id']
#     current['claim'] = item['claim']
#     current['gold_verdict'] = item['gold_verdict']
#     current['rank1_distance'] = item['nli_results'][0]['distance']
#     current['best_contradiction'] = max(r['probabilities']['contradiction'] for r in item['nli_results'])
#     current['best_entailment'] = max(r['probabilities']['entailment'] for r in item['nli_results'])

#     result.append(current)

# print(result[0]) 



# print(data[0]['id'])
# print(data[0]['claim'])
# print(data[0]['nli_results'][0]['pmid'])
# print(data[0]['nli_results'][0]['distance'])
# print(data[0]['nli_results'][0]['text'])
# print(data[0]['nli_results'][0]['probabilities'])
# print(data[0]['nli_results'][0]['probabilities']['contradiction'])

