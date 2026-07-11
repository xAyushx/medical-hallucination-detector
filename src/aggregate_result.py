CONTRADICTION_THRESHOLD = 0.85  # TODO: tune after Phase 6 gold eval set
ENTAILMENT_THRESHOLD = 0.85     # TODO: tune after Phase 6 gold eval set
LOW_CONTRADICTION_THRESHOLD = 0.40  # TODO: tune after Phase 6 gold eval set


def getresult(scores):
    best_contradiction_chunk = max(scores, key=lambda r: r["probabilities"]["contradiction"])
    best_entailment_chunk = max(scores, key=lambda r: r["probabilities"]["entailment"])

    refute = best_contradiction_chunk["probabilities"]["contradiction"]
    support = best_entailment_chunk["probabilities"]["entailment"]

    if refute >= CONTRADICTION_THRESHOLD:
        verdict = "REFUTED"
        text = best_contradiction_chunk["text"]
        pmid = best_contradiction_chunk["pmid"]
    elif support >= ENTAILMENT_THRESHOLD and refute < LOW_CONTRADICTION_THRESHOLD:
        verdict = "SUPPORTED"
        text = best_entailment_chunk["text"]
        pmid = best_entailment_chunk["pmid"]
    else:
        verdict = "NOT_ENOUGH_INFO"
        text = None
        pmid = None

    return {
        "verdict": verdict,
        "text": text,
        "pmid": pmid
    }