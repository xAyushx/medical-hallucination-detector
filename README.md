# Medical Hallucination Detector

A GenAI system that catches factual hallucinations in LLM-generated (or any)medical text using retrieval-augmented verification. Not a general-purpose RAG chatbot — a fact-checking pipeline purpose-built for catching subtly wrong medical claims before they reach a user.

## What it does

Given a piece of LLM-generated (or any) medical text, the system:
1. Breaks it down into atomic, independently-checkable factual claims
2. Retrieves relevant evidence for each claim from a PubMed corpus
3. Verifies each claim against that evidence and produces a verdict — **Supported**, **Rejected**, or **Not Enough Info**
4. Returns the verdict along with the evidence and source citation (PMID)

## Architecture

```
LLM-generated medical text
        |
        v
Claim Extraction (Llama 3.1 8B via Ollama, few-shot prompted)
        |
        v
Retrieval (PubMedBERT embeddings + ChromaDB, cosine similarity)
        |
        v
Verification (BiomedBERT NLI: entailment / contradiction / neutral)
        |
        v
Aggregation (safety-first: strong contradiction -> Rejected)
        |
        v
Verdict + Evidence + Citation
```

Demoed through a Streamlit UI.

## Tech stack

| Component | Tool | Why |
|---|---|---|
| Claim extraction | Llama 3.1 8B (Ollama) | Local, free, good enough for structured few-shot decomposition |
| Embeddings | `NeuML/pubmedbert-base-embeddings` (768-dim) | Domain-specific — empirically gives sharper separation between related and unrelated medical content than general-purpose embedding models |
| Vector store | ChromaDB (persistent, cosine space) | Binds vector + text + metadata together — needed for citing sources, which FAISS doesn't natively support |
| Verification | `pritamdeka/PubMedBERT-MNLI-MedNLI` | Trained on both general MNLI and medical-specific MedNLI (real clinical notes), not just a biomedical-sounding base model |
| Frontend | Streamlit | Fast to build a working demo UI |
| Ingestion | Raw `requests` + `xml.etree.ElementTree` | No Biopython — this is fundamentally HTTP + XML parsing, and doing it raw meant actually building retry logic and rate-limiting myself rather than hiding it behind a framework |
| Sentence segmentation | `pysbd` | Rule-based, purpose-built for this problem — regex splitting kept breaking on real medical text (decimals, abbreviations, genus names) |

**Deliberately not used**: LangChain (the pipeline is a fixed, well-understood sequence — a raw implementation keeps every design decision explainable) and Biopython (see above).

## Corpus

~155,000 sentence-level chunks from PubMed abstracts across seven conditions: diabetes, hypertension, heart failure, coronary artery disease, stroke, asthma, and COPD.

Sourced via NCBI E-utilities (`esearch` + `efetch`), English-only, last ~25 years (wide enough to catch foundational drug facts that recent papers don't bother restating, narrow enough to avoid outdated standards of care), with a separate pass specifically pulling review articles — reviews are far more likely to restate established mechanisms than primary research papers, which mostly report novel findings. Deduplicated by PMID, with condition tags merged (not dropped) for articles that span multiple conditions.

The corpus started smaller (3 conditions, ~4,300 chunks) and was expanded after testing surfaced real coverage gaps — see Known Limitations for what that testing found and fixed.

## Evaluation

Accuracy is measured against a 50-claim gold set I hand-built and hand-labeled myself: 20 true claims reworded from real corpus sentences, 20 false claims built by flipping one specific, plausible detail in a true claim (a number, a direction, an entity), and 10 "not enough info" claims about conditions outside the corpus, each verified absent via keyword search before being included.

### Results

```
                 precision  recall  f1-score  support
SUPPORTED           1.00     0.85     0.92       20
REJECTED            0.87     1.00     0.93       20
NOT_ENOUGH_INFO     1.00     1.00     1.00       10
accuracy                              0.94       50
```

100% recall on REJECTED — every real hallucination in the gold set was caught, none slipped through. 100% on NOT_ENOUGH_INFO — the retrieval cutoff correctly flagged every claim with no genuinely relevant evidence in the corpus. All 3 errors went the same direction: true claims wrongly flagged as REJECTED, which is the expected cost of a threshold deliberately set to catch hallucinations aggressively (more on that below).

### How the thresholds were actually chosen

```
DISTANCE_CUTOFF = 0.3          # retrieval relevance cutoff
CONTRADICTION_THRESHOLD = 0.8  # deliberately low
ENTAILMENT_THRESHOLD = 0.9
```

These came from looking at the real data, not guessing:

- **Distance cutoff**: sorting retrieval distances by gold verdict showed a clean gap — every SUPPORTED/REJECTED claim had a top match under ~0.28, every NOT_ENOUGH_INFO claim's top match was above ~0.32. 0.3 sits in that gap.
- **Contradiction threshold**: REJECTED claims' contradiction scores clustered at 0.98+, with one exception (a claim where both contradiction and entailment fired high simultaneously — a distinct failure mode, not a threshold problem). SUPPORTED claims' contradiction scores clustered near 0, with two exceptions traced to specific NLI weaknesses below. 0.8 catches essentially every real hallucination without adding new false positives beyond those two known cases.
- **Why contradiction is checked first, and why the threshold is loose (0.8) rather than strict**: I had to pick which failure mode to favor — a true claim getting wrongly flagged, or a hallucination slipping through unflagged. In a medical context the second one is worse, so the rule is deliberately biased toward catching contradictions even at the cost of a few false rejections.
- These thresholds were then run against the real pipeline output (not simulated) to get the confusion matrix and metrics above.

**Caveat worth being upfront about**: these thresholds were tuned and evaluated on the same 50 examples, which risks overfitting to them. With only 50 labeled claims split across 3 classes, a proper held-out validation split wasn't really practical — each half would be too small to trust. So this shows the approach is sound and grounded in real signal, not that these exact numbers are provably optimal.

**Why only 50 examples**: hand-labeling doesn't parallelize for a solo project — each claim means reading the real source, writing a genuine (not copy-pasted) rewording, constructing a plausible single-detail flip, and personally verifying the label. Farming this out to an LLM would defeat the point, since the whole value of a gold set is an independent, human-checked ground truth. A production system would want hundreds or thousands of examples, ideally labeled by multiple people with agreement checks — that's a team task, not something to force through solo alongside building the rest of the system. 50 was the largest set I could build and personally verify with real confidence in the time available.
The 3 misclassifications, each traced to a specific cause

Contrast-conjunction evidence throws NLI off. The correct evidence sentence contained a "but" clause introducing unrelated content (e.g., "...is the best predictor... but it is often induced by arterial hypertension..."). NLI gave this 99.9% contradiction despite the claim being genuinely supported — it seems to weight the contrast word itself over the actual semantic relationship.
Reordered multi-fact claims confuse NLI. A claim restating three grouped statistics (high/normal/low platelet survival rates) in a different order than the source's single "respectively"-style sentence was scored 99.96% contradiction despite being factually identical — NLI didn't correctly re-derive which number belonged to which group.
Simultaneous high contradiction and entailment. One claim scored ~0.999 on both contradiction and entailment against its evidence at once — an apparent confidence failure on ambiguous phrasing. The safety-first rule (contradiction checked first) resolved it as REJECTED, which happened to be wrong here but is the correct general policy.

All three are NLI limitations, not retrieval or aggregation bugs — the right evidence was retrieved every time, and the aggregation logic did exactly what it was designed to do given the probabilities it received. Being able to say precisely which pipeline stage owns each failure is only possible because retrieval and verification are separated with inspectable output at every step, rather than being one opaque call.
Known limitations
Claim extraction can silently drop negated clauses in compound sentences. Given "Asthma is not a chronic disease but affects over 230 million people," extraction produced only one claim — the negated half was discarded entirely, not just merged or mis-split. Fixed via an explicit prompt rule and worked example; confirmed resolved by direct re-testing. A related prompt bug also surfaced and was fixed: sentences like "[drug] is the standard treatment for [condition]" occasionally produced a phantom second claim asserting the condition "exists," even though that was never actually stated.
Retrieval doesn't detect negation or truth value. By design — negation/truth judgment is verification's job, not retrieval's.
Retrieval can favor topical pattern-matching over the actual named entity, sometimes badly. For "Montelukast is used to control asthma," retrieval missed montelukast-specific content entirely and instead surfaced a chunk about severe asthma and oral glucocorticosteroids — a different drug and severity tier — which NLI then confidently scored as a REJECTED contradiction. The same pattern showed up with ibuprofen (top-3 chunks were all generic analgesic content, missing 8 real ibuprofen-specific chunks in the corpus) and near a case where two claims stating the same fact landed 0.004 apart on either side of the retrieval cutoff, producing opposite verdicts. This is the most operationally serious failure mode found: it doesn't just miss evidence, it can produce a confidently wrong REJECTED using evidence that isn't actually about the claim.
NLI can misjudge a more specific or hedged evidence sentence as contradicting a general claim. A general claim about inhaled corticosteroids being standard asthma maintenance therapy was rejected against evidence that actually supports it, just with more clinical detail. Confirmed as a repeatable pattern (not a one-off) using a second case, beta-blockers as standard heart failure treatment, plus a control case (a plain, non-hedged claim) that classified correctly — isolating the cause to specificity mismatch, not general-vs-specific claims overall.
NLI's numeric handling fails in both directions. It can miss a genuinely wrong number (a claim altered from 50% to 15% wasn't flagged as contradictory), and separately, it can treat vague evidence as confirming a precise claim it never actually stated (evidence saying asthma affects "millions" was accepted as SUPPORTED for a claim asserting "hundreds of millions"). In both cases the model appears to pattern-match on overall similarity rather than doing real numeric comparison.
Claim extraction occasionally produces near-duplicate or inconsistently-sized claims. Prompt refinements reduced this but didn't eliminate it; a post-processing dedup pass would be a cleaner fix than continuing to patch the prompt.
Corpus coverage is uneven even within a covered condition, and the review-article-pass fix was partial, not universal. Early testing found zero chunks connecting metformin to its own basic mechanism (reducing hepatic glucose production) or its common GI side effects, despite metformin being the most prescribed type 2 diabetes drug. Adding a review-article search pass fixed this for montelukast and sinomenine — but re-testing metformin directly (keyword search, require_all=True on ["metformin", "hepatic glucose"] and ["metformin", "gastrointestinal"]) confirmed 0 hits on both, meaning the gap was not actually closed for metformin. The mitigation is real but condition-specific, not a general guarantee.
The whole thing runs locally and isn't deployable as a public service as-is, because claim extraction depends on Ollama running on this machine. A hosted version would need a GPU cloud instance or a hosted LLM API for extraction — not done here, on purpose, to keep total cost at zero.

## A pattern worth naming

Several of the limitations above share the same shape: claim extraction, retrieval, and NLI can each behave "correctly" in isolation and still produce a wrong final verdict, depending on the syntactic shape of the input — a stray conjunction, a reordered list, a claim that's more specific than its evidence. Atomicity and structural alignment turn out to be load-bearing assumptions the pipeline doesn't fully guarantee on its own. The clearest example: a compound negated claim was initially judged as entailment by NLI, which looked like an NLI bug — but testing the negated clause alone against the same evidence gave the correct answer (99.92% contradiction). The NLI model was fine; the claim reaching it wasn't atomic. Verification's reliability turns out to depend on claim extraction having already done its job, not just on NLI being accurate.

## Project status

- [x] Corpus ingestion — 7 conditions, cleaned, deduplicated, condition tags merged for cross-listed articles
- [x] Embeddings + vector storage — ChromaDB, cosine space, ~155k chunks
- [x] Retrieval, validated against known-good and known-bad queries
- [x] Claim extraction — Llama 3.1 8B, few-shot, iteratively refined
- [x] NLI verification — has its own documented limitations, see above
- [x] Aggregation logic — thresholds derived from the gold set, not guessed
- [x] 50-example hand-labeled gold set
- [x] Evaluation metrics — precision/recall/F1, every error individually root-caused
- [x] Streamlit frontend
- [ ] Not deployed as a public service (see limitations)

## Setup

```bash
python -m venv venv
venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # or the CUDA build matching your GPU
pip install sentence-transformers pysbd requests chromadb streamlit pandas scikit-learn

# Ollama installed separately from ollama.com, then:
ollama pull llama3.1:8b
```