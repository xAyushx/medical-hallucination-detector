# FIX THE PROMPT##****












# Medical Hallucination Detector

A GenAI system that detects factual hallucinations in LLM-generated medical text using retrieval-augmented verification — not a general-purpose RAG chatbot, but a fact-checking pipeline purpose-built for catching subtly wrong medical claims before they reach a user.

## What it does

Given a piece of LLM-generated (or any) medical text, the system:
1. Decomposes it into atomic, independently-checkable factual claims
2. Retrieves relevant evidence for each claim from a curated PubMed corpus
3. Verifies each claim against that evidence, producing a verdict: **Supported**, **Refuted**, or **Not Enough Info**
4. Returns each verdict with its supporting/contradicting evidence and source citation (PMID)

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
Aggregation (safety-first: any contradiction -> Refuted)
        |
        v
Verdict + Evidence + Citation
```

Served via a FastAPI backend (`/verify` endpoint) with a Streamlit demo UI.

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Claim extraction | Llama 3.1 8B (Ollama) | Local, free, sufficient for structured few-shot decomposition |
| Embeddings | `NeuML/pubmedbert-base-embeddings` (768-dim) | Domain-specific; empirically verified sharper separation between related/unrelated medical content vs. general-purpose models (see Design Decisions) |
| Vector store | ChromaDB (persistent, cosine space) | Binds vector + text + metadata as one unit — essential for citation/provenance, which FAISS does not natively support |
| Verification | BiomedBERT (NLI-finetuned) | *(planned — checkpoint TBD)* |
| Backend | FastAPI | Simple, standard REST serving |
| Frontend | Streamlit | Fast to build a demo UI |
| Ingestion | Raw `requests` + `xml.etree.ElementTree` | No Biopython — avoids bioinformatics-framework bloat for what's fundamentally HTTP + XML parsing; demonstrates real retry/rate-limit engineering |
| Sentence segmentation | `pysbd` | Rule-based, purpose-built; regex-based splitting failed on real-world medical text edge cases (decimals, abbreviations, genus names) |

**Deliberately not used**: LangChain (pipeline is fixed and well-understood; raw implementation keeps every design decision explainable), Biopython (see above).

## Corpus

~4,300 sentence-level chunks from PubMed abstracts across three conditions:
- Cardiovascular diseases
- Diabetes mellitus
- Asthma

Sourced via NCBI E-utilities (`esearch` + `efetch`), English-only, last ~10 years, deduplicated across overlapping condition searches.

**Why only 3 conditions, not all of PubMed**: a deliberately scoped corpus allows personal validation of every source abstract and a rigorous hand-labeled evaluation set — both of which don't scale to millions of documents. The pipeline architecture itself is corpus-agnostic; scaling to more conditions requires adding ingestion queries, not redesigning the system.

## Evaluation

A 50-example gold-labeled evaluation set (hand-written and hand-verdicted against source abstracts) is used to measure pipeline accuracy independently of the pipeline's own outputs. Metrics are split into **retrieval accuracy** (was the correct evidence found?) and **verification accuracy** (given correct evidence, was the verdict right?) to isolate where errors originate, rather than reporting a single opaque end-to-end number.

*(Status: eval set and metrics pending — see Known Limitations / Project Status)*

## Known Limitations
### Corpus is biased toward novel findings, underrepresenting foundational drug facts
Verified empirically: searching the corpus for chunks mentioning "metformin" alongside "hepatic" or "glucose production" — metformin's textbook primary mechanism — returned **zero results**, despite metformin being the most commonly prescribed type 2 diabetes drug. The corpus was built from recent (last ~10 years) PubMed abstracts, which tend to report *novel* findings (new drug trials, mechanism studies on newer compounds, real-world outcome studies) rather than restate long-established textbook facts about older, well-studied drugs. As a result, retrieval performs well for claims about recent/novel research findings but is comparatively weak at verifying basic, foundational drug facts — arguably the most common type of claim a general medical Q&A system would actually need to check. This mirrors an earlier finding (see montelukast example above): even when a drug is mentioned in the corpus, coverage often skews toward a specific angle (e.g., regulatory/statistical) rather than the therapeutic-mechanism angle most claims are about.

**Planned improvement**: broaden the ingestion query strategy — e.g., explicitly including PubMed review articles or pharmacology-focused MeSH terms (e.g., "Metformin/pharmacology") alongside the current treatment-outcome-focused queries — to better capture foundational/mechanistic content, not just recent novel findings. Not implemented in the current version due to placement timeline constraints; documented here as a known, deliberate scoping tradeoff rather than an unnoticed gap.
### Retrieval can favor topical pattern-matching over named-entity precision
Embedding-based semantic search compares overall sentence meaning, not keyword overlap. As a result, retrieval can occasionally surface evidence about the *wrong* drug or entity if it shares a strong topical/structural similarity with the claim (e.g., "Corticosteroids treat asthma" scoring higher than "FDA issued a boxed warning for montelukast" for the claim "Montelukast is used to control asthma" — despite the second chunk being the one that actually mentions the drug in question). This happens when the corpus lacks strong on-topic (therapeutic-description) coverage for the specific entity named in a claim, even if other coverage of that entity exists (e.g., regulatory/statistical content). Verified empirically by directly comparing cosine similarity scores between a test claim and known entity-specific chunks vs. the chunks actually retrieved.

### Retrieval does not detect negation or truth value
Embedding similarity captures topical relevance, not factual correctness. A claim that negates a true fact (e.g., "Asthma is *not* a chronic disease...") can still retrieve the correct supporting evidence as a top match, since the embedding model has no mechanism for distinguishing assertion from negation at the semantic-similarity level. This is by design in this architecture — negation and truth-value judgment are the explicit responsibility of the downstream NLI verification stage, not retrieval. Retrieval's job is to find topically relevant evidence; verification's job is to judge whether that evidence supports or contradicts the claim.

### Claim extraction occasionally produces near-duplicate or inconsistently-granular claims
LLM-based claim decomposition does not always produce claims at a fully consistent level of atomicity — observed failure modes include: appending a generic causal restatement as a separate "claim" alongside the specific fact it follows from, fragmenting a single fact (e.g., dose amount + frequency) into incomplete pieces, and producing two claims that paraphrase the same underlying fact. Prompt refinements reduced but did not eliminate these cases. A planned refinement is a post-processing deduplication/consistency step rather than continuing to patch the prompt indefinitely.

### Corpus scope is deliberately limited to 3 conditions
The evidence corpus covers cardiovascular disease, diabetes, and asthma only (~4,300 sentence-level chunks from ~150 PubMed abstracts per condition), not all of PubMed or all medical conditions. This is a deliberate scoping decision to allow personal validation of the corpus and a hand-labeled evaluation set, not a claim that the system generalizes to arbitrary medical domains. Even within a covered condition, topical coverage can be uneven — e.g., a specific drug may be present in the corpus but discussed only from a regulatory/statistical angle rather than a therapeutic-mechanism angle, limiting retrieval quality for claims about that drug's clinical use.

## Design Insight: Verification and Claim Extraction Have a Load-Bearing Dependency

During NLI verification testing (`pritamdeka/PubMedBERT-MNLI-MedNLI`), a negation case initially appeared to fail: the claim "Asthma is not a chronic respiratory disease but affects over 230 million people" was judged as **entailment (99.96%)** against evidence stating asthma *is* a chronic respiratory disease affecting over 230 million people — the opposite of the correct verdict (contradiction).

Isolated testing traced the cause: the NLI model handles negation correctly and reliably when given a properly atomic claim (verified on multiple simple negation pairs, e.g., "Aspirin causes stomach irritation" vs. "Aspirin does not cause stomach irritation" → 99.91% contradiction, correctly identified). The failure only occurred with a **compound claim** — one negated clause ("not a chronic disease") joined to a second, true, unnegated clause ("but affects over 230 million people") via "but." Testing the negated clause alone against the same evidence produced the correct contradiction verdict (99.92%).

**Conclusion**: this was not an NLI model weakness, but a symptom of a non-atomic claim reaching the verification stage. This confirms that claim extraction's atomicity requirement (Phase 3) is not merely a formatting preference — it is a **load-bearing precondition** for verification to function correctly. A compound claim slipping through extraction can silently produce an incorrect verdict at verification, even with a well-performing NLI model. This is a specific, tested example of why the pipeline's stages are interdependent rather than independently correct: verification's reliability assumes claim extraction has already done its job properly, and a bug in one stage can manifest as an apparent failure in a downstream stage.
## Project Status

- [x] Corpus ingestion (PubMed, 3 conditions, cleaned and deduplicated)
- [x] Embeddings + vector storage (ChromaDB, cosine space, ~4,300 chunks)
- [x] Retrieval validated (positive and negative test queries)
- [x] Claim extraction (Llama 3.1 8B, few-shot prompted, iteratively refined)
- [ ] Claim extraction wired to retrieval end-to-end
- [ ] NLI verification (BiomedBERT)
- [ ] Full pipeline aggregation logic
- [ ] Hand-labeled gold evaluation set (50 examples)
- [ ] Evaluation metrics (precision/recall/F1, retrieval vs. verification error split)
- [ ] FastAPI backend
- [ ] Streamlit frontend

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126  # or appropriate CUDA build
pip install sentence-transformers pysbd requests chromadb fastapi uvicorn streamlit pandas scikit-learn

# Install Ollama separately from ollama.com, then:
ollama pull llama3.1:8b
```

## Repository Structure

```
medical-hallucination-detector/
├── data/
│   ├── pubmed_raw.jsonl        # frozen raw abstract snapshot
│   └── pubmed_chunks.jsonl     # cleaned, chunked corpus
├── chroma_db/                  # persistent vector store
├── src/
│   ├── ingestion_pubmed.py     # Phase 1: corpus ingestion
│   ├── build_chroma_index.py   # Phase 2: embeddings + storage
│   ├── claim_extraction.py     # Phase 3: LLM claim decomposition
│   ├── nli_verification.py     # Phase 4: BiomedBERT verification (planned)
│   └── pipeline.py             # Full pipeline wiring (planned)
├── eval/
│   ├── gold_labels.jsonl       # hand-labeled eval set (planned)
│   └── run_eval.py             # evaluation script (planned)
├── api/
│   └── main.py                 # FastAPI backend (planned)
├── app.py                      # Streamlit frontend (planned)
└── README.md
```