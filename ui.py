# app.py
# Run: python -m streamlit run app.py
import streamlit as st

st.set_page_config(page_title="Medical Hallucination Detector", layout="wide")

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #1A1C1E !important;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", "IBM Plex Sans", sans-serif;
    }

    .main-title {
        font-size: 1.9rem;
        font-weight: 650;
        color: #E3E8E5;
        margin-bottom: 0.1rem;
        letter-spacing: -0.01em;
        text-align: center;
    }
    .subtitle {
        color: #8A968F;
        font-size: 0.95rem;
        margin-bottom: 1.8rem;
        text-align: center;
    }
    
    .claim-card {
        background-color: #222824;
        border: 1px solid #333C36;
        border-radius: 6px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    
    .claim-text {
        font-size: 1.02rem;
        color: #FAFAFA;
        margin-bottom: 0.6rem;
        font-weight: 500;
    }
    
    .verdict-badge {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.6rem;
    }
    .verdict-supported { background-color: #1B382B; color: #8BE3B2; border: 1px solid #285440; }
    .verdict-rejected  { background-color: #471E1E; color: #FFA1A1; border: 1px solid #632B2B; }
   
    .verdict-nei       { background-color: #3B341F; color: #EAD08D; border: 1px solid #544A2D; }
    
    .citation-block {
        border-left: 3px solid #6E5E4E;
        padding-left: 0.8rem;
        margin-top: 0.5rem;
        color: #C2CBC5;
        font-size: 0.9rem;
        font-style: italic;
    }
    
    .pmid-link {
        font-size: 0.85rem;
        color: #79BFA1;
        text-decoration: none;
        font-weight: 500;
    }
    
    .reason-text {
        color: #8A968F;
        font-size: 0.88rem;
        margin-top: 0.4rem;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #333C36;
        border-radius: 6px;
        background-color: #222824;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Medical Hallucination Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Verifies LLM-generated medical claims against a PubMed evidence corpus.</div>', unsafe_allow_html=True)

left_pad, center_workspace, right_pad = st.columns([1, 3, 1])

with center_workspace:
    with st.expander("How this works"):
        st.markdown("""
    This tool detects **medical hallucinations** in LLM-generated text by verifying each factual claim against real PubMed literature.

    **1. Claim Extraction** Using **Llama 3.1:8B (via Ollama)**, the input is split into atomic, independently verifiable medical claims.

    **2. Evidence Retrieval** Each claim is matched against a corpus of **150,000+ sentence-level chunks** from **13,000+ PubMed abstracts** using semantic search to retrieve the most relevant evidence.

    **3. Claim Verification** A **PubMedBERT-based biomedical NLI model** determines whether the retrieved evidence **supports**, **contradicts**, or is **neutral** toward the claim.

    **4. Final Verdict** Each claim receives one of three outcomes:
    - **SUPPORTED** – Evidence supports the claim.
    - **REJECTED** – Evidence contradicts the claim.
    - **NOT ENOUGH INFO** – Available evidence is insufficient.

    Every verdict includes the supporting PubMed evidence and PMID for verification.
        """)


    @st.cache_resource(show_spinner="Loading models (embedding + NLI+ollama). !!WAIT!!")
    def load_pipeline():
        from src.claim_extractor import extract_claims
        from src.retriever import retrieve
        from src.nli_verifier import verify_claim
        from src.aggregate_result import getresult
        return extract_claims, retrieve, verify_claim, getresult

    extract_claims, retrieve, verify_claim, getresult = load_pipeline()

    VERDICT_META = {
        "SUPPORTED": ("SUPPORTED", "verdict-supported"),
        "REJECTED": ("REJECTED", "verdict-rejected"),
        "NOT_ENOUGH_INFO": ("NOT ENOUGH INFO", "verdict-nei"),
    }

    sample_text = st.text_area(
        "Text to verify",
        height=140,
        placeholder="""Paste LLM-generated(or any) medical response!!
ex.Asthma is a chronic respiratory disease affecting over 300 million people worldwide.
""",
        label_visibility="collapsed",
    )

    run_button = st.button("Verify Claims", type="primary", use_container_width=True)

    if run_button and sample_text.strip():
        with st.spinner("Extracting claims..."):
            claims = extract_claims(sample_text)

        if not claims:
            st.warning("No claims were extracted from this text.")
        else:
            with st.spinner(f"Retrieving evidence and verifying {len(claims)} claim(s)..."):
                results = retrieve(claims)

            for i, claim in enumerate(claims):
                docs = results["documents"][i]
                metas = results["metadatas"][i]
                dists = results["distances"][i]

                evidence = [
                    {"text": d, "pmid": m["pmid"], "distance": dist}
                    for d, m, dist in zip(docs, metas, dists)
                ]

                nli_results = verify_claim(claim, evidence)
                output = getresult(nli_results)

                verdict = output["verdict"]
                label, css_class = VERDICT_META[verdict]

                card_html = f'<div class="claim-card">'
                card_html += f'<div class="verdict-badge {css_class}">{label}</div>'
                card_html += f'<div class="claim-text">{claim}</div>'

                if output.get("pmid"):
                    card_html += (
                        f'<a class="pmid-link" href="https://pubmed.ncbi.nlm.nih.gov/{output["pmid"]}/" target="_blank">'
                        f'PMID {output["pmid"]}</a>'
                    )
                    card_html += f'<div class="citation-block">{output["evidence_text"]}</div>'
                elif output.get("reason"):
                    card_html += f'<div class="reason-text">{output["reason"]}</div>'

                card_html += '</div>'
                st.markdown(card_html, unsafe_allow_html=True)

                with st.expander("Retrieval + NLI details"):
                    for j, ev in enumerate(nli_results):
                        probs = ev["probabilities"]
                        st.markdown(f"**Chunk {j+1}** — PMID `{ev['pmid']}` — distance `{ev['distance']:.4f}`")
                        st.markdown(f"> {ev['text']}")
                        cols = st.columns(3)
                        cols[0].metric("Contradiction", f"{probs['contradiction']:.2%}")
                        cols[1].metric("Entailment", f"{probs['entailment']:.2%}")
                        cols[2].metric("Neutral", f"{probs['neutral']:.2%}")

    elif run_button:
        st.warning("Please enter some text to verify.")