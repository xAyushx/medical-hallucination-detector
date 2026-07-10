import requests
import xml.etree.ElementTree as ET
import time
import json
import re
import pysbd
from pathlib import Path
from collections import Counter

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

CONTACT_EMAIL = "ayushnr09@gmail.com"

CONDITIONS = {
    "cardiovascular_diseases": '"cardiovascular diseases"[MeSH Terms]',
    "diabetes": '"diabetes mellitus"[MeSH Terms]',
    "asthma": '"asthma"[MeSH Terms]',
}
MIN_CHUNK_LENGTH = 10
RETMAX_PER_CONDITION = 150
RELDATE_DAYS = 3650
BATCH_SIZE = 50
REQUEST_DELAY = 0.4

_segmenter = pysbd.Segmenter(language="en", clean=False)

CITE_TAG_PATTERN = re.compile(r"\[cite:\s*\d+\]")


def clean_text(text: str) -> str:
    """Strip publisher-side citation artifacts and collapse resulting
    extra whitespace."""
    if not text:
        return ""
    cleaned = CITE_TAG_PATTERN.sub('', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """
    Sentence segmentation via pysbd — correctly handles decimals,
    abbreviations, statistical notation, and genus abbreviations
    out of the box.
    """
    sentences = _segmenter.segment(text)
    return [s.strip() for s in sentences if s.strip()]


def esearch(query: str, retmax: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": f"{query} AND treatment",
        "retmax": retmax,
        "retmode": "json",
        "datetype": "pdat",
        "reldate": RELDATE_DAYS,
        "tool": "medical-hallucination-detector",
        "email": CONTACT_EMAIL,
    }
    r = requests.get(ESEARCH_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def efetch_batch_xml(pmid_batch: list[str]) -> str:
    params = {
        "db": "pubmed",
        "id": ",".join(pmid_batch),
        "retmode": "xml",
        "tool": "medical-hallucination-detector",
        "email": CONTACT_EMAIL,
    }
    r = requests.get(EFETCH_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.text


def parse_records(xml_text: str, condition: str) -> list[dict]:
    records = []
    root = ET.fromstring(xml_text)

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or pmid_el.text is None:
            continue
        pmid = pmid_el.text.strip()

        # English-only filter
        lang_el = article.find(".//Language")
        lang = (lang_el.text or "").strip().lower() if lang_el is not None else ""
        if lang and lang != "eng":
            continue

        title_el = article.find(".//ArticleTitle")
        raw_title = title_el.text if title_el is not None else "Untitled"
        title = clean_text(raw_title) if raw_title else "Untitled"

        abstract_els = article.findall(".//AbstractText")
        if not abstract_els:
            continue

        raw_abstract = " ".join(el.text.strip() for el in abstract_els if el.text)
        abstract = clean_text(raw_abstract)
        if not abstract:
            continue

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "condition": condition,
        })

    return records


def fetch_condition(condition: str, query: str, retmax: int) -> list[dict]:
    print(f"\n[{condition}] Searching PubMed...")
    pmids = esearch(query, retmax)
    print(f"[{condition}] Found {len(pmids)} PMIDs")

    all_records = []
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i:i + BATCH_SIZE]
        print(f"[{condition}] Fetching batch {i // BATCH_SIZE + 1} ({len(batch)} PMIDs)...")
        xml_text = efetch_batch_xml(batch)
        batch_records = parse_records(xml_text, condition)
        print(f"[{condition}]    -> {len(batch_records)}/{len(batch)} had usable English abstracts")
        all_records.extend(batch_records)
        time.sleep(REQUEST_DELAY)

    return all_records


def build_sentence_chunks(records: list[dict]) -> list[dict]:
    chunks = []
    skipped_short = 0
    for r in records:
        sentences = split_sentences(r["abstract"])
        for idx, sentence in enumerate(sentences):
            if len(sentence.strip()) < MIN_CHUNK_LENGTH:
                skipped_short += 1
                continue
            chunks.append({
                "chunk_id": f"{r['pmid']}_{idx}",
                "pmid": r["pmid"],
                "title": r["title"][:150],
                "condition": r["condition"],
                "chunk_text": sentence,
            })
    print(f"Skipped {skipped_short} chunks under {MIN_CHUNK_LENGTH} chars")
    return chunks

def main():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    all_records = []
    for condition, query in CONDITIONS.items():
        records = fetch_condition(condition, query, RETMAX_PER_CONDITION)
        all_records.extend(records)

    print(f"\nTotal abstracts collected: {len(all_records)}")

    raw_path = data_dir / "pubmed_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved raw abstracts to {raw_path}")

    chunks = build_sentence_chunks(all_records)
    chunks_path = data_dir / "pubmed_chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} sentence-level chunks to {chunks_path}")

    condition_counts = Counter(r["condition"] for r in all_records)
    print("\nAbstracts per condition:")
    for cond, count in condition_counts.items():
        print(f"  {cond}: {count}")

    chunk_counts = Counter(c["condition"] for c in chunks)
    print("\nSentence chunks per condition:")
    for cond, count in chunk_counts.items():
        print(f"  {cond}: {count}")


if __name__ == "__main__":
    main()