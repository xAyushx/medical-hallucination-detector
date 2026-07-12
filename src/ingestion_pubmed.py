import requests
import xml.etree.ElementTree as ET
import time
import json
import re
import pysbd
from pathlib import Path
from collections import Counter, defaultdict

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

CONTACT_EMAIL = "ayushnr09@gmail.com"

CONDITIONS = {
    "diabetes": '"diabetes mellitus"[MeSH Terms]',
    "hypertension": '"hypertension"[MeSH Terms]',
    "heart_failure": '"heart failure"[MeSH Terms]',
    "coronary_artery_disease": '"coronary artery disease"[MeSH Terms]',
    "stroke": '"stroke"[MeSH Terms]',
    "asthma": '"asthma"[MeSH Terms]',
    "copd": '"pulmonary disease, chronic obstructive"[MeSH Terms]'
}

MIN_WORDS = 8
RETMAX_PER_CONDITION = 2000
REVIEW_RETMAX_PER_CONDITION = 300
RELDATE_DAYS = 9125  # ~25 years
BATCH_SIZE = 100
REQUEST_DELAY = 0.4
MAX_RETRIES = 3

_segmenter = pysbd.Segmenter(language="en", clean=False)
CITE_TAG_PATTERN = re.compile(r"\[cite:\s*\d+\]")


def get_full_text(element) -> str:
    """
    Extract ALL text from an XML element, including text inside and
    after nested inline tags (e.g. <sub>, <sup>, <i>, <b>) — plain
    .text only captures text before the FIRST nested child, silently
    dropping content in chemical formulas (ECCO2R), isotopes (18F-FDG),
    species names (E. coli), and similar. Verified against multiple
    real-world-style cases before integrating.
    """
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = CITE_TAG_PATTERN.sub('', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = re.sub(r'\([^)]*funding[^)]*\)', '', cleaned, flags=re.I)
    cleaned = re.sub(r'\([^)]*supplementary[^)]*\)', '', cleaned, flags=re.I)
    cleaned = re.sub(r'Author information.*', '', cleaned)
    cleaned = re.sub(r'Copyright.*', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r"\[[0-9,\-\s]+\]", "", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    sentences = _segmenter.segment(text)
    return [s.strip() for s in sentences if s.strip()]


def _get_with_retry(url: str, params: dict, timeout: int) -> requests.Response:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = 1.5 * (attempt + 1)
            print(f"    request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_err}")


def esearch(query: str, retmax: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": f"({query}) AND hasabstract[text] AND english[lang]",
        "retmax": retmax,
        "retmode": "json",
        "datetype": "pdat",
        "reldate": RELDATE_DAYS,
        "tool": "medical-hallucination-detector",
        "email": CONTACT_EMAIL,
    }
    r = _get_with_retry(ESEARCH_URL, params, timeout=15)
    return r.json().get("esearchresult", {}).get("idlist", [])


def efetch_batch_xml(pmid_batch: list[str]) -> str:
    params = {
        "db": "pubmed",
        "id": ",".join(pmid_batch),
        "retmode": "xml",
        "tool": "medical-hallucination-detector",
        "email": CONTACT_EMAIL,
    }
    r = _get_with_retry(EFETCH_URL, params, timeout=30)
    return r.text


def parse_records(xml_text: str, condition: str) -> list[dict]:
    records = []
    root = ET.fromstring(xml_text)

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or pmid_el.text is None:
            continue
        pmid = pmid_el.text.strip()

        lang_el = article.find(".//Language")
        lang = (lang_el.text or "").strip().lower() if lang_el is not None else ""
        if lang and lang != "eng":
            continue

        title_el = article.find(".//ArticleTitle")
        raw_title = get_full_text(title_el) if title_el is not None else "Untitled"
        title = clean_text(raw_title) if raw_title else "Untitled"

        abstract_els = article.findall(".//AbstractText")
        if not abstract_els:
            continue

        raw_abstract = " ".join(get_full_text(el) for el in abstract_els if el is not None)
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


def fetch_condition_reviews(condition: str, query: str, retmax: int) -> list[dict]:
    review_query = (
    f"({query}) "
    "AND review[pt] "
    "AND hasabstract[text] "
    "AND english[lang]"
)
    print(f"\n[{condition}] Searching PubMed for REVIEW articles...")
    return fetch_condition(condition, review_query, retmax)


def deduplicate_and_merge_conditions(all_records: list[dict]) -> list[dict]:
    pmid_to_conditions = defaultdict(set)
    pmid_to_record = {}

    for r in all_records:
        pmid_to_conditions[r["pmid"]].add(r["condition"])
        if r["pmid"] not in pmid_to_record:
            pmid_to_record[r["pmid"]] = r

    unique_records = []
    for pmid, record in pmid_to_record.items():
        merged_conditions = ",".join(sorted(pmid_to_conditions[pmid]))
        record = dict(record)
        record["condition"] = merged_conditions
        unique_records.append(record)

    return unique_records


def build_sentence_chunks(records: list[dict]) -> list[dict]:
    chunks = []
    skipped_short = 0
    for r in records:
        sentences = split_sentences(r["abstract"])
        for idx, sentence in enumerate(sentences):
            if len(sentence.split()) < MIN_WORDS:
                skipped_short += 1
                continue
            chunks.append({
                "chunk_id": f"{r['pmid']}_{idx}",
                "pmid": r["pmid"],
                "title": r["title"][:150],
                "condition": r["condition"],
                "chunk_text": sentence,
            })
    print(f"Skipped {skipped_short} chunks under {MIN_WORDS} words")
    return chunks


def main():
    data_dir = Path(__file__).parent.parent / "data"  
    data_dir.mkdir(exist_ok=True)

    all_records = []

    for condition, query in CONDITIONS.items():
        records = fetch_condition(condition, query, RETMAX_PER_CONDITION)
        all_records.extend(records)

    for condition, query in CONDITIONS.items():
        review_records = fetch_condition_reviews(condition, query, REVIEW_RETMAX_PER_CONDITION)
        all_records.extend(review_records)

    print(f"\nTotal abstracts collected (pre-dedup, primary + reviews): {len(all_records)}")

    all_records = deduplicate_and_merge_conditions(all_records)

    print(f"Total abstracts after cross-condition/cross-pass dedup: {len(all_records)}")

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
    print("\nAbstracts per condition (merged tags):")
    for cond, count in condition_counts.items():
        print(f"  {cond}: {count}")

    chunk_counts = Counter(c["condition"] for c in chunks)
    print("\nSentence chunks per condition (merged tags):")
    for cond, count in chunk_counts.items():
        print(f"  {cond}: {count}")


if __name__ == "__main__":
    main()