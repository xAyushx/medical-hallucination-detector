import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"


PROMPT = """
You are a medical claim extraction system.
Extract all factual claims from the paragraph.
Rules:
- Return ONLY valid JSON.
- Return exactly this format:
{{
  "claims": [
    "...",
    "..."
  ]
}}
- Each claim must be atomic, independently verifiable, and self-contained.
- Each claim must express one complete factual statement, not a fragment or keyword phrase.
- Keep all explicit qualifiers attached to the claim: negation, dose, frequency, numeric values, time, comparison, caution, and causal language.
- Do not duplicate the same claim in different words.
- Do not infer facts that are not explicitly stated.
- Do not add prerequisite or logically implied claims that are not written in the paragraph.
- Replace pronouns with the entity they refer to when the antecedent is explicit in the same paragraph.
- Preserve the original meaning as closely as possible.
- Do not turn one sentence into a shorter but weaker paraphrase that loses information.
- For contrast structures, extract each explicitly stated clause separately:
  - "not X, but Y" -> one claim for "not X" and one claim for "Y"
  - "X, however Y" -> one claim for "X" and one claim for "Y" if both are explicitly stated
  - "although X, Y" -> one claim for "X" only if X is explicitly asserted, and one claim for "Y" only if Y is explicitly asserted
- Never invent a claim like "X exists" or "X is activated" just because another clause mentions X, describes X, or names X as part of a larger statement. A condition, drug, or entity being the SUBJECT or OBJECT of a sentence is not itself a separate fact — only extract what is actually asserted about it.
- When a sentence has the shape "[X] is the primary/standard/foundational treatment for [condition]", extract exactly ONE claim describing that relationship. Do NOT extract a second claim asserting that the condition itself "exists" or is a recognized disease — that is never explicitly asserted, it is only referenced.
- Never split one factual clause into multiple overlapping claims.
Example 1:
Sentence:
"Aspirin should be taken with food. It may cause stomach irritation."
Output:
{{
  "claims": [
    "Aspirin should be taken with food.",
    "Aspirin may cause stomach irritation."
  ]
}}
Example 2:
Sentence:
"Asthma is not a chronic respiratory disease but affects over 230 million people worldwide."
Output:
{{
  "claims": [
    "Asthma is not a chronic respiratory disease.",
    "Asthma affects over 230 million people worldwide."
  ]
}}
Example 3:
Sentence:
"Metformin is typically started at a dose of 500 mg twice daily and works by reducing hepatic glucose production."
Output:
{{
  "claims": [
    "Metformin is typically started at a dose of 500 mg twice daily.",
    "Metformin works by reducing hepatic glucose production."
  ]
}}
Example 4:
Sentence:
"EA also downregulated activation of the STAT3/HIF-1α signalling pathway."
Output:
{{
  "claims": [
    "EA also downregulated activation of the STAT3/HIF-1α signalling pathway."
  ]
}}
Example 5:
Sentence:
"Inhaled corticosteroids are the foundational daily maintenance therapy for persistent asthma."
Output:
{{
  "claims": [
    "Inhaled corticosteroids are the foundational daily maintenance therapy for persistent asthma."
  ]
}}
Paragraph:
{paragraph}
"""


def extract_claims(paragraph):

    prompt = PROMPT.format(paragraph=paragraph)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    claims = json.loads(result["response"])["claims"]

    return claims