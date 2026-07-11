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
  "claims":[
    "...",
    "..."
  ]
}}

- Each claim must contain exactly one independently verifiable fact.
- Keep dose and frequency together.
- Do not duplicate claims.
- Preserve wording.
- Do not infer facts not explicitly stated.
- Replace pronouns with the entity they refer to.

Example:

Sentence:
"Aspirin should be taken with food. It may cause stomach irritation."

Output:
{{
    "claims":[
        "Aspirin should be taken with food.",
        "Aspirin may cause stomach irritation."
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