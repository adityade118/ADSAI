# ------------------- CnC.py -------------------
import time
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import faiss

# -------- Setup Gemini --------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "models/gemini-1.5-pro-latest"
embedder = SentenceTransformer("all-mpnet-base-v2")

# -------- Model answer structure --------
MODEL_ANSWER = [
    {"id": "b1", "text": "Quicksort average-case is O(n log n)"},
    {"id": "b2", "text": "Quicksort worst-case is O(n^2)"},
    {"id": "b3", "text": "Pivot choice determines partition quality"},
]

T_PRESENT = 0.75
T_PARTIAL = 0.5

# -------- Gemini prompt for extracting claims --------
CLAIM_PROMPT = """
Extract key factual or conceptual claims from this transcript as JSON list.
Each claim must have:
- claim_text
- entities
- predicate
Transcript:
\"\"\"{text}\"\"\"
Return JSON only.
"""

def call_gemini(prompt):
    try:
        r = genai.generate_content(
            model=GEMINI_MODEL,
            contents=[{"role": "user", "parts":[{"text": prompt}]}]
        )
        return r.text
    except Exception as e:
        print("Gemini error:", e)
        return "[]"

def extract_claims(text):
    prompt = CLAIM_PROMPT.format(text=text)
    out = call_gemini(prompt)
    try:
        claims = json.loads(out)
    except Exception:
        claims = [{"claim_text": s.strip(), "entities": [], "predicate": ""} for s in text.split('.') if s.strip()]
    return claims

def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True)

def map_claims(claims, bullets):
    bullet_texts = [b["text"] for b in bullets]
    bullet_emb = embed_texts(bullet_texts)
    claim_texts = [c["claim_text"] for c in claims]
    claim_emb = embed_texts(claim_texts)

    d = bullet_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(bullet_emb)
    index.add(bullet_emb)
    faiss.normalize_L2(claim_emb)
    D, I = index.search(claim_emb, 1)

    mappings = []
    for i, c in enumerate(claims):
        mappings.append({
            "claim": c,
            "matched_bullet": bullets[int(I[i][0])],
            "score": float(D[i][0])
        })
    return mappings

def compute_coverage(mappings, bullets):
    best = {b["id"]: 0.0 for b in bullets}
    for m in mappings:
        bid = m["matched_bullet"]["id"]
        best[bid] = max(best[bid], m["score"])
    covered = sum(v >= T_PRESENT for v in best.values())
    partial = sum(T_PARTIAL <= v < T_PRESENT for v in best.values())
    total = len(bullets)
    return {
        "coverage": covered / total,
        "details": best,
        "covered": covered,
        "partial": partial
    }

def generate_followups(mappings, coverage_info, bullets):
    #TODO###
    followups = []
    # incomplete parts
    for b in bullets:
        if coverage_info["details"][b["id"]] < T_PRESENT:
            followups.append(f"You mentioned some parts, could you also explain: '{b['text']}'?")
    # wrong/uncertain claims
    for m in mappings:
        if m["score"] < 0.6:
            followups.append(f"You said '{m['claim']['claim_text']}'. Could you clarify what you meant?")
    return followups[:2]  # top 2 followups

def run_pipeline(transcript_queue):
    """Continuously consume transcripts and generate follow-ups."""
    buffer = []
    last_time = time.time()

    while True:
        index, text = transcript_queue.get()
        buffer.append(text)

        # Detect “pause” — no transcript for ~20s (or end of answer)
        now = time.time()
        if now - last_time > 20 or len(buffer) >= 3:  # roughly 30s of speech
            utterance = " ".join(buffer)
            print("\n[PIPELINE] Evaluating answer segment:")
            print(utterance)
            claims = extract_claims(utterance)
            mappings = map_claims(claims, MODEL_ANSWER)
            coverage = compute_coverage(mappings, MODEL_ANSWER)
            followups = generate_followups(mappings, coverage, MODEL_ANSWER)
            print("[PIPELINE] Follow-ups:")
            for f in followups:
                print(" →", f)
            buffer.clear()
        last_time = now
