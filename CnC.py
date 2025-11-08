# ------------------- CnC.py -------------------
import time
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# -------- Setup Gemini --------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "models/gemini-2.5-flash"

# -------- Model answer structure --------
MODEL_ANSWER = [
    {"id": "b1", "text": "Quicksort average-case is O(n log n)"},
    {"id": "b2", "text": "Quicksort worst-case is O(n^2)"},
    {"id": "b3", "text": "Pivot choice determines partition quality"},
]

# -------- Tunable thresholds --------
FOLLOWUP_COOLDOWN = 30  # seconds between follow-ups per bullet

# -------- Gemini prompt templates --------
UNCERTAINTY_PROMPT = """
You are analyzing a transcript segment from a technical interview.
Determine if the speaker seems confident, uncertain, or admits not knowing.

Return JSON with a single key "state" whose value is one of:
["knows", "uncertain", "does_not_know"].

Transcript:
\"\"\"{text}\"\"\"

Return JSON only.
"""

COVERAGE_PROMPT = """
You are evaluating whether the following bullet point has been fully addressed
in the candidate's complete answer.

Bullet point:
"{bullet}"

Candidate's full answer so far:
\"\"\"{answer}\"\"\"

Classify as one of: ["covered", "partial", "incomplete"].

You are judging whether each point has been covered by the candidate's answer. 
Mark as:
- COMPLETE: The idea is clearly or **implicitly covered, even with different wording**.
- PARTIAL: The idea is touched upon but lacks clarity or completeness.
- INCOMPLETE: The idea is missing or wrong.

Be forgiving to rephrasings like "O of n log n" vs "log-linear time".

Be tolerant of spoken variations like:
- "O of n square" instead of O(n^2)
- "split into halves" instead of "divide recursively"
- "exponential n" instead of "O(exp n)"

Examples:
Bullet: "Quicksort average-case is O(n log n)"
Answer: "Quicksort takes roughly n log n time on average" -> return complete
Bullet: "Quicksort worst-case is O(n^2)"
Answer: "It can degrade if the pivot is bad" -> partial
Bullet: "Pivot choice determines partition quality"
Answer: "Choosing the pivot carefully matters" -> complete

Return JSON: {{"status": "<one of covered/partial/incomplete>"}}.

"""

# ---------------- Gemini Helpers ----------------
def call_gemini(prompt):
    """Base Gemini API call with minimal error handling."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini error:", e)
        return "{}"

def detect_uncertainty_gemini(text):
    """Detect whether the speaker sounds confident, uncertain, or doesn't know."""
    prompt = UNCERTAINTY_PROMPT.format(text=text)
    out = call_gemini(prompt)
    try:
        result = json.loads(out)
        return result.get("state", "knows")
    except Exception:
        return "knows"

def classify_coverage_gemini(bullet_text, full_answer):
    """Gemini-based classification for completeness per bullet."""
    prompt = COVERAGE_PROMPT.format(bullet=bullet_text, answer=full_answer)
    out = call_gemini(prompt)
    try:
        data = json.loads(out)
        return data.get("status", "incomplete")
    except Exception:
        return "incomplete"

# --------- Global State Machines ---------
BULLET_STATE = {b["id"]: "uncovered" for b in MODEL_ANSWER}
LAST_FOLLOWUP_TIME = {b["id"]: 0 for b in MODEL_ANSWER}
FULL_TRANSCRIPT = []

# ---------------- State Management ----------------
def recompute_coverage_with_gemini(full_answer):
    """
    Evaluate all bullet points for completeness using the entire transcript.
    Directly updates BULLET_STATE.
    """
    summary = {}
    for b in MODEL_ANSWER:
        status = classify_coverage_gemini(b["text"], full_answer)
        BULLET_STATE[b["id"]] = status
        summary[b["id"]] = status
    return summary

def update_states(coverage_summary, last_followup, transcript):
    """
    Incorporate uncertainty cues for the most recent segment.
    """
    gem_state = detect_uncertainty_gemini(transcript)
    if last_followup:
        bid = last_followup["bullet_id"]
        if gem_state == "does_not_know":
            BULLET_STATE[bid] = "skipped"
        elif gem_state == "uncertain":
            BULLET_STATE[bid] = "pending"
        elif gem_state == "knows" and BULLET_STATE[bid] != "covered":
            BULLET_STATE[bid] = "uncovered"

def select_followup(coverage_summary, last_followup):
    """
    Select one relevant follow-up question per cycle, prioritizing partially
    covered points, enforcing cooldowns, and avoiding repetition.
    """
    now = time.time()
    candidates = []

    # Prioritize partially covered points
    for b in MODEL_ANSWER:
        bid = b["id"]
        if coverage_summary.get(bid) == "partial":
            candidates.append((bid, b))
    # Then incomplete ones
    for b in MODEL_ANSWER:
        bid = b["id"]
        if coverage_summary.get(bid) == "incomplete":
            candidates.append((bid, b))

    for bid, b in candidates:
        if BULLET_STATE[bid] in ["uncovered", "pending", "partial", "incomplete"]:
            if last_followup and last_followup["bullet_id"] == bid:
                continue  # skip repeating same follow-up
            if now - LAST_FOLLOWUP_TIME[bid] < FOLLOWUP_COOLDOWN:
                continue  # respect cooldown
            LAST_FOLLOWUP_TIME[bid] = now
            BULLET_STATE[bid] = "pending"
            return {
                "bullet_id": bid,
                "text": f"You haven’t clearly covered this point yet: '{b['text']}'. Could you elaborate?"
            }

    return None

# ---------------- Main Pipeline ----------------
def run_pipeline(transcript_queue):
    """
    Continuously consume interview transcript segments and
    generate context-aware follow-ups.
    """
    buffer = []
    last_followup = None
    last_time = time.time()

    while True:
        index, text = transcript_queue.get()
        buffer.append(text)
        FULL_TRANSCRIPT.append(text)

        now = time.time()
        # Evaluate periodically (every 20s or 3 utterances)
        if now - last_time > 20 or len(buffer) >= 3:
            utterance = " ".join(buffer)
            print("\n[PIPELINE] Evaluating segment:")
            print(utterance)

            # Full answer context
            full_answer = " ".join(FULL_TRANSCRIPT)

            # Step 1: Recompute completeness
            coverage_summary = recompute_coverage_with_gemini(full_answer)

            # Step 2: Update based on confidence
            update_states(coverage_summary, last_followup, utterance)

            # Step 3: Generate follow-up if needed
            followup = select_followup(coverage_summary, last_followup)
            if followup:
                print("[PIPELINE] Follow-up:")
                print("->>", followup["text"])
                last_followup = followup
                FULL_TRANSCRIPT.append(f"[FOLLOW-UP] {followup['text']}")
            else:
                print("[PIPELINE] No new follow-ups needed.")

            print("[STATE]", BULLET_STATE)
            buffer.clear()
        last_time = now
