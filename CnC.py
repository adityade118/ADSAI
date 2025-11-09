# ------------------- CnC.py -------------------
import time
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# -------- Load environment and configure Gemini --------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use the faster but accurate Gemini model
GEMINI_MODEL = "models/gemini-2.5-flash"
gemini = genai.GenerativeModel(GEMINI_MODEL)


# -------- Session Management Class --------
class CnCSession:
    def __init__(self, question_id, question_text, model_answer, tags=None, subtags=None):
        self.question_id = question_id
        self.question_text = question_text
        self.model_answer = model_answer
        self.tags = tags or []
        self.subtags = subtags or []

        self.topics = []
        self.covered = set()
        self.uncovered = set()

        self.transcript = []
        self.followups = []
        self.start_time = time.time()
        self.end_time = None

        self._initialize_topics()

    # -------- Internal: Split model answer into conceptual points --------
    def _initialize_topics(self):
        prompt = f"""
        You are an expert interviewer.
        Break the following model answer into a numbered list of atomic conceptual points.
        Each point should be one self-contained idea that can be verified independently.

        Model answer:
        {self.model_answer}
        """
        try:
            response = gemini.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    candidate_count=1,
                ),
            )
            text = response.text.strip()
            self.topics = [line.split(".", 1)[-1].strip() for line in text.splitlines() if "." in line]
            self.uncovered = set(range(len(self.topics)))
            print(f"\n[Session {self.question_id}] Initialized {len(self.topics)} key points.")
        except Exception as e:
            print("[Gemini initialization error]", e)
            self.topics = []

    # -------- Helper: Gemini-based coverage classifier (binary, decisive) --------
    def _classify_coverage(self, bullet_text, full_answer):
        COVERAGE_PROMPT = f"""
        You are a strict evaluator determining if a candidate's answer covers a bullet point.

        Label only as "covered" or "uncovered" — never partial, uncertain, or maybe.

        Interpret natural speech variations:
        - "O of n log n" = "O(n log n)"
        - "ex" = "e^x"
        - "exponential of x" = "e^x"
        Focus on conceptual correctness, not phrasing.

        Respond only in valid JSON like:
        {{"status": "covered"}} or {{"status": "uncovered"}}.

        Bullet: "{bullet_text}"
        Candidate answer:
        \"\"\"{full_answer}\"\"\"
        """

        try:
            response = gemini.generate_content(
                COVERAGE_PROMPT,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,     # deterministic
                    top_p=0.8,
                    candidate_count=1,
                ),
            )
            text = response.text.strip()
            if "{" in text and "}" in text:
                json_part = text[text.index("{") : text.rindex("}") + 1]
                data = json.loads(json_part)
                status = data.get("status", "uncovered").lower()
            else:
                status = "uncovered"
        except Exception as e:
            print("[Gemini coverage error]", e)
            status = "uncovered"

        return status

    # -------- Update coverage with new transcript chunk --------
    def update_with_transcript(self, chunk_text):
        self.transcript.append(chunk_text)
        full_answer = " ".join(self.transcript)

        # Re-evaluate coverage only for still-uncovered topics
        for i, topic in enumerate(self.topics):
            if i in self.covered:
                continue
            status = self._classify_coverage(topic, full_answer)
            if status == "covered":
                self.covered.add(i)
                self.uncovered.discard(i)
            else:
                self.uncovered.add(i)

        return {
            "covered": [self.topics[i] for i in self.covered],
            "uncovered": [self.topics[i] for i in self.uncovered],
        }

    # -------- Suggest one follow-up question --------
    def generate_followup(self):
        if not self.uncovered:
            return None

        uncovered_points = [self.topics[i] for i in self.uncovered]
        prompt = f"""
        These ideas have not been discussed yet:
        {uncovered_points}

        Ask ONE concise, natural follow-up question to help the candidate
        cover one of the above points. Keep it conversational and short.
        """

        try:
            response = gemini.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Slight creativity for follow-up phrasing
                    top_p=0.9,
                ),
            )
            question = response.text.strip()
            self.followups.append(question)
            self.transcript.append(f"Follow-up asked: {question}")
            return question
        except Exception as e:
            print("[Gemini follow-up error]", e)
            return None

    # -------- Compute final session score --------
    def finalize(self):
        self.end_time = time.time()
        score = 100 * len(self.covered) / len(self.topics) if self.topics else 0

        report = {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "tags": self.tags,
            "subtags": self.subtags,
            "score": round(score, 2),
            "covered_points": [self.topics[i] for i in self.covered],
            "missed_points": [self.topics[i] for i in self.uncovered],
            "followups": self.followups,
            "transcript": self.transcript,
            "duration_sec": round(self.end_time - self.start_time, 2),
        }

        print(f"\n[Session {self.question_id}] Final Score: {score:.2f}%")
        return report


# -------- Helper: Store session results --------
def save_session_report(report, path="session_history.json"):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                history = json.load(f)
        else:
            history = []
        history.append(report)
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[Saved] Session {report['question_id']} -> {path}")
    except Exception as e:
        print("[Error saving report]", e)
