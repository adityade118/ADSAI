from CnC import CnCSession, save_session_report

# Initialize session for one question
session = CnCSession(
    question_id="Q1",
    question_text="Explain the difference between processes and threads.",
    model_answer="""
    A process has its own memory and system resources.
    Threads are lightweight and share memory within the same process.
    Context switching between processes is slower than between threads.
    Threads can communicate easily because they share memory.
    """,
    tags=["Operating Systems"],
    subtags=["Concurrency", "Processes", "Threads"]
)

# Simulated candidate responses (transcripts)
segments = [
    "Processes have separate memory, while threads share memory within a process.",
]

# Simulate pipeline operation
for seg in segments:
    summary = session.update_with_transcript(seg)
    print("[Coverage update]", summary)
    followup = session.generate_followup()
    if followup:
        print("Follow-up:", followup)

# End of session
report = session.finalize()
save_session_report(report)
