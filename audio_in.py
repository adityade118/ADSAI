import sounddevice as sd
import numpy as np
import queue
import threading
from scipy.io.wavfile import write
import io
import requests
import CnC  # import the pipeline module

# ---------------- Parameters ----------------
SERVER_URL = "https://floatiest-uninsurable-joellen.ngrok-free.dev/transcribe"
fs = 16000
frame_samples = 512
mic_index = 2
CHUNK_DURATION = 10

MAX_BUFFER_FRAMES = int(CHUNK_DURATION * fs / frame_samples)

# ---------------- Thread-safe queues ----------------
audio_queue = queue.Queue()
transcript_queue = queue.Queue()  # send to pipeline

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Status:", status)
    audio_queue.put(indata.copy())

# ---------------- Worker: Record -> Transcribe ----------------
def processing_thread():
    buffer = []
    transcript_counter = 1

    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)

        if len(buffer) >= MAX_BUFFER_FRAMES:
            audio = np.concatenate(buffer, axis=0).flatten()
            print(f"Processing {len(audio)/fs:.2f} sec audio for transcription...")

            def transcribe_chunk(audio, index):
                wav_bytes = io.BytesIO()
                write(wav_bytes, fs, (audio * 32767).astype(np.int16))
                wav_bytes.seek(0)
                try:
                    response = requests.post(
                        SERVER_URL,
                        files={"file": ("speech.wav", wav_bytes, "audio/wav")}
                    )
                    text = response.json().get("text", "")
                    print(f"[Whisper] Transcript {index}: {text}")

                    # Push transcript to pipeline queue
                    transcript_queue.put((index, text))

                except Exception as e:
                    print("Error transcribing chunk:", e)

            threading.Thread(target=transcribe_chunk, args=(audio, transcript_counter), daemon=True).start()
            transcript_counter += 1
            buffer.clear()

# ---------------- Start threads ----------------
threading.Thread(target=processing_thread, daemon=True).start()
threading.Thread(target=CnC.run_pipeline, args=(transcript_queue,), daemon=True).start()

# ---------------- Start audio stream ----------------
with sd.InputStream(
    samplerate=fs,
    channels=1,
    dtype='float32',
    blocksize=frame_samples,
    callback=audio_callback,
    device=mic_index
):
    print(f"Recording audio in {CHUNK_DURATION}-second chunks (Ctrl+C to stop)...")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Stopped.")
