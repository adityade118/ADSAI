import sounddevice as sd
import numpy as np

# List all audio devices
print(sd.query_devices())

# Set the device index for recording
mic_index = 2  # your Microphone Array device

fs = 16000          # sample rate
frame_samples = 512  # number of samples per chunk
duration = 5
# Record a short test clip
print("Talk:")
audio = sd.rec(int(duration*fs), samplerate=fs, channels=1, dtype='float32', device=mic_index)
sd.wait()

print("Recorded audio shape:", audio.shape)

import numpy as np

# Check some basic stats
print("Min:", np.min(audio))
print("Max:", np.max(audio))
print("Mean:", np.mean(audio))
print("Std:", np.std(audio))

sd.play(audio, fs)
sd.wait()
