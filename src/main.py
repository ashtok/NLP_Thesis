# main.py - Test your foundation

from pathlib import Path
from src.audio_utils import AudioLoader
from asr_models import SpeechRecognizer, LanguageIdentifier

# 1. Load audio
loader = AudioLoader(target_sr=16000)
waveform, sr = loader.load(Path("../data/hindi_audio/hindi_001.wav"))

# 2. Identify language
lid = LanguageIdentifier()
lang_probs = lid.identify(waveform, sr, top_k=5)
print("Language:", lang_probs)

# 3. Transcribe with beam search
recognizer = SpeechRecognizer(beam_size=10)
hypotheses = recognizer.transcribe(waveform, sr, n_best=5)
print("\nTop 5 hypotheses:")
for i, hyp in enumerate(hypotheses, 1):
    print(f"{i}. {hyp}")
