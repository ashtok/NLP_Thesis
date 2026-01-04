from pathlib import Path
from qwen_agent import QwenAgent

print("="*80)
print("Testing Agent with Multiple ASR Systems")
print("="*80)

agent = QwenAgent(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    panlex_path=Path("../data/panlex.csv"),
    load_in_8bit=False
)

# Update ASRToolkit to load Whisper
agent.toolkit = ASRToolkit(
    panlex_path=Path("../data/panlex.csv"),
    load_whisper=True,
    whisper_model="small"
)
agent.registry = agent.toolkit.create_registry()

print("\nRunning agent on hindi_001.wav...")
result = agent.run(
    audio_path=Path("../data/hindi_audio/hindi_001.wav"),
    max_iterations=15,
    verbose=True
)

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)
print(f"Language: {result.get('language', 'N/A')}")
print(f"Transcription: {result.get('transcription', 'N/A')}")
print(f"Confidence: {result.get('confidence', 'N/A')}")
print(f"Reasoning: {result.get('reasoning', 'N/A')}")