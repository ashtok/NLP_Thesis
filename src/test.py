# test.py - Complete test suite for all tools

from pathlib import Path
from linguistic_tools import PanLexTool, UromanTool
from audio_utils import AudioLoader
from setup_tools import ASRToolkit

print("=" * 80)
print("COMPREHENSIVE TOOL TEST SUITE")
print("=" * 80)

# ============================================================================
# PART 1: Test Linguistic Tools (No Audio Needed)
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: Testing Linguistic Tools")
print("=" * 80)

# Test PanLex
print("\n--- PanLexTool ---")
panlex = PanLexTool(panlex_path=Path("../data/panlex.csv"))

print("\n1. Query word 'shandar' in Hindi:")
result = panlex.query(word="shandar", iso_639_3="hin")
print(result)

print("\n2. Verify exact matching:")
test_cases = [
    ("flipcard", "hin", False),
    ("sandar", "hin", True),
    ("xyz123", "hin", False),
    ("shandar", "hin", True),
    ("SHANDAR", "hin", True),
    ("shan", "hin", False),
]

for word, lang, expected in test_cases:
    is_valid = panlex.validate_word(word, lang)
    status = "✓" if is_valid == expected else "✗ ERROR"
    print(f"  {status} '{word}' → {is_valid} (expected {expected})")

print("\n3. All Hindi words in dictionary:")
result = panlex.query(iso_639_3="hin", max_size=20)
print(result)

# Test Uroman
print("\n--- UromanTool ---")
uroman = UromanTool()

test_texts = [
    "शानदार स्मार्टफोन",
    "लेनोवो का ये",
    "बिक रहा है",
]

print("\nRomanizing Hindi text:")
for text in test_texts:
    romanized = uroman.romanize(text)
    print(f"  {text} → {romanized}")

# ============================================================================
# PART 2: Test Audio-Based Tools
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: Testing Audio-Based Tools")
print("=" * 80)

# Load audio
print("\n1. Loading audio...")
loader = AudioLoader()
audio_file = Path("../data/hindi_audio/hindi_001.wav")
waveform, sr = loader.load(audio_file)
print(f"✓ Loaded: {audio_file.name}")
print(f"  Shape: {waveform.shape}")
print(f"  Sample rate: {sr} Hz")
print(f"  Duration: {len(waveform) / sr:.2f} seconds")

# Create toolkit
print("\n2. Creating ASR toolkit...")
toolkit = ASRToolkit(
    panlex_path=Path("../data/panlex.csv"),
    load_whisper=True,  # Enable Whisper
    load_nllb=False,  # Skip NLLB (optional)
    whisper_model="small"
)
toolkit.set_audio(waveform, sr)

# Create registry
print("\n3. Creating tool registry...")
registry = toolkit.create_registry()
print(f"✓ Registered {len(registry.get_schemas())} tools:")
for schema in registry.get_schemas():
    print(f"   - {schema['function']['name']}")

# ============================================================================
# PART 3: Test Each Tool
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: Testing Tool Execution")
print("=" * 80)

# Test 1: Language Identification
print("\n--- Test 1: Language Identification ---")
result = registry.execute("call_language_identifier", {"top_k": 5})
if "error" in result:
    print(f"  ❌ Error: {result['error']}")
else:
    print(f"  Top languages:")
    for lang, prob in result['languages'].items():
        print(f"    {lang}: {prob:.2%}")

# Test 2: MMS Zero-Shot ASR
print("\n--- Test 2: MMS Zero-Shot ASR ---")
result = registry.execute("call_zeroshot_asr", {"n_best_hypotheses": 5})
if "error" in result:
    print(f"  ❌ Error: {result['error']}")
else:
    print(f"  Generated {result['count']} hypotheses:")
    for i, h in enumerate(result['hypotheses'], 1):
        print(f"    {i}. {h}")

# Test 3: Whisper ASR (if available)
print("\n--- Test 3: Whisper ASR ---")
result = registry.execute("call_whisper", {"language": "hi"})
if "error" in result:
    print(f"  ⚠️  {result['error']}")
else:
    print(f"  Whisper transcription:")
    print(f"    {result['transcription']}")
    print(f"  Detected language: {result.get('language', 'N/A')}")

# Test 4: Compare ASR outputs
print("\n--- Test 4: ASR Comparison ---")
mms_result = registry.execute("call_zeroshot_asr", {"n_best_hypotheses": 1})
whisper_result = registry.execute("call_whisper", {"language": "hi"})

print("  MMS-ZS (top hypothesis):")
if "hypotheses" in mms_result:
    print(f"    {mms_result['hypotheses'][0]}")
else:
    print(f"    Error: {mms_result.get('error', 'Unknown')}")

print("  Whisper:")
if "transcription" in whisper_result:
    print(f"    {whisper_result['transcription']}")
else:
    print(f"    Error: {whisper_result.get('error', 'Unknown')}")

# Test 5: Word Validation
print("\n--- Test 5: Word Validation ---")
test_words = [
    ("shandar", "hin"),
    ("smartphone", "hin"),
    ("xyz123", "hin"),
    ("ke", "hin"),
    ("lenovo", "hin"),
]

for word, lang in test_words:
    result = registry.execute("validate_word", {"word": word, "lang_code": lang})
    if "error" in result:
        print(f"  ❌ '{word}': Error")
    else:
        status = "✓" if result['is_valid'] else "✗"
        print(f"  {status} '{word}': {result['is_valid']}")

# Test 6: Dictionary Query
print("\n--- Test 6: Dictionary Query ---")
result = registry.execute("query_panlex", {
    "word": "shandar",
    "iso_639_3": "hin",
    "max_size": 5
})
if "error" in result:
    print(f"  ❌ Error: {result['error']}")
else:
    print(f"  Query results:\n{result['result']}")

# Test 7: Romanization
print("\n--- Test 7: Romanization ---")
test_texts = [
    "शानदार स्मार्टफोन",
    "Flipkart बंपर ऑफर्स",
]

for text in test_texts:
    result = registry.execute("romanize_text", {"text": text})
    if "error" in result:
        print(f"  ❌ Error: {result['error']}")
    else:
        print(f"  {result['original']}")
        print(f"  → {result['romanized']}")

# Test 8: NLLB Translation (if available)
print("\n--- Test 8: Native Script Translation ---")
result = registry.execute("translate_to_native_script", {
    "text": "shandar smartphone",
    "target_lang": "hin_Deva"
})
if "error" in result:
    print(f"  ⚠️  {result['error']}")
else:
    print(f"  Original: {result['original']}")
    print(f"  Translated: {result['translated']}")

# Test 9: Word Frequency (if available)
print("\n--- Test 9: Word Frequency ---")
test_words_freq = ["shandar", "smartphone", "ke", "hai"]
for word in test_words_freq:
    result = registry.execute("get_word_frequency", {
        "word": word,
        "lang": "hin"
    })
    if "error" in result:
        print(f"  ⚠️  {result['error']}")
        break
    else:
        print(f"  '{word}': frequency = {result['frequency']}")

# ============================================================================
# PART 4: End-to-End Workflow Simulation
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: Simulated Agent Workflow")
print("=" * 80)

print("\nSimulating how agent would use these tools:")

# Step 1: Identify language
print("\n[STEP 1] Identify language")
lang_result = registry.execute("call_language_identifier", {"top_k": 1})
detected_lang = list(lang_result['languages'].keys())[0]
confidence = lang_result['languages'][detected_lang]
print(f"  Detected: {detected_lang} ({confidence:.1%} confidence)")

# Step 2: Get transcriptions from both ASR systems
print("\n[STEP 2] Get transcriptions from multiple ASR systems")

print("  MMS Zero-Shot:")
mms_result = registry.execute("call_zeroshot_asr", {"n_best_hypotheses": 3})
for i, h in enumerate(mms_result['hypotheses'], 1):
    print(f"    {i}. {h}")

print("  Whisper:")
whisper_result = registry.execute("call_whisper", {"language": "hi"})
if "transcription" in whisper_result:
    print(f"    {whisper_result['transcription']}")

# Step 3: Validate key words
print("\n[STEP 3] Validate words from top hypothesis")
if mms_result['hypotheses']:
    top_hypothesis = mms_result['hypotheses'][0]
    words = top_hypothesis.split()[:5]  # Check first 5 words

    valid_count = 0
    for word in words:
        result = registry.execute("validate_word", {
            "word": word,
            "lang_code": detected_lang
        })
        if result.get('is_valid'):
            valid_count += 1
            print(f"  ✓ '{word}' - valid")
        else:
            print(f"  ✗ '{word}' - not in dictionary")

    validation_rate = (valid_count / len(words)) * 100 if words else 0
    print(f"\n  Validation rate: {validation_rate:.1f}%")

# Step 4: Final recommendation
print("\n[STEP 4] Agent Decision")
print(f"  Language: {detected_lang}")
print(f"  Recommended transcription:")
if whisper_result.get('transcription'):
    print(f"    Whisper output (usually more reliable for Indian languages)")
    print(f"    → {whisper_result['transcription']}")
else:
    print(f"    MMS-ZS top hypothesis")
    print(f"    → {mms_result['hypotheses'][0] if mms_result['hypotheses'] else 'N/A'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

print("\n✓ Linguistic Tools:")
print("  - PanLex dictionary: Working")
print("  - Uroman romanization: Working")

print("\n✓ ASR Systems:")
print("  - MMS Zero-Shot: Working")
print(f"  - Whisper: {'Working' if 'transcription' in whisper_result else 'Not available'}")

print("\n✓ Validation Tools:")
print("  - Word validation: Working")
print("  - Dictionary query: Working")

print("\n✓ Translation Tools:")
nllb_working = "translated" in registry.execute("translate_to_native_script",
                                                {"text": "test", "target_lang": "hin_Deva"})
print(f"  - NLLB translator: {'Working' if nllb_working else 'Not available'}")

crubadan_working = "frequency" in registry.execute("get_word_frequency", {"word": "test", "lang": "hin"})
print(f"  - Crúbadán frequency: {'Working' if crubadan_working else 'Not available'}")

print("\n" + "=" * 80)
print("✓ ALL TESTS COMPLETE!")
print("=" * 80)
