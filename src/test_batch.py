# test_batch.py - Run agent on all audio files (UPDATED)

import json
from pathlib import Path
from datetime import datetime
from qwen_agent import QwenAgent


def test_batch():
    """Test agent on all audio files and save results"""

    # Setup
    audio_dir = Path("../data/hindi_audio")
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    # Initialize agent WITH WHISPER
    agent = QwenAgent(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        panlex_path=Path("../data/panlex.csv"),
        load_in_8bit=False,
        load_whisper=True,  # ✅ Enable Whisper
        whisper_model="small"  # ✅ Small model for speed
    )

    # Get all audio files
    audio_files = sorted(audio_dir.glob("*.wav"))  # Sort for consistency
    print(f"Found {len(audio_files)} audio files")
    print(f"Starting batch processing with Qwen 3B + Whisper...")

    results = []

    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing {i}/{len(audio_files)}: {audio_path.name}")
        print(f"{'=' * 80}")

        try:
            # Run agent
            result = agent.run(
                audio_path=audio_path,
                max_iterations=15,  # ✅ Increased from 10 (agent needs more steps for 2 ASR systems)
                verbose=False  # ✅ Set to False for cleaner output (change to True if debugging)
            )

            # Save result
            results.append({
                "file": audio_path.name,
                "language": result.get("language", ""),
                "transcription": result.get("transcription", ""),
                "confidence": result.get("confidence", ""),
                "reasoning": result.get("reasoning", ""),
                "success": True
            })

            # Print summary for this file
            print(f"✓ Success")
            print(f"  Language: {result.get('language', 'N/A')}")
            print(f"  Transcription: {result.get('transcription', 'N/A')[:80]}...")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "file": audio_path.name,
                "error": str(e),
                "success": False
            })

        # Save intermediate results every 10 files (in case of crash)
        if i % 10 == 0:
            temp_file = results_dir / f"temp_results_{i}.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Intermediate save: {temp_file}")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"qwen3b_whisper_results_{timestamp}.json"  # ✅ Renamed to indicate Whisper

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'=' * 80}")

    # Summary
    successful = sum(1 for r in results if r.get("success", False))
    failed = [r for r in results if not r.get("success", False)]
    empty_transcriptions = sum(1 for r in results if r.get("success") and not r.get("transcription", "").strip())

    print(f"\nSummary:")
    print(f"  Total files: {len(results)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ❌ Failed: {len(results) - successful}")
    print(f"  ⚠️  Empty transcriptions: {empty_transcriptions}")

    if failed:
        print(f"\nFailed files:")
        for r in failed[:5]:
            print(f"  - {r['file']}: {r.get('error', 'Unknown')[:60]}...")

    if empty_transcriptions > 0:
        print(f"\n⚠️  Warning: {empty_transcriptions} files have empty transcriptions!")
        print(f"   This might indicate the agent is not calling ASR tools properly.")


if __name__ == "__main__":
    test_batch()
