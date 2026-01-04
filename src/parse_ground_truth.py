from pathlib import Path
import json


def parse_transcriptions_txt(txt_file: Path, output_file: Path):
    """Parse transcriptions.txt into ground_truth.json"""

    ground_truth = []

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split by first whitespace
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print(f"⚠️  Skipping malformed line: {line}")
                continue

            file_name = parts[0]
            transcription = parts[1]

            ground_truth.append({
                "file": file_name,
                "transcription": transcription,
                "language": "hin"
            })

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"✓ Parsed {len(ground_truth)} transcriptions")
    print(f"  Input:  {txt_file}")
    print(f"  Output: {output_file}")
    print(f"\nFirst 3 entries:")
    for item in ground_truth[:3]:
        print(f"  {item['file']}: {item['transcription'][:50]}...")

    return ground_truth


if __name__ == "__main__":
    txt_file = Path("../data/hindi_audio/transcriptions.txt")
    output_file = Path("../data/ground_truth.json")

    if not txt_file.exists():
        print(f"❌ Error: {txt_file} not found!")
        exit(1)

    parse_transcriptions_txt(txt_file, output_file)
