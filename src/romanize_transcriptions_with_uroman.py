from __future__ import annotations

from pathlib import Path
import sys


def romanize_transcriptions_file(
    input_path: Path,
    output_path: Path,
    lang_code: str = "hin",
) -> None:
    """
    Romanize a UTF-8 transcription file line-by-line using uroman (Python).
    """
    # Point sys.path to the directory that contains uroman.py
    # For you: NLP_Thesis/uroman/uroman
    repo_root = Path(__file__).resolve().parent.parent
    uroman_dir = repo_root / "uroman" / "uroman"
    sys.path.insert(0, str(uroman_dir))

    import uroman as ur  # now resolves to NLP_Thesis/uroman/uroman/uroman.py

    u = ur.Uroman()  # loads data dir automatically[web:1]

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                fout.write("\n")
                continue

            rom = u.romanize_string(line, lcode=lang_code)  # e.g. 'hin'[web:1]
            fout.write(rom.strip() + "\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "hindi_audio"

    in_path = data_dir / "transcriptions.txt"
    out_path = data_dir / "transcriptions_uroman.txt"

    print(f"Romanizing {in_path} -> {out_path}")
    romanize_transcriptions_file(
        input_path=in_path,
        output_path=out_path,
        lang_code="hin",
    )
    print("Done.")


if __name__ == "__main__":
    main()
