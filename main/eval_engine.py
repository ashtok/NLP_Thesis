# main/eval_engine.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from audio_loader import HFAudioLoader
from asr_whisper_baseline import run_whisper_baseline
from asr_mms_1b_baseline_with_lang import run_mms_baseline
from asr_mms_zeroshot_baseline import (
    run_mms_zeroshot_baseline_basic,
    ASR_SAMPLING_RATE,
)


def evaluate_model(config: Dict[str, Any]) -> Dict[str, Any]:
    backend = config["backend"]
    model_name = config["model_name"]
    data_root = Path(config["data_root"])
    transcription_file = config["transcription_file"]
    target_lang = config.get("target_lang")
    language = config.get("language", "hi")

    base_dir_abs = str(data_root)
    trans_path = str(data_root / transcription_file)

    # Common dataset (Devanagari refs from transcriptions.txt)
    loader = HFAudioLoader(target_sr=16_000)
    ds = loader.from_dir_with_text(base_dir_abs, trans_path)

    if backend == "whisper":
        result = run_whisper_baseline(
            loader=loader,
            ds=ds,
            model_name=model_name,
            language=language,
        )

    elif backend == "mms":
        result = run_mms_baseline(
            loader=loader,
            ds=ds,
            model_id=model_name,
            target_lang=target_lang or "hin",
        )

    elif backend == "omni":
        # Import Omni only when needed
        from asr_omni_baseline import run_omni_baseline

        result = run_omni_baseline(
            loader=loader,
            ds=ds,
            model_card=model_name,
            lang_tag=config.get("lang_tag", "hin_Deva"),
        )

    elif backend == "mms_zeroshot":
        # For MMS-ZS we want:
        # - audio + original Devanagari ds from transcriptions.txt
        # - romanized refs from transcriptions_uroman.txt
        # so ignore transcription_file here and load the uroman refs separately.
        roman_path = data_root / "transcriptions_uroman.txt"
        with roman_path.open("r", encoding="utf-8") as f:
            refs_roman = [ln.rstrip("\n") for ln in f]

        # Rebuild loader/dataset with the audio + Devanagari file explicitly,
        # in case config["transcription_file"] was pointing to the uroman file.
        loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
        ds = loader.from_dir_with_text(
            str(data_root),
            str(data_root / "transcriptions.txt"),
        )

        result = run_mms_zeroshot_baseline_basic(
            loader=loader,
            ds=ds,
            refs_roman=refs_roman,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    result["data_root"] = str(data_root)
    result["transcription_file"] = transcription_file
    return result
