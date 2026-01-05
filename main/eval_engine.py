# main/eval_engine.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from audio_loader import HFAudioLoader
from asr_whisper_baseline import run_whisper_baseline
from asr_mms_1b_baseline_with_lang import run_mms_baseline
from asr_omni_baseline import run_omni_baseline



def evaluate_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic entry point to evaluate one ASR model on one dataset.
    """
    backend = config["backend"]
    model_name = config["model_name"]
    data_root = Path(config["data_root"])
    transcription_file = config["transcription_file"]
    target_lang = config.get("target_lang")
    language = config.get("language", "hi")  # for Whisper

    base_dir_abs = str(data_root)
    trans_path = str(data_root / transcription_file)

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
        result = run_omni_baseline(
            loader=loader,
            ds=ds,
            model_card=model_name,  # e.g. "omniASR_CTC_300M"
            lang_tag=config.get("lang_tag", "hin_Deva"),
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    result["data_root"] = str(data_root)
    result["transcription_file"] = transcription_file

    return result
