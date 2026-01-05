from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import torch
from jiwer import wer, cer
from transformers import AutoProcessor, Wav2Vec2ForCTC

from audio_loader import HFAudioLoader


ASR_SAMPLING_RATE = 16_000
MODEL_ID = "mms-meta/mms-zeroshot-300m"  # basic zero-shot MMS model


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return torch.device("mps")
    else:
        return torch.device("cpu")


def run_mms_zeroshot_baseline_basic(
    loader: HFAudioLoader,
    ds: Any,
    refs_roman: List[str],
) -> Dict[str, float]:
    """
    Run MMS zero-shot 300M baseline with greedy CTC decoding (no lexicon/LM),
    and compute WER/CER in uroman space.

    Args:
        loader: HFAudioLoader (for symmetry; not used directly here).
        ds: Dataset with columns: audio, text (original Devanagari).
        refs_roman: Precomputed romanized references (uroman), one per sample.

    Returns:
        dict with model name, WER, CER, and n_samples.
    """
    print("[MMS-ZS-BASIC] Loading MMS zero-shot model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    device = _get_device()
    model.to(device)
    print(f"[MMS-ZS-BASIC] Using device: {device}")

    hyps_roman: List[str] = []

    N = len(ds)
    if len(refs_roman) != N:
        raise ValueError(
            f"refs_roman length ({len(refs_roman)}) does not match dataset size ({N}). "
            "Make sure transcriptions_uroman.txt is aligned with transcriptions.txt."
        )

    for i in range(N):
        audio_info = ds[i]["audio"]

        if isinstance(audio_info, dict):
            path = audio_info["path"]
        else:
            path = audio_info

        # Load audio from file as float32 at 16 kHz mono
        audio, sr = librosa.load(path, sr=ASR_SAMPLING_RATE, mono=True)
        audio = audio.astype(np.float32)

        inputs = processor(
            audio,
            sampling_rate=ASR_SAMPLING_RATE,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            logits = model(**inputs).logits  # (batch, frames, vocab)

        # Greedy CTC decoding -> uroman tokens
        pred_ids = torch.argmax(logits, dim=-1)
        hyp_roman = processor.batch_decode(pred_ids)[0].strip()
        hyps_roman.append(hyp_roman)

        ref_dev = ds[i]["text"]
        ref_roman = refs_roman[i]

        print(f"[MMS-ZS-BASIC] Sample {i}/{N}")
        print(f"PATH: {path}")
        print(f"REF_DEV: {ref_dev}")
        print(f"REF_UROMAN: {ref_roman}")
        print(f"HYP_MMS_UROMAN: {hyp_roman}\n")

    wer_val = float(wer(refs_roman, hyps_roman))
    cer_val = float(cer(refs_roman, hyps_roman))

    print(f"[MMS-ZS-BASIC] WER (uroman): {wer_val}")
    print(f"[MMS-ZS-BASIC] CER (uroman): {cer_val}")

    return {
        "model": f"{MODEL_ID} (zeroshot-greedy, uroman)",
        "wer": wer_val,
        "cer": cer_val,
        "n_samples": N,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    base_dir = repo_root / "data" / "hindi_audio"

    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    # Dataset with Devanagari references
    ds = loader.from_dir_with_text(
        str(base_dir),
        str(base_dir / "transcriptions.txt"),
    )

    # Romanized refs (uroman), aligned line-by-line
    roman_path = base_dir / "transcriptions_uroman.txt"
    with roman_path.open("r", encoding="utf-8") as f:
        refs_roman = [ln.rstrip("\n") for ln in f]

    run_mms_zeroshot_baseline_basic(loader=loader, ds=ds, refs_roman=refs_roman)


if __name__ == "__main__":
    main()
