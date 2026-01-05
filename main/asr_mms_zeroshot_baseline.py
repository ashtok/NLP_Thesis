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
MODEL_ID = "mms-meta/mms-zeroshot-300m"  # basic zero-shot MMS model [web:485]


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
) -> Dict[str, float]:
    """
    Run MMS zero-shot 300M baseline with greedy CTC decoding (no lexicon/LM).

    Args:
        loader: HFAudioLoader (for symmetry; not used directly here).
        ds: Dataset with columns: audio, text.

    Returns:
        dict with model name, WER, CER, and n_samples.
    """
    print("[MMS-ZS-BASIC] Loading MMS zero-shot model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    device = _get_device()
    model.to(device)
    print(f"[MMS-ZS-BASIC] Using device: {device}")

    refs: List[str] = []
    hyps: List[str] = []

    N = len(ds)
    for i in range(N):
        audio_info = ds[i]["audio"]

        # HFAudioLoader stores either a dict with 'path' or a path string
        if isinstance(audio_info, dict):
            path = audio_info["path"]
        else:
            path = audio_info  # already a path string

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

        # Greedy CTC decoding
        pred_ids = torch.argmax(logits, dim=-1)
        hyp = processor.batch_decode(pred_ids)[0].strip()

        ref = ds[i]["text"]

        refs.append(ref)
        hyps.append(hyp)

        print(f"[MMS-ZS-BASIC] Sample {i}/{N}")
        print(f"PATH: {path}")
        print(f"REF: {ref}")
        print(f"HYP_MMS_ZS_BASIC: {hyp}\n")

    wer_val = float(wer(refs, hyps))
    cer_val = float(cer(refs, hyps))

    print(f"[MMS-ZS-BASIC] WER: {wer_val}")
    print(f"[MMS-ZS-BASIC] CER: {cer_val}")

    return {
        "model": f"{MODEL_ID} (zeroshot-greedy)",
        "wer": wer_val,
        "cer": cer_val,
        "n_samples": N,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent / "data" / "hindi_audio"

    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    ds = loader.from_dir_with_text(
        str(base_dir),
        str(base_dir / "transcriptions_uroman.txt"),
    )

    run_mms_zeroshot_baseline_basic(loader=loader, ds=ds)


if __name__ == "__main__":
    main()
