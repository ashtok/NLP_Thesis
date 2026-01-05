# main/asr_mms_zeroshot_baseline.py

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import torch
from jiwer import cer, wer
from huggingface_hub import hf_hub_download
from torchaudio.models.decoder import ctc_decoder
from transformers import AutoProcessor, Wav2Vec2ForCTC

from audio_loader import HFAudioLoader
from utils.text_norm import text_normalize
from utils.lm import create_unigram_lm, maybe_generate_pseudo_bigram_arpa  # type: ignore


ASR_SAMPLING_RATE = 16_000

WORD_SCORE_DEFAULT_IF_LM = -0.18
WORD_SCORE_DEFAULT_IF_NOLM = -3.5
LM_SCORE_DEFAULT = 1.48

MODEL_ID = "mms-meta/mms-zeroshot-300m"  # MMS zero-shot 300M [web:485]

# Paths for uroman
UROMAN_DIR = Path(__file__).resolve().parent.parent / "uroman"
UROMAN_PL = UROMAN_DIR / "bin" / "uroman.pl"


def _ensure_uroman_available() -> None:
    if not UROMAN_DIR.exists() or not UROMAN_PL.exists():
        raise FileNotFoundError(
            f"uroman not found at {UROMAN_PL}. "
            f"Expected 'uroman' directory at repo root with bin/uroman.pl."
        )


def _norm_uroman(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def _uromanize(words: List[str]) -> Dict[str, str]:
    """Map words -> uroman character sequences."""
    _ensure_uroman_available()
    iso = "xxx"  # language-agnostic mode [web:490]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf_in, tempfile.NamedTemporaryFile(
        mode="r", delete=False
    ) as tf_out:
        in_path = tf_in.name
        out_path = tf_out.name

        tf_in.write("\n".join(words))
        tf_in.flush()

    try:
        cmd = f"perl {UROMAN_PL} -l {iso} < {in_path} > {out_path}"
        os.system(cmd)

        lexicon: Dict[str, str] = {}
        with open(out_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                line = re.sub(r"\s+", "", _norm_uroman(line)).strip()
                lexicon[words[idx]] = " ".join(line) + " |"
    finally:
        # Cleanup temp files
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)

    return lexicon


def _filter_lexicon(lexicon: Dict[str, str], word_counts: Dict[str, int]) -> Dict[str, str]:
    """Resolve multiple words mapping to same uroman spelling by keeping most frequent / shortest."""
    spelling_to_words: Dict[str, List[str]] = {}
    for w, s in lexicon.items():
        spelling_to_words.setdefault(s, [])
        spelling_to_words[s].append(w)

    new_lex: Dict[str, str] = {}
    for s, ws in spelling_to_words.items():
        if len(ws) > 1:
            ws.sort(key=lambda w: (-word_counts[w], len(w)))
        new_lex[ws[0]] = s
    return new_lex


def _load_words(filepath: str) -> Tuple[Dict[str, int], int]:
    """Load word counts from a text file with one sentence per line."""
    word_counts: Dict[str, int] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    num_sentences = len(lines)
    all_sentences = " ".join(l.strip() for l in lines)
    norm_all_sentences = text_normalize(all_sentences)  # from MMS utils [web:485]
    for w in norm_all_sentences.split():
        word_counts.setdefault(w, 0)
        word_counts[w] += 1
    return word_counts, num_sentences


def _prepare_mms_zeroshot_decoder(
    words_file: str,
    autolm: bool = True,
) -> Tuple[ctc_decoder, Dict[str, int], str]:
    """
    Build MMS zero-shot CTC decoder (lexicon + LM) once for the full experiment.
    Returns (decoder, word_counts, tmp_lm_path).
    """
    print(f"[MMS-ZS] Loading words from {words_file}")
    word_counts, num_sentences = _load_words(words_file)
    print(f"[MMS-ZS] Loaded {len(word_counts)} unique words from {num_sentences} lines")

    print("[MMS-ZS] Building uroman lexicon...")
    lexicon = _uromanize(list(word_counts.keys()))
    print(f"[MMS-ZS] Lexicon size (raw): {len(lexicon)}")

    tmp_lm = tempfile.NamedTemporaryFile(delete=False)
    lm_path = tmp_lm.name
    tmp_lm.close()

    # Optionally auto-create unigram LM; only if enough repetition
    if autolm and any(cnt > 2 for cnt in word_counts.values()):
        print("[MMS-ZS] Creating unigram LM...")
        create_unigram_lm(word_counts, num_sentences, lm_path)
        maybe_generate_pseudo_bigram_arpa(lm_path)
        use_lm = True
    else:
        lm_path = ""
        use_lm = False

    if not use_lm:
        print("[MMS-ZS] Filtering lexicon (no LM)...")
        lexicon = _filter_lexicon(lexicon, word_counts)
        print(f"[MMS-ZS] Lexicon size after filtering: {len(lexicon)}")

    # Load MMS tokens.txt from the HF repo
    token_file = hf_hub_download(repo_id=MODEL_ID, filename="tokens.txt")

    word_score = (
        WORD_SCORE_DEFAULT_IF_LM if use_lm else WORD_SCORE_DEFAULT_IF_NOLM
    )
    lm_score = LM_SCORE_DEFAULT if use_lm else 0.0

    print(f"[MMS-ZS] Using word score: {word_score}")
    print(f"[MMS-ZS] Using LM score: {lm_score} (use_lm={use_lm})")

    # Write lexicon to a temp file for torchaudio decoder
    lexicon_tmp = tempfile.NamedTemporaryFile(mode="w", delete=False)
    lexicon_path = lexicon_tmp.name
    with open(lexicon_path, "w", encoding="utf-8") as f:
        for word, spelling in lexicon.items():
            f.write(f"{word} {spelling}\n")
    lexicon_tmp.close()

    decoder = ctc_decoder(
        lexicon=lexicon_path,
        tokens=token_file,
        lm=lm_path if use_lm else None,
        nbest=1,
        beam_size=500,
        beam_size_token=50,
        lm_weight=lm_score,
        word_score=word_score,
        sil_score=0.0,
        blank_token="<s>",
    )

    return decoder, word_counts, lm_path


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


def run_mms_zeroshot_baseline(
    loader: HFAudioLoader,
    ds: Any,
    words_file: str,
) -> Dict[str, float]:
    """
    Run MMS zero-shot 300M baseline on a given dataset and return WER/CER.

    Args:
        loader: HFAudioLoader (for consistency with other baselines; not used directly).
        ds: Hugging Face Dataset-like object with columns: audio, text.
        words_file: Path to a text file containing sentences in the target language
                    (e.g., the same transcriptions.txt used for evaluation).

    Returns:
        dict with model name, WER, CER, and n_samples.
    """
    _ensure_uroman_available()

    print("[MMS-ZS] Loading MMS zero-shot model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    device = _get_device()
    model.to(device)
    print(f"[MMS-ZS] Using device: {device}")

    print("[MMS-ZS] Preparing zero-shot decoder (lexicon + LM)...")
    decoder, _, _ = _prepare_mms_zeroshot_decoder(words_file)

    refs: List[str] = []
    hyps: List[str] = []

    N = len(ds)
    for i in range(N):
        audio_info = ds[i]["audio"]
        # HFAudioLoader stores path + array; path is what we need for librosa
        if isinstance(audio_info, dict):
            path = audio_info["path"]
        else:
            path = audio_info  # fallback if already a path

        # Load audio with librosa (16 kHz mono)
        audio_samples, _sr = librosa.load(path, sr=ASR_SAMPLING_RATE, mono=True)
        audio_samples = audio_samples.astype(np.float32)

        inputs = processor(
            audio_samples,
            sampling_rate=ASR_SAMPLING_RATE,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            logits = model(**inputs).logits  # (batch, frames, vocab)
        emissions = logits.cpu()

        # torchaudio CTC decoder expects CPU (batch, frames, tokens)
        beam_search_result = decoder(emissions)
        hyp_words = beam_search_result[0][0].words
        hyp = " ".join(hyp_words).strip()

        ref = ds[i]["text"]

        refs.append(ref)
        hyps.append(hyp)

        print(str(path))
        print(f"REF: {ref}")
        print(f"HYP_MMS_ZS: {hyp}\n")

    wer_val = float(wer(refs, hyps))
    cer_val = float(cer(refs, hyps))

    print(f"[MMS-ZS] WER: {wer_val}")
    print(f"[MMS-ZS] CER: {cer_val}")

    return {
        "model": f"{MODEL_ID} (zeroshot)",
        "wer": wer_val,
        "cer": cer_val,
        "n_samples": N,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent / "data" / "hindi_audio"

    loader = HFAudioLoader(target_sr=ASR_SAMPLING_RATE)
    ds = loader.from_dir_with_text(
        str(base_dir),
        str(base_dir / "transcriptions.txt"),
    )

    run_mms_zeroshot_baseline(
        loader=loader,
        ds=ds,
        words_file=str(base_dir / "transcriptions.txt"),
    )

if __name__ == "__main__":
    main()
