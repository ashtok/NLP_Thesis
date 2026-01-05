from pathlib import Path
from typing import List

from jiwer import wer

from audio_loader import HFAudioLoader
from src.asr_models import SpeechRecognizer  # <- use your wrapper


def main():
    base_dir = Path(__file__).resolve().parent.parent / "data" / "hindi_audio"
    base_dir_str = str(base_dir)

    loader = HFAudioLoader(target_sr=16_000)
    ds = loader.from_dir_with_text(
        base_dir_str,
        str(base_dir / "transcriptions.txt"),
    )

    recognizer = SpeechRecognizer(
        model_name="facebook/mms-1b-all",
        beam_size=10,
        topk_per_timestep=20,
    )

    refs: List[str] = []
    hyps: List[str] = []

    N = len(ds)
    for i in range(N):
        waveform, sr, path = loader.get_example(i)

        # start with greedy decoding (n_best=1) to reduce weird punctuation
        hyp = recognizer.transcribe(waveform, sr, n_best=1)
        ref = ds[i]["text"]

        refs.append(ref)
        hyps.append(hyp)

        print(str(path))
        print(f"REF: {ref}")
        print(f"HYP: {hyp}\n")

    # print("WER:", wer(refs, hyps))
    # print("CER:", cer(refs, hyps))



if __name__ == "__main__":
    main()