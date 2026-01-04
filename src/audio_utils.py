import librosa
import numpy as np
from pathlib import Path
from typing import Tuple


class AudioLoader:
    """Load and preprocess audio files"""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate

        Returns:
            waveform (np.ndarray): Audio as 1D numpy array
            sample_rate (int): Sample rate (should match target_sr)
        """
        waveform, sr = librosa.load(audio_path, sr=self.target_sr)
        return waveform, sr

    def load_batch(self, audio_paths: list[Path]) -> list[Tuple[np.ndarray, int]]:
        """Load multiple audio files"""
        return [self.load(path) for path in audio_paths]
