import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC
from typing import Union, List
from SimpleCTCDecoder import SimpleCTCDecoder


class SpeechRecognizer:
    """
    Wrapper for MMS Zero-Shot ASR with proper beam search
    """

    def __init__(
            self,
            model_name: str = "facebook/mms-1b-all",
            device: str = None,
            beam_size: int = 10,
            topk_per_timestep: int = 20
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name).to(self.device).eval()

        # Get CTC blank ID
        blank_id = getattr(self.processor.tokenizer, "pad_token_id", 0)
        if blank_id is None:
            blank_id = 0
        self.blank_id = int(blank_id)

        # Initialize CTC decoder
        self.decoder = SimpleCTCDecoder(
            blank_id=self.blank_id,
            beam_size=beam_size,
            top_k_per_timestep=topk_per_timestep  # ✅ correct name
        )

        # Vocabulary for decoding
        vocab = self.processor.tokenizer.get_vocab()
        self.labels = [k for k in sorted(vocab, key=vocab.get)]

    @torch.inference_mode()
    def transcribe(
            self,
            waveform: np.ndarray,
            sr: int,
            n_best: int = 1
    ) -> Union[str, List[str]]:
        """
        Transcribe audio waveform

        Args:
            waveform: 1D numpy array (float32)
            sr: sample rate (should be 16000)
            n_best: number of hypotheses to return

        Returns:
            If n_best=1: single string (best hypothesis)
            If n_best>1: list of strings (top n hypotheses)
        """
        # Preprocess audio
        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        # Get model logits
        logits = self.model(input_values).logits  # (B, T, V)

        if n_best == 1:
            # Greedy decoding (fast)
            pred_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(pred_ids.cpu().numpy())[0]
            return text
        else:
            # Beam search (better quality, n-best)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (T, V)

            # Decode with beam search
            token_seqs = self.decoder.decode(log_probs, n_best=n_best)

            # Convert token IDs to text
            texts = self.processor.batch_decode(token_seqs, skip_special_tokens=True)
            return texts


from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor


class LanguageIdentifier:
    """MMS Language Identification"""

    def __init__(
            self,
            model_name: str = "facebook/mms-lid-1024",
            device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)

    @torch.inference_mode()
    @torch.inference_mode()
    def identify(
            self,
            waveform: np.ndarray,
            sr: int,
            top_k: int = 5
    ) -> dict[str, float]:
        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt"
        )

        # Move tensors to device properly
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits

        probs = torch.nn.functional.softmax(logits, dim=-1).cpu()
        topk = probs.topk(top_k, dim=-1)

        return {
            self.model.config.id2label[i.item()]: p.item()
            for i, p in zip(topk.indices[0], topk.values[0])
        }

