# setup_tools.py - Complete tool system with multiple ASR models

from pathlib import Path
from typing import Callable, Dict, Any, Optional
import numpy as np
import torch
import tempfile
import os
import json

# Core ASR models
from asr_models import SpeechRecognizer, LanguageIdentifier
from linguistic_tools import PanLexTool, UromanTool

# Optional imports - install with: pip install openai-whisper transformers sentencepiece
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not available. Install: pip install openai-whisper")

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    NLLB_AVAILABLE = True
except ImportError:
    NLLB_AVAILABLE = False
    print("⚠️  NLLB not available. Install: pip install transformers sentencepiece")


class ToolRegistry:
    """Registry of tools that LLM can call"""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: list[dict] = []

    def register(self, name: str, func: Callable, schema: dict):
        """Register a tool"""
        self.tools[name] = func
        self.schemas.append(schema)

    def execute(self, name: str, args: dict) -> Any:
        """Execute a tool by name"""
        if name not in self.tools:
            return {"error": f"Unknown tool: {name}"}

        try:
            return self.tools[name](**args)
        except Exception as e:
            return {"error": str(e)}

    def get_schemas(self) -> list[dict]:
        """Get all tool schemas for LLM"""
        return self.schemas


class CrubadanTool:
    """Word frequency lookup using Crúbadán data"""

    def __init__(self, freq_file: Optional[Path] = None):
        self.freq_data = {}
        if freq_file and freq_file.exists():
            try:
                with open(freq_file, 'r', encoding='utf-8') as f:
                    self.freq_data = json.load(f)
                print(f"✓ Crúbadán data loaded: {len(self.freq_data)} languages")
            except Exception as e:
                print(f"⚠️  Could not load Crúbadán: {e}")

    def get_frequency(self, word: str, lang: str = "hin") -> int:
        """Get word frequency (higher = more common)"""
        return self.freq_data.get(lang, {}).get(word.lower(), 0)

    def score_hypothesis(self, hypothesis: str, lang: str = "hin") -> float:
        """Score hypothesis by average word frequency"""
        words = hypothesis.split()
        if not words:
            return 0.0

        total_freq = sum(self.get_frequency(w, lang) for w in words)
        return total_freq / len(words)


class ASRToolkit:
    """All tools needed for agentic ASR"""

    def __init__(
            self,
            panlex_path: Path,
            crubadan_path: Optional[Path] = None,
            load_whisper: bool = True,
            load_nllb: bool = False,
            whisper_model: str = "small"
    ):
        print("Initializing ASR Toolkit...")

        # Core models (always loaded)
        print("  Loading MMS models...")
        self.recognizer = SpeechRecognizer(beam_size=50)  # Increased beam size
        self.lid = LanguageIdentifier()

        # Linguistic tools
        print("  Loading linguistic tools...")
        self.panlex = PanLexTool(panlex_path=panlex_path)
        self.uroman = UromanTool()

        # Crúbadán (optional)
        if crubadan_path:
            self.crubadan = CrubadanTool(crubadan_path)
        else:
            self.crubadan = None

        # Whisper (optional)
        self.whisper_model = None
        if load_whisper and WHISPER_AVAILABLE:
            print(f"  Loading Whisper ({whisper_model})...")
            try:
                self.whisper_model = whisper.load_model(whisper_model)
                print("  ✓ Whisper loaded")
            except Exception as e:
                print(f"  ⚠️  Whisper load failed: {e}")

        # NLLB Translator (optional)
        self.nllb_tokenizer = None
        self.nllb_model = None
        if load_nllb and NLLB_AVAILABLE:
            print("  Loading NLLB translator...")
            try:
                self.nllb_tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/nllb-200-distilled-600M"
                )
                self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                    "facebook/nllb-200-distilled-600M"
                )
                print("  ✓ NLLB loaded")
            except Exception as e:
                print(f"  ⚠️  NLLB load failed: {e}")

        # Current audio context
        self.current_waveform = None
        self.current_sr = None

        print("✓ Toolkit ready!")
        print(f"  Available tools:")
        print(f"    - MMS Zero-Shot ASR ✓")
        print(f"    - Language Identifier ✓")
        print(f"    - PanLex Dictionary ✓")
        print(f"    - Uroman Romanization ✓")
        print(f"    - Whisper ASR {'✓' if self.whisper_model else '✗'}")
        print(f"    - NLLB Translator {'✓' if self.nllb_model else '✗'}")
        print(f"    - Crúbadán Frequency {'✓' if self.crubadan else '✗'}")

    def set_audio(self, waveform: np.ndarray, sr: int):
        """Set current audio for tool calls"""
        self.current_waveform = waveform
        self.current_sr = sr

    def create_registry(self) -> ToolRegistry:
        """Create and populate tool registry"""
        registry = ToolRegistry()

        # Tool 1: MMS Zero-Shot ASR
        registry.register(
            name="call_zeroshot_asr",
            func=self._call_zeroshot_asr,
            schema={
                "type": "function",
                "function": {
                    "name": "call_zeroshot_asr",
                    "description": "Transcribe speech using MMS Zero-Shot ASR. Returns n-best hypotheses in romanized form.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n_best_hypotheses": {
                                "type": "integer",
                                "description": "Number of hypotheses (1-5)",
                                "default": 3
                            }
                        }
                    }
                }
            }
        )

        # Tool 2: Language ID
        registry.register(
            name="call_language_identifier",
            func=self._call_language_identifier,
            schema={
                "type": "function",
                "function": {
                    "name": "call_language_identifier",
                    "description": "Identify the language of audio. Returns top-k language probabilities.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top languages to return",
                                "default": 3
                            }
                        }
                    }
                }
            }
        )

        # Tool 3: Whisper ASR (if available)
        if self.whisper_model:
            registry.register(
                name="call_whisper",
                func=self._call_whisper,
                schema={
                    "type": "function",
                    "function": {
                        "name": "call_whisper",
                        "description": "Transcribe speech using Whisper ASR (good for Indian languages). Returns romanized transcription.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "language": {
                                    "type": "string",
                                    "description": "Language code (hi=Hindi, ur=Urdu)",
                                    "default": "hi"
                                }
                            }
                        }
                    }
                }
            )

        # Tool 4: Validate word
        registry.register(
            name="validate_word",
            func=self._validate_word,
            schema={
                "type": "function",
                "function": {
                    "name": "validate_word",
                    "description": "Check if a word exists in the dictionary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "lang_code": {"type": "string"}
                        },
                        "required": ["word", "lang_code"]
                    }
                }
            }
        )

        # Tool 5: Query PanLex
        registry.register(
            name="query_panlex",
            func=self._query_panlex,
            schema={
                "type": "function",
                "function": {
                    "name": "query_panlex",
                    "description": "Search dictionary for words.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "iso_639_3": {"type": "string"},
                            "max_size": {"type": "integer", "default": 10}
                        }
                    }
                }
            }
        )

        # Tool 6: Romanize text
        registry.register(
            name="romanize_text",
            func=self._romanize_text,
            schema={
                "type": "function",
                "function": {
                    "name": "romanize_text",
                    "description": "Convert text from native script to Latin script.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"}
                        },
                        "required": ["text"]
                    }
                }
            }
        )

        # Tool 7: Compare transcriptions (if you added this)
        if hasattr(self, '_compare_transcriptions'):
            registry.register(
                name="compare_transcriptions",
                func=self._compare_transcriptions,
                schema={
                    "type": "function",
                    "function": {
                        "name": "compare_transcriptions",
                        "description": "Compare two transcriptions and score them.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "transcription1": {"type": "string"},
                                "transcription2": {"type": "string"},
                                "lang_code": {"type": "string", "default": "hin"}
                            },
                            "required": ["transcription1", "transcription2"]
                        }
                    }
                }
            )

        return registry

    # Tool implementations
    def _call_zeroshot_asr(self, n_best_hypotheses: int = 3):
        """MMS Zero-Shot ASR"""
        if self.current_waveform is None:
            return {"error": "No audio loaded"}

        hypotheses = self.recognizer.transcribe(
            self.current_waveform,
            self.current_sr,
            n_best=n_best_hypotheses
        )
        return {
            "hypotheses": hypotheses,
            "source": "mms_zeroshot",
            "count": len(hypotheses)
        }

    def _call_language_identifier(self, top_k: int = 3):
        """Language identification"""
        if self.current_waveform is None:
            return {"error": "No audio loaded"}

        lang_probs = self.lid.identify(
            self.current_waveform,
            self.current_sr,
            top_k=top_k
        )
        return {"languages": lang_probs}

    def _call_whisper(self, language: str = "hi"):
        """Whisper ASR with automatic romanization"""
        if self.current_waveform is None:
            return {"error": "No audio loaded"}

        if not self.whisper_model:
            return {"error": "Whisper not available"}

        try:
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, self.current_waveform, self.current_sr)
                temp_path = f.name

            # Transcribe
            result = self.whisper_model.transcribe(
                temp_path,
                language=language,
                task="transcribe"
            )

            os.unlink(temp_path)

            transcription_native = result["text"].strip()

            # Auto-romanize if output contains non-Latin characters
            if any(ord(c) > 127 for c in transcription_native):
                transcription_romanized = self.uroman.romanize(transcription_native)
            else:
                transcription_romanized = transcription_native

            return {
                "transcription": transcription_romanized,  # Romanized version
                "transcription_native": transcription_native,  # Original
                "language": result.get("language", language),
                "source": "whisper"
            }
        except Exception as e:
            return {"error": f"Whisper failed: {str(e)}"}

    def _validate_word(self, word: str, lang_code: str):
        """Validate word in dictionary"""
        is_valid = self.panlex.validate_word(word, lang_code)
        return {
            "word": word,
            "lang_code": lang_code,
            "is_valid": is_valid
        }

    def _query_panlex(self, word: str = None, iso_639_3: str = None, max_size: int = 10):
        """Query PanLex dictionary"""
        result = self.panlex.query(
            word=word,
            iso_639_3=iso_639_3,
            max_size=max_size
        )
        return {"result": result}

    def _romanize_text(self, text: str):
        """Romanize native script"""
        romanized = self.uroman.romanize(text)
        return {
            "original": text,
            "romanized": romanized
        }

    def _translate_to_native(self, text: str, target_lang: str = "hin_Deva"):
        """Translate to native script using NLLB"""
        if not self.nllb_model:
            return {"error": "NLLB not available"}

        try:
            inputs = self.nllb_tokenizer(text, return_tensors="pt")

            translated_tokens = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=self.nllb_tokenizer.lang_code_to_id[target_lang],
                max_length=512
            )

            translation = self.nllb_tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True
            )[0]

            return {
                "original": text,
                "translated": translation,
                "target_script": target_lang
            }
        except Exception as e:
            return {"error": f"NLLB failed: {str(e)}"}

    def _get_word_frequency(self, word: str, lang: str = "hin"):
        """Get word frequency score"""
        if not self.crubadan:
            return {"error": "Crúbadán not available"}

        freq = self.crubadan.get_frequency(word, lang)
        return {
            "word": word,
            "language": lang,
            "frequency": freq
        }

    def _compare_transcriptions(self, transcription1: str, transcription2: str, lang_code: str = "hin"):
        """Compare two transcriptions and score them"""

        # Tokenize
        words1 = transcription1.lower().split()
        words2 = transcription2.lower().split()

        # Validate words
        valid1 = sum(1 for w in words1 if self.panlex.validate_word(w, lang_code))
        valid2 = sum(1 for w in words2 if self.panlex.validate_word(w, lang_code))

        score1 = (valid1 / len(words1) * 100) if words1 else 0
        score2 = (valid2 / len(words2) * 100) if words2 else 0

        return {
            "transcription1": transcription1,
            "transcription2": transcription2,
            "valid_words1": valid1,
            "valid_words2": valid2,
            "total_words1": len(words1),
            "total_words2": len(words2),
            "score1": score1,
            "score2": score2,
            "recommendation": "transcription1" if score1 > score2 else "transcription2"
        }