# qwen_agent.py - COMPLETE working version

import json
import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

from audio_utils import AudioLoader
from setup_tools import ASRToolkit


class QwenAgent:
    """Agentic ASR system using Qwen 2.5"""

    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-3B-Instruct",
            panlex_path: Path = Path("../data/panlex.csv"),
            device: str = None,
            load_in_8bit: bool = False,
            load_whisper: bool = True,  # NEW: Enable Whisper by default
            whisper_model: str = "small"
    ):
        print(f"Initializing Qwen Agent with {model_name}...")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load Qwen model
        print("Loading Qwen model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if load_in_8bit and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )

        self.model.eval()
        print(f"✓ Model loaded on {self.device}")

        # Setup tools WITH WHISPER
        self.toolkit = ASRToolkit(
            panlex_path=panlex_path,
            load_whisper=load_whisper,
            whisper_model=whisper_model
        )
        self.registry = self.toolkit.create_registry()

        print(f"✓ Registered {len(self.registry.get_schemas())} tools")
        print("✓ Qwen Agent ready!")

    def run(
            self,
            audio_path: Path,
            max_iterations: int = 10,
            verbose: bool = True
    ) -> Dict[str, Any]:
        """Run agent on audio file"""
        # Load audio
        loader = AudioLoader()
        waveform, sr = loader.load(audio_path)
        self.toolkit.set_audio(waveform, sr)

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Processing: {audio_path.name}")
            print(f"{'=' * 80}")

        # System prompt
        system_prompt = """You are a speech recognition agent. Your task: transcribe audio accurately.

        AVAILABLE TOOLS:
        1. call_language_identifier - Detect language (call first)
        2. call_zeroshot_asr - MMS Zero-Shot ASR (returns romanized, multiple hypotheses)
        3. call_whisper - Whisper ASR (returns romanized, single output, usually more accurate)
        4. validate_word - Check if word exists in dictionary
        5. query_panlex - Search dictionary
        6. romanize_text - Convert native script to Latin

        MANDATORY WORKFLOW:
        Step 1: Call call_language_identifier → get language
        Step 2: Call BOTH ASR systems:
                - call_zeroshot_asr (n_best_hypotheses=3)
                - call_whisper (language from step 1)
        Step 3: Compare outputs and validate words
        Step 4: Choose best transcription

        SELECTION CRITERIA:
        - Prefer Whisper if it looks reasonable (usually more accurate)
        - Use dictionary validation to check word validity
        - Compare both systems and explain choice

        Output format:
        LANGUAGE: <name> (<code>)
        TRANSCRIPTION: <best transcription in romanized form>
        CONFIDENCE: <high/medium/low>
        REASONING: <why you chose this transcription, mention which ASR system(s) used>

        CRITICAL: You MUST call both ASR systems before providing final answer!"""

        user_prompt = "Please transcribe this audio. Use the tools systematically."

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Agent loop
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Format messages for Qwen
            text = self.tokenizer.apply_chat_template(
                messages,
                tools=self.registry.get_schemas(),
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate response
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

            if verbose:
                print(f"Raw response: {response[:200]}...")

            # Parse tool calls (Qwen format: <tool_call>...</tool_call>)
            if "<tool_call>" in response:
                tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
                tool_calls_raw = re.findall(tool_call_pattern, response, re.DOTALL)

                if tool_calls_raw:
                    if verbose:
                        print(f"Agent called {len(tool_calls_raw)} tool(s)")

                    # Add assistant message
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })

                    # Execute tools
                    tool_results = []
                    for tool_call_str in tool_calls_raw:
                        try:
                            tool_call = json.loads(tool_call_str.strip())
                            tool_name = tool_call.get("name")
                            tool_args = tool_call.get("arguments", {})

                            if verbose:
                                print(f"  → {tool_name}({tool_args})")

                            # Execute
                            result = self.registry.execute(tool_name, tool_args)
                            tool_results.append(f"{tool_name} returned: {json.dumps(result, ensure_ascii=False)}")

                            if verbose:
                                # Pretty print result
                                if isinstance(result, dict) and "hypotheses" in result:
                                    print(f"     Got {len(result['hypotheses'])} hypotheses")
                                    for i, h in enumerate(result['hypotheses'][:3], 1):
                                        print(f"       {i}. {h}")
                                elif isinstance(result, dict) and "languages" in result:
                                    for lang, prob in list(result['languages'].items())[:3]:
                                        print(f"       {lang}: {prob:.1%}")
                                elif isinstance(result, dict) and "is_valid" in result:
                                    status = "✓" if result['is_valid'] else "✗"
                                    print(f"       {status} {result['word']}: {result['is_valid']}")
                                else:
                                    print(f"     {result}")

                        except Exception as e:
                            if verbose:
                                print(f"  Error parsing tool call: {e}")
                            tool_results.append(f"Error: {e}")

                    # Add tool results as user message
                    messages.append({
                        "role": "user",
                        "content": "Tool results:\n" + "\n".join(tool_results)
                    })

                    continue  # Next iteration

            # Check for final answer
            if any(keyword in response.upper() for keyword in ["LANGUAGE:", "TRANSCRIPTION:"]):
                if verbose:
                    print(f"\n✓ Final answer received")

                messages.append({
                    "role": "assistant",
                    "content": response
                })

                return self._parse_final_answer(response, messages)

            # If no tool calls and no final answer, prompt for more
            messages.append({
                "role": "assistant",
                "content": response
            })

            if iteration < max_iterations - 1:
                messages.append({
                    "role": "user",
                    "content": "Please continue. If you have enough information, provide your final answer."
                })

        # Max iterations reached
        return {
            "final_answer": "Max iterations reached",
            "language": "unknown",
            "transcription": "",
            "confidence": "",
            "reasoning": "",
            "message_history": messages
        }

    def _parse_final_answer(self, response: str, messages: List[Dict]) -> Dict[str, Any]:
        """Parse the final answer from LLM response"""
        result = {
            "final_answer": response,
            "language": "",
            "transcription": "",
            "confidence": "",
            "reasoning": "",
            "message_history": messages
        }

        # Extract fields
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("LANGUAGE:"):
                result["language"] = line.replace("LANGUAGE:", "").strip()
            elif line.startswith("TRANSCRIPTION:"):
                result["transcription"] = line.replace("TRANSCRIPTION:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                result["confidence"] = line.replace("CONFIDENCE:", "").strip()
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()

        return result


# Test script
if __name__ == "__main__":

    agent = QwenAgent(
        model_name="Qwen/Qwen2.5-3B-Instruct",  # Better reasoning
        panlex_path=Path("../data/panlex.csv"),
        load_in_8bit=False
    )

    result = agent.run(
        audio_path=Path("../data/hindi_audio/hindi_001.wav"),
        max_iterations=10,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if result:  # Check if result is not None
        print(f"Language: {result.get('language', 'N/A')}")
        print(f"Transcription: {result.get('transcription', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    else:
        print("ERROR: No result returned from agent")
