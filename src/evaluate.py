# evaluate.py - Calculate WER/CER with romanization and GPU acceleration

import json
import re
from pathlib import Path
import sys
import torch

try:
    from torchmetrics.text import CharErrorRate, WordErrorRate

    has_torchmetrics = True
except ImportError:
    print("⚠️  torchmetrics not installed, falling back to jiwer")
    try:
        import jiwer

        has_torchmetrics = False
    except ImportError:
        print("❌ Error: Neither torchmetrics nor jiwer installed")
        print("Run: pip install torchmetrics  (recommended for GPU)")
        print("Or:  pip install jiwer  (CPU only)")
        sys.exit(1)

from linguistic_tools import UromanTool


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def batch_romanize(texts: list, uroman: UromanTool) -> list:
    """Romanize texts in batch for efficiency"""
    romanized = []
    for text in texts:
        if any(ord(c) > 127 for c in text):
            romanized.append(uroman.romanize(text))
        else:
            romanized.append(text)
    return romanized


def calculate_metrics_gpu(references: list, hypotheses: list, device) -> list:
    """Calculate WER and CER using torchmetrics on GPU"""
    if not has_torchmetrics:
        return None

    # Initialize metrics on the specified device
    wer_metric = WordErrorRate().to(device)
    cer_metric = CharErrorRate().to(device)

    results = []
    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            results.append(None)
            continue

        try:
            # torchmetrics works with lists of strings
            wer = wer_metric([hyp], [ref]).item()
            cer = cer_metric([hyp], [ref]).item()

            results.append({
                "wer": wer * 100,
                "cer": cer * 100
            })
        except Exception as e:
            print(f"  ⚠️  Error calculating metrics: {e}")
            results.append(None)

    return results


def calculate_metrics_cpu(reference: str, hypothesis: str) -> dict:
    """Fallback CPU calculation using jiwer"""
    try:
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        return {
            "wer": wer * 100,
            "cer": cer * 100
        }
    except Exception as e:
        print(f"  ⚠️  Error calculating metrics: {e}")
        return None


def evaluate_results(results_file: Path, ground_truth_file: Path):
    """Evaluate agent results against ground truth"""

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda" and has_torchmetrics

    print(f"\n{'=' * 80}")
    print(f"EVALUATION")
    print(f"{'=' * 80}")
    print(f"Results:       {results_file}")
    print(f"Ground truth:  {ground_truth_file}")
    print(f"Device:        {device} {'(GPU Accelerated ✓)' if use_gpu else '(CPU)'}")
    if not has_torchmetrics and device.type == "cuda":
        print(f"⚠️  Install torchmetrics for GPU acceleration: pip install torchmetrics")
    print(f"{'=' * 80}\n")

    # Initialize uroman
    print("Initializing Uroman...")
    uroman = UromanTool()

    # Load data
    print(f"Loading agent results...")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"  Found {len(results)} results")

    print(f"Loading ground truth...")
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    print(f"  Found {len(ground_truth)} references")

    gt_dict = {item['file']: item for item in ground_truth}

    # Prepare data for batch processing
    valid_pairs = []
    file_names = []
    missing_gt = []

    for result in results:
        if not result.get('success', True):
            continue

        file_name = result['file']
        hypothesis = result.get('transcription', '')

        if file_name not in gt_dict:
            missing_gt.append(file_name)
            continue

        reference = gt_dict[file_name]['transcription']
        valid_pairs.append((reference, hypothesis))
        file_names.append(file_name)

    if not valid_pairs:
        print("❌ No valid pairs to evaluate!")
        return None

    print(f"\n{'=' * 80}")
    print("PROCESSING FILES (Batch Mode)")
    print(f"{'=' * 80}\n")

    # Batch romanization
    print(f"Romanizing {len(valid_pairs)} references...")
    references_orig = [pair[0] for pair in valid_pairs]
    references_romanized = batch_romanize(references_orig, uroman)

    # Normalize all texts
    print(f"Normalizing texts...")
    references_norm = [normalize_text(ref) for ref in references_romanized]
    hypotheses_norm = [normalize_text(pair[1]) for pair in valid_pairs]

    # Calculate metrics
    print(f"Calculating metrics...")
    if use_gpu:
        metric_results = calculate_metrics_gpu(references_norm, hypotheses_norm, device)
    else:
        metric_results = [
            calculate_metrics_cpu(ref, hyp)
            for ref, hyp in zip(references_norm, hypotheses_norm)
        ]

    # Compile results
    metrics = []
    for i, (file_name, metric, ref_orig, ref_norm, hyp_norm) in enumerate(
            zip(file_names, metric_results, references_orig, references_norm, hypotheses_norm), 1
    ):
        if metric is None:
            print(f"[{i}/{len(file_names)}] ⚠️  {file_name}: Metric calculation failed")
            continue

        metric_dict = {
            'file': file_name,
            'wer': metric['wer'],
            'cer': metric['cer'],
            'reference_original': ref_orig,
            'reference_romanized': ref_norm,
            'hypothesis': hyp_norm,
            'ref_words': len(ref_norm.split()),
            'hyp_words': len(hyp_norm.split())
        }
        metrics.append(metric_dict)

        print(f"[{i}/{len(file_names)}] {file_name}")
        print(f"  REF (orig): {ref_orig[:60]}...")
        print(f"  REF (rom):  {ref_norm[:60]}...")
        print(f"  HYP:        {hyp_norm[:60]}...")
        print(f"  WER: {metric['wer']:.2f}%  |  CER: {metric['cer']:.2f}%")
        print()

    # Calculate averages
    if not metrics:
        print("❌ No metrics calculated!")
        return None

    avg_wer = sum(m['wer'] for m in metrics) / len(metrics)
    avg_cer = sum(m['cer'] for m in metrics) / len(metrics)

    print(f"\n{'=' * 80}")
    print(f"AVERAGE METRICS")
    print(f"{'=' * 80}")
    print(f"Average WER:     {avg_wer:.2f}%")
    print(f"Average CER:     {avg_cer:.2f}%")
    print(f"Files evaluated: {len(metrics)}/{len(results)}")

    if missing_gt:
        print(f"Missing GT:      {len(missing_gt)} files")

    # Save results
    output_file = results_file.parent / f"evaluation_{results_file.stem}.json"
    evaluation_results = {
        "summary": {
            "average_wer": avg_wer,
            "average_cer": avg_cer,
            "files_evaluated": len(metrics),
            "total_files": len(results),
            "missing_ground_truth": missing_gt,
            "device_used": str(device)
        },
        "per_file_metrics": metrics
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detailed results saved to: {output_file}")
    print(f"{'=' * 80}\n")

    return evaluation_results


if __name__ == "__main__":
    results_dir = Path("../results")

    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        print("   Run test_batch.py first to generate results")
        sys.exit(1)

    json_files = sorted(results_dir.glob("qwen*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not json_files:
        print(f"❌ Error: No results files found in {results_dir}")
        print("   Run test_batch.py first to generate results")
        sys.exit(1)

    results_file = json_files[0]
    print(f"📊 Using latest results file: {results_file.name}\n")

    ground_truth_file = Path("../data/ground_truth.json")

    if not ground_truth_file.exists():
        print(f"❌ Error: Ground truth not found: {ground_truth_file}")
        print(f"   Run: python parse_ground_truth.py first")
        sys.exit(1)

    evaluate_results(results_file, ground_truth_file)
