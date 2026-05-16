"""
evaluate.py
Оценка качества модели:
  - Perplexity на test set
  - BLEU-1/2/4
  - ROUGE-L
  - Примеры генераций
"""

import os
import json
import argparse
import logging
import math
from typing import List, Dict

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Метрики
# ──────────────────────────────────────────────

def compute_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """BLEU-1/2/4 через sacrebleu."""
    try:
        from sacrebleu.metrics import BLEU
        results = {}
        for n in [1, 2, 4]:
            bleu = BLEU(max_ngram_order=n)
            score = bleu.corpus_score(hypotheses, [references])
            results[f"BLEU-{n}"] = round(score.score, 4)
        return results
    except ImportError:
        logger.warning("sacrebleu не установлен (pip install sacrebleu). BLEU пропускаем.")
        # Fallback: простой unigram BLEU
        from collections import Counter
        scores = []
        for hyp, ref in zip(hypotheses, references):
            hyp_tokens = hyp.lower().split()
            ref_tokens = set(ref.lower().split())
            if not hyp_tokens:
                scores.append(0.0)
                continue
            matches = sum(1 for t in hyp_tokens if t in ref_tokens)
            scores.append(matches / len(hyp_tokens))
        return {"BLEU-1-approx": round(sum(scores) / max(len(scores), 1) * 100, 4)}


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """ROUGE-L через rouge-score."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        f1s = []
        for hyp, ref in zip(hypotheses, references):
            score = scorer.score(ref, hyp)
            f1s.append(score["rougeL"].fmeasure)
        return {"ROUGE-L": round(sum(f1s) / max(len(f1s), 1) * 100, 4)}
    except ImportError:
        logger.warning("rouge-score не установлен (pip install rouge-score). ROUGE пропускаем.")
        return {}


# ──────────────────────────────────────────────
# Основная оценка
# ──────────────────────────────────────────────

def evaluate_model(
    checkpoint_dir: str,
    test_jsonl: str,
    max_samples: int = 1000,
    batch_size: int = 8,
    max_new_tokens: int = 80,
    device: str = None,
    output_file: str = "eval_results.json",
    show_examples: int = 5,
):
    from inference import SupportBot

    bot = SupportBot(
        checkpoint_dir=checkpoint_dir,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    from dataset import build_tokenizer, UbuntuDialogDataset
    from torch.utils.data import DataLoader
    from model import SupportGPT2

    tokenizer = bot.tokenizer
    device_obj = bot.device

    # ── 1. Perplexity ──
    logger.info("Вычисление perplexity...")
    test_ds = UbuntuDialogDataset(test_jsonl, tokenizer, max_samples=max_samples)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    n_batches = 0
    bot.model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Perplexity"):
            input_ids = batch["input_ids"].to(device_obj)
            attention_mask = batch["attention_mask"].to(device_obj)
            labels = batch["labels"].to(device_obj)
            with autocast(enabled=(device_obj.type == "cuda")):
                out = bot.model(input_ids, attention_mask, labels=labels)
            total_loss += out.loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    logger.info(f"Perplexity: {ppl:.2f} (loss={avg_loss:.4f})")

    # ── 2. BLEU / ROUGE на генерации ──
    logger.info("Генерация ответов для BLEU/ROUGE...")
    import json as _json
    test_pairs = []
    with open(test_jsonl) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            test_pairs.append(_json.loads(line.strip()))

    hypotheses = []
    references = []
    examples = []

    for item in tqdm(test_pairs, desc="Generating"):
        context = item["context"]
        reference = item["response"]
        history = context.split(" [SEP] ")

        generated = bot.generate(history, num_candidates=1)[0]
        hypotheses.append(generated)
        references.append(reference)

        if len(examples) < show_examples:
            examples.append({
                "context": context,
                "reference": reference,
                "generated": generated,
            })

    bleu_scores = compute_bleu(hypotheses, references)
    rouge_scores = compute_rouge(hypotheses, references)

    # ── Результаты ──
    results = {
        "perplexity": round(ppl, 4),
        "loss": round(avg_loss, 4),
        **bleu_scores,
        **rouge_scores,
        "n_samples": len(hypotheses),
    }

    logger.info("=" * 50)
    logger.info("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    logger.info("=" * 50)
    for k, v in results.items():
        logger.info(f"  {k:20s}: {v}")
    logger.info("=" * 50)

    # Примеры
    logger.info(f"\nПримеры генерации ({show_examples}):")
    for i, ex in enumerate(examples, 1):
        logger.info(f"\n[{i}] Контекст: {ex['context'][:120]}...")
        logger.info(f"    Референс:  {ex['reference']}")
        logger.info(f"    Генерация: {ex['generated']}")

    # Сохранение
    full_results = {
        "metrics": results,
        "examples": examples,
        "checkpoint": checkpoint_dir,
        "test_file": test_jsonl,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        _json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nРезультаты сохранены: {output_file}")

    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка модели техподдержки")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/checkpoint-best")
    parser.add_argument("--test_data", type=str, default="./data/processed/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    evaluate_model(
        checkpoint_dir=args.checkpoint,
        test_jsonl=args.test_data,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        output_file=args.output,
        show_examples=args.examples,
    )
