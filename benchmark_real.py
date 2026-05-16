"""
benchmark_real.py
Бенчмарк на реальных вопросах пользователей Ubuntu.
Использует тот же benchmark.py но с другим датасетом.

Запуск:
    python benchmark_real.py
"""

import json
import os
import sys
import logging
import time
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("benchmark_real.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


def load_real_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Приводим к формату совместимому с benchmark.py
    pairs = []
    for item in data:
        pairs.append({
            "dialog_id": str(item["id"]),
            "context":   item["context"],
            "response":  item["reference"],
            "category":  item.get("category", ""),
        })
    logger.info(f"Загружено {len(pairs)} реальных вопросов")
    return pairs


def run_real_benchmark():
    # Импортируем всё нужное из benchmark.py
    sys.path.insert(0, os.path.dirname(__file__))
    from benchmark import (
        default_benchmark_config, init_clearml,
        HFModelAdapter, OllamaAdapter,
        compute_bleu, compute_rouge,
        compute_hf_perplexity, compute_score,
        log_results_to_clearml, ModelResult
    )
    import torch
    import numpy as np
    from dataclasses import asdict

    # Загружаем реальный датасет
    dataset_path = os.path.join(os.path.dirname(__file__), "real_user_benchmark.json")
    pairs = load_real_dataset(dataset_path)

    # Конфигурация (берём базовую и меняем нужное)
    cfg = default_benchmark_config()
    cfg["clearml_task"] = "Benchmark_RealUsers_GPT2_vs_Mistral"
    cfg["n_samples"] = len(pairs)

    # ClearML
    clearml_task, clearml_logger = None, None
    if cfg.get("use_clearml", True):
        clearml_task, clearml_logger = init_clearml(cfg)
        if clearml_task:
            clearml_task.set_name("Benchmark_RealUsers_GPT2_vs_Mistral")

    # Модели для тестирования
    models_cfg = {
        "gpt2_original": {
            "type": "hf", "model_name": "gpt2",
            "label": "GPT-2 Original (без обучения)", "params_millions": 117
        },
        "gpt2_finetuned": {
            "type": "hf_checkpoint", "checkpoint": "./checkpoints/checkpoint-best",
            "label": "GPT-2 наш (fine-tuned)", "params_millions": 117
        },
        "mistral": {
            "type": "ollama", "model_name": "mistral:7b",
            "label": "Mistral 7B (Ollama)", "params_millions": 7000
        },
    }

    all_results = []

    for model_key, model_cfg in models_cfg.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Тестируем: {model_cfg['label']}")
        logger.info(f"{'='*60}")

        result = ModelResult(
            label=model_cfg["label"],
            model_key=model_key,
            params_millions=model_cfg["params_millions"],
            n_samples=len(pairs),
        )

        # Загрузка модели
        try:
            if model_cfg["type"] == "hf":
                adapter = HFModelAdapter(model_cfg["model_name"], is_checkpoint=False)
            elif model_cfg["type"] == "hf_checkpoint":
                adapter = HFModelAdapter(model_cfg["checkpoint"], is_checkpoint=True)
            else:
                adapter = OllamaAdapter(model_cfg["model_name"])
        except Exception as e:
            logger.error(f"Не удалось загрузить {model_cfg['label']}: {e}")
            result.error = str(e)[:100]
            all_results.append(result)
            continue

        # Генерация
        hypotheses, references, gen_times = [], [], []
        category_results = {}  # результаты по категориям
        examples_log = []

        for i, pair in enumerate(pairs):
            if i % 10 == 0:
                logger.info(f"  {i}/{len(pairs)} ...")

            try:
                text, elapsed = adapter.generate(
                    context=pair["context"],
                    max_new_tokens=cfg["max_new_tokens"],
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                    top_k=cfg["top_k"],
                    repetition_penalty=cfg["repetition_penalty"],
                )
            except Exception as e:
                logger.warning(f"Ошибка генерации #{i}: {e}")
                text, elapsed = "", 0.0

            hypotheses.append(text if text else " ")
            references.append(pair["response"])
            gen_times.append(elapsed)

            cat = pair.get("category", "Other")
            if cat not in category_results:
                category_results[cat] = {"hyps": [], "refs": []}
            category_results[cat]["hyps"].append(text if text else " ")
            category_results[cat]["refs"].append(pair["response"])

            if len(examples_log) < 5:
                examples_log.append({
                    "id": pair["dialog_id"],
                    "category": cat,
                    "question": pair["context"],
                    "reference": pair["response"],
                    "generated": text,
                })

        # Метрики
        bleu = compute_bleu(hypotheses, references)
        result.bleu1 = bleu.get("bleu1", 0.0)
        result.bleu2 = bleu.get("bleu2", 0.0)
        result.bleu4 = bleu.get("bleu4", 0.0)
        result.rouge_l = compute_rouge(hypotheses, references).get("rouge_l", 0.0)

        valid_times = [t for t in gen_times if t > 0]
        result.avg_gen_time_sec = round(np.mean(valid_times), 4) if valid_times else 0.0
        avg_len = np.mean([len(h.split()) for h in hypotheses]) if hypotheses else 1
        result.tokens_per_sec = round(avg_len / max(result.avg_gen_time_sec, 0.001), 2)

        # Perplexity только для HF
        if model_cfg["type"] in ("hf", "hf_checkpoint") and isinstance(adapter, HFModelAdapter):
            try:
                from dataset import is_dialogpt
                result.perplexity = compute_hf_perplexity(
                    model=adapter.model,
                    tokenizer=adapter.tokenizer,
                    pairs=pairs,
                    device=adapter.device,
                    batch_size=cfg["perplexity_batch_size"],
                    dialogpt_mode=adapter.dialogpt_mode,
                )
            except Exception as e:
                logger.warning(f"Perplexity ошибка: {e}")

        # Метрики по категориям
        cat_metrics = {}
        for cat, data in category_results.items():
            cat_bleu = compute_bleu(data["hyps"], data["refs"])
            cat_rouge = compute_rouge(data["hyps"], data["refs"])
            cat_metrics[cat] = {
                "bleu4": cat_bleu.get("bleu4", 0.0),
                "rouge_l": cat_rouge.get("rouge_l", 0.0),
                "n": len(data["hyps"])
            }

        logger.info(f"\nРезультаты {model_cfg['label']}:")
        logger.info(f"  BLEU-1: {result.bleu1:.4f}")
        logger.info(f"  BLEU-4: {result.bleu4:.4f}")
        logger.info(f"  ROUGE-L: {result.rouge_l:.4f}")
        logger.info(f"  PPL: {result.perplexity:.2f}")
        logger.info(f"  Tok/sec: {result.tokens_per_sec:.1f}")

        logger.info(f"\n  По категориям:")
        for cat, m in sorted(cat_metrics.items(), key=lambda x: x[1]["rouge_l"], reverse=True):
            logger.info(f"    {cat:<30} BLEU-4={m['bleu4']:6.2f}  ROUGE-L={m['rouge_l']:6.2f}  n={m['n']}")

        # Логируем примеры в ClearML
        if clearml_logger:
            try:
                import pandas as pd
                df_ex = pd.DataFrame(
                    [[e["category"], e["question"], e["reference"], e["generated"]]
                     for e in examples_log],
                    columns=["Category", "Question", "Reference", "Generated"]
                )
                clearml_logger.report_table(
                    title=f"Real Examples: {model_cfg['label']}",
                    series="samples", iteration=1, table_plot=df_ex,
                )
                # Метрики по категориям
                df_cat = pd.DataFrame(
                    [[cat, m["n"], f"{m['bleu4']:.2f}", f"{m['rouge_l']:.2f}"]
                     for cat, m in cat_metrics.items()],
                    columns=["Category", "N", "BLEU-4", "ROUGE-L"]
                )
                clearml_logger.report_table(
                    title=f"By Category: {model_cfg['label']}",
                    series="categories", iteration=1, table_plot=df_cat,
                )
            except Exception as e:
                logger.warning(f"ClearML таблица ошибка: {e}")

        all_results.append(result)

    # Score
    for r in all_results:
        if r.error is None:
            r.score = compute_score(r, cfg["score_weights"], all_results)

    # Итоговая таблица
    logger.info("\n" + "="*80)
    logger.info("ИТОГОВЫЕ РЕЗУЛЬТАТЫ — РЕАЛЬНЫЕ ВОПРОСЫ ПОЛЬЗОВАТЕЛЕЙ")
    logger.info("="*80)
    logger.info(f"{'Модель':<35} {'BLEU-4':>7} {'ROUGE-L':>8} {'PPL':>8} {'SCORE':>8}")
    logger.info("-"*80)
    for r in sorted(all_results, key=lambda x: x.score, reverse=True):
        ppl_str = f"{r.perplexity:.2f}" if r.perplexity < 999 else "N/A"
        status = "✓" if r.error is None else "ERR"
        logger.info(f"{r.label:<35} {r.bleu4:>7.2f} {r.rouge_l:>8.2f} {ppl_str:>8} {r.score:>8.1f} {status}")
    logger.info("="*80)

    # Логируем в ClearML
    log_results_to_clearml(clearml_logger, all_results, cfg)

    # Сохраняем
    output = {
        "dataset": "real_user_questions",
        "n_questions": len(pairs),
        "results": [asdict(r) for r in all_results]
    }
    with open("benchmark_real_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Результаты сохранены: benchmark_real_results.json")

    if clearml_task:
        clearml_task.upload_artifact("real_benchmark_results", "benchmark_real_results.json")
        clearml_task.close()


if __name__ == "__main__":
    run_real_benchmark()
