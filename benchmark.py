"""
benchmark.py
Бенчмарк сравнения 4 моделей на Ubuntu Dialogue тестовой выборке:
  1. GPT-2 оригинальный (без fine-tuning)
  2. GPT-2 наш (fine-tuned на Ubuntu)
  3. Локальная LLM через Ollama (Llama/Mistral/Gemma)
  4. Наша обученная модель (checkpoint-best)

Метрики: BLEU-1/2/4, ROUGE-L, Perplexity, скорость генерации
Итог: итоговый Score (0-100) для каждой модели

Логирование: ClearML — скаляры, таблицы, сравнительные графики

Запуск:
    python benchmark.py --config benchmark_config.json
"""

import os
import json
import math
import time
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("benchmark.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════

def default_benchmark_config() -> Dict:
    return {
        # Данные
        "test_data": "./data/processed_filtered/test.jsonl",
        "n_samples": 200,           # сколько примеров использовать (None = все)
        "max_context_len": 200,     # макс токенов в промпте
        "max_new_tokens": 80,       # макс токенов в ответе

        # Модели
        "models": {
            "gpt2_original": {
                "enabled": True,
                "type": "hf",
                "model_name": "gpt2",
                "label": "GPT-2 Original",
                "params_millions": 117,
                "color": "#808080"
            },
            "gpt2_finetuned": {
                "enabled": True,
                "type": "hf_checkpoint",
                "checkpoint": "./checkpoints/checkpoint-best",
                "label": "GPT-2 Fine-tuned (наш)",
                "params_millions": 117,
                "color": "#2E75B6"
            },
            "ollama_llm": {
                "enabled": True,
                "type": "ollama",
                "model_name": "llama3.1:8b",  # или mistral:7b, gemma2:9b
                "label": "Llama 3.1 8B (Ollama)",
                "params_millions": 8000,
                "color": "#70AD47"
            },
            "our_model": {
                "enabled": True,
                "type": "hf_checkpoint",
                "checkpoint": "./checkpoints/checkpoint-best",
                "label": "Наша модель (best checkpoint)",
                "params_millions": 117,
                "color": "#FF6600"
            }
        },

        # Веса для итогового Score
        "score_weights": {
            "bleu4":      0.25,   # качество n-gram совпадения
            "rouge_l":    0.25,   # качество подпоследовательности
            "perplexity": 0.30,   # насколько уверена модель
            "speed":      0.20,   # скорость генерации
        },

        # ClearML
        "use_clearml": True,
        "clearml_project": "UbuntuSupportAI",
        "clearml_task": "Benchmark_4Models",

        # Генерация
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.3,

        # Perplexity — только для HF моделей
        "compute_perplexity": True,
        "perplexity_batch_size": 8,
    }


# ══════════════════════════════════════════════════════════════
# РЕЗУЛЬТАТЫ
# ══════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    label: str
    model_key: str
    params_millions: int
    bleu1: float = 0.0
    bleu2: float = 0.0
    bleu4: float = 0.0
    rouge_l: float = 0.0
    perplexity: float = 999.0
    avg_gen_time_sec: float = 0.0
    tokens_per_sec: float = 0.0
    score: float = 0.0
    n_samples: int = 0
    error: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# МЕТРИКИ
# ══════════════════════════════════════════════════════════════

def compute_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    try:
        from sacrebleu.metrics import BLEU
        results = {}
        for n in [1, 2, 4]:
            bleu = BLEU(max_ngram_order=n)
            score = bleu.corpus_score(hypotheses, [references])
            results[f"bleu{n}"] = round(score.score, 4)
        return results
    except ImportError:
        # Простой fallback
        from collections import Counter
        def ngrams(tokens, n):
            return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        results = {}
        for n in [1, 2, 4]:
            scores = []
            for hyp, ref in zip(hypotheses, references):
                h = hyp.lower().split()
                r = ref.lower().split()
                if not h:
                    scores.append(0.0)
                    continue
                h_ng = ngrams(h, n)
                r_ng = ngrams(r, n)
                matches = sum(min(h_ng[k], r_ng[k]) for k in h_ng)
                total = sum(h_ng.values())
                scores.append(matches / max(total, 1))
            results[f"bleu{n}"] = round(np.mean(scores) * 100, 4)
        return results


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        f1s = [scorer.score(ref, hyp)["rougeL"].fmeasure
               for hyp, ref in zip(hypotheses, references)]
        return {"rouge_l": round(np.mean(f1s) * 100, 4)}
    except ImportError:
        # LCS fallback
        def lcs(a, b):
            a, b = a.lower().split(), b.lower().split()
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(1, m+1):
                for j in range(1, n+1):
                    dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        scores = []
        for hyp, ref in zip(hypotheses, references):
            l = lcs(hyp, ref)
            p = l / max(len(hyp.split()), 1)
            r = l / max(len(ref.split()), 1)
            f1 = 2*p*r/max(p+r, 1e-8)
            scores.append(f1)
        return {"rouge_l": round(np.mean(scores) * 100, 4)}


def compute_hf_perplexity(
    model, tokenizer, pairs: List[Dict],
    device, batch_size: int = 8, dialogpt_mode: bool = False
) -> float:
    """Считает perplexity для HuggingFace модели."""
    from dataset import encode_pair
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        batch_input_ids, batch_labels, batch_masks = [], [], []

        for pair in batch_pairs:
            enc = encode_pair(
                context=pair["context"],
                response=pair["response"],
                tokenizer=tokenizer,
                max_length=256,
                dialogpt_mode=dialogpt_mode,
            )
            batch_input_ids.append(enc["input_ids"])
            batch_labels.append(enc["labels"])
            batch_masks.append(enc["attention_mask"])

        input_ids = torch.stack(batch_input_ids).to(device)
        labels = torch.stack(batch_labels).to(device)
        attention_mask = torch.stack(batch_masks).to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return round(math.exp(avg_loss), 4) if avg_loss < 20 else 999.0


# ══════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════

def load_test_pairs(path: str, n: Optional[int] = None) -> List[Dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info(f"Загружено {len(pairs)} тестовых пар из {path}")
    return pairs


# ══════════════════════════════════════════════════════════════
# АДАПТЕРЫ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════

class HFModelAdapter:
    """Адаптер для HuggingFace GPT-2 / fine-tuned checkpoint."""

    def __init__(self, model_name_or_path: str, is_checkpoint: bool = False):
        from transformers import AutoTokenizer
        from model import SupportGPT2
        from dataset import build_tokenizer, is_dialogpt

        if is_checkpoint:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = SupportGPT2.load(
                model_name_or_path, vocab_size=len(self.tokenizer)
            )
            # Читаем имя модели из meta.json для определения режима
            meta_path = os.path.join(model_name_or_path, "meta.json")
            model_name = "gpt2"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                model_name = meta.get("config", {}).get("model_name", "gpt2")
        else:
            self.tokenizer = build_tokenizer(model_name_or_path)
            self.model = SupportGPT2(
                model_name=model_name_or_path,
                vocab_size=len(self.tokenizer)
            )
            model_name = model_name_or_path

        self.dialogpt_mode = is_dialogpt(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"HF модель загружена: {model_name_or_path} | device={self.device}")

    def generate(self, context: str, max_new_tokens: int = 80,
                 temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 50, repetition_penalty: float = 1.3) -> Tuple[str, float]:
        """Генерирует ответ. Возвращает (текст, время_сек)."""
        from dataset import CONTEXT_TOKEN, RESPONSE_TOKEN

        eos = self.tokenizer.eos_token or ""

        if self.dialogpt_mode:
            turns = context.split(" [SEP] ")
            prompt = "".join(t.strip() + eos for t in turns if t.strip())
        else:
            bos = self.tokenizer.bos_token or ""
            prompt = f"{bos}{CONTEXT_TOKEN} {context} {RESPONSE_TOKEN}"

        enc = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=200
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        t0 = time.time()
        new_tokens = self.model.generate_response(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0

        text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        text = text.split("\n")[0].strip()
        return text, elapsed


class OllamaAdapter:
    """Адаптер для локальных моделей через Ollama API."""

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self._check_connection()

    def _check_connection(self):
        try:
            import requests
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model_name in m for m in models):
                logger.warning(
                    f"Модель '{self.model_name}' не найдена в Ollama. "
                    f"Доступны: {models}. "
                    f"Запустите: ollama pull {self.model_name}"
                )
            else:
                logger.info(f"Ollama: модель {self.model_name} готова")
        except Exception as e:
            logger.warning(f"Ollama недоступен: {e}. Убедитесь что запущен: ollama serve")

    def generate(self, context: str, max_new_tokens: int = 80,
                 temperature: float = 0.7, **kwargs) -> Tuple[str, float]:
        """Генерирует ответ через Ollama REST API."""
        import requests

        # Формируем системный промпт для техподдержки Ubuntu
        system = (
            "You are a helpful Ubuntu/Linux technical support assistant. "
            "Give short, direct, technical answers. "
            "Use commands like sudo apt-get, systemctl etc when appropriate."
        )

        # Контекст → последний вопрос пользователя
        turns = context.split(" [SEP] ")
        user_message = turns[-1].strip() if turns else context

        prompt_data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
            }
        }

        t0 = time.time()
        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json=prompt_data,
                timeout=60
            )
            r.raise_for_status()
            data = r.json()
            text = data.get("message", {}).get("content", "").strip()
            # Берём только первое предложение для сравнимости
            text = text.split("\n")[0].strip()[:200]
        except Exception as e:
            logger.error(f"Ollama ошибка: {e}")
            text = ""
        elapsed = time.time() - t0
        return text, elapsed


# ══════════════════════════════════════════════════════════════
# СИСТЕМА ОЧКОВ
# ══════════════════════════════════════════════════════════════

def compute_score(result: ModelResult, weights: Dict, all_results: List[ModelResult]) -> float:
    """
    Считает итоговый Score (0-100) для модели.
    Нормализует метрики относительно лучшей модели в сравнении.

    Принцип нормализации:
    - BLEU, ROUGE: выше = лучше → нормализуем к максимуму
    - Perplexity: ниже = лучше → инвертируем
    - Speed: выше tok/sec = лучше → нормализуем к максимуму
    """
    valid = [r for r in all_results if r.error is None]
    if not valid:
        return 0.0

    def normalize_high(val, all_vals):
        """Нормализует метрику где выше = лучше."""
        max_v = max(all_vals) if max(all_vals) > 0 else 1.0
        return val / max_v

    def normalize_low(val, all_vals):
        """Нормализует метрику где ниже = лучше (perplexity)."""
        min_v = min(all_vals) if min(all_vals) > 0 else 1.0
        # Инвертируем: лучший получает 1.0
        return min_v / max(val, 1.0)

    all_bleu4    = [r.bleu4 for r in valid]
    all_rouge    = [r.rouge_l for r in valid]
    all_ppl      = [r.perplexity for r in valid if r.perplexity < 999]
    all_speed    = [r.tokens_per_sec for r in valid]

    bleu4_norm = normalize_high(result.bleu4, all_bleu4)
    rouge_norm = normalize_high(result.rouge_l, all_rouge)
    ppl_norm   = normalize_low(result.perplexity, all_ppl) if all_ppl else 0.5
    speed_norm = normalize_high(result.tokens_per_sec, all_speed) if max(all_speed) > 0 else 0.5

    score = (
        weights.get("bleu4", 0.25)      * bleu4_norm +
        weights.get("rouge_l", 0.25)    * rouge_norm +
        weights.get("perplexity", 0.30) * ppl_norm   +
        weights.get("speed", 0.20)      * speed_norm
    ) * 100

    return round(score, 2)


# ══════════════════════════════════════════════════════════════
# ClearML ЛОГИРОВАНИЕ
# ══════════════════════════════════════════════════════════════

def init_clearml(cfg: Dict):
    try:
        from clearml import Task
        task = Task.init(
            project_name=cfg["clearml_project"],
            task_name=cfg["clearml_task"],
            reuse_last_task_id=False,
        )
        task.connect(cfg, name="benchmark_config")
        clearml_logger = task.get_logger()
        logger.info(f"ClearML Task ID: {task.id}")
        logger.info(f"ClearML URL: https://app.clear.ml/projects/*/experiments/{task.id}/output/log")
        return task, clearml_logger
    except ImportError:
        logger.warning("ClearML не установлен")
        return None, None
    except Exception as e:
        logger.warning(f"ClearML ошибка: {e}")
        return None, None


def log_results_to_clearml(clearml_logger, results: List[ModelResult], cfg: Dict):
    """Логирует все метрики и сравнительные таблицы в ClearML."""
    if clearml_logger is None:
        return

    # 1. Скалярные графики — каждая метрика отдельным графиком
    for i, r in enumerate(results):
        if r.error:
            continue
        iteration = i  # используем индекс как iteration для группировки

        metrics = {
            "BLEU-1":      r.bleu1,
            "BLEU-2":      r.bleu2,
            "BLEU-4":      r.bleu4,
            "ROUGE-L":     r.rouge_l,
            "Score":       r.score,
        }
        for metric_name, value in metrics.items():
            clearml_logger.report_scalar(
                title=metric_name,
                series=r.label,
                value=value,
                iteration=1,
            )

        # Perplexity (инвертируем для графика — меньше = лучше)
        if r.perplexity < 999:
            clearml_logger.report_scalar(
                title="Perplexity",
                series=r.label,
                value=r.perplexity,
                iteration=1,
            )

        # Скорость генерации
        clearml_logger.report_scalar(
            title="Tokens per Second",
            series=r.label,
            value=r.tokens_per_sec,
            iteration=1,
        )

        # Параметры модели vs Score (для scatter)
        clearml_logger.report_scalar(
            title="Score vs Params",
            series=r.label,
            value=r.score,
            iteration=r.params_millions,
        )

    # 2. Итоговая таблица сравнения
    table_data = []
    for r in sorted(results, key=lambda x: x.score, reverse=True):
        params_str = f"{r.params_millions}M" if r.params_millions < 1000 else f"{r.params_millions//1000}B"
        table_data.append([
            r.label,
            params_str,
            f"{r.bleu1:.2f}",
            f"{r.bleu2:.2f}",
            f"{r.bleu4:.2f}",
            f"{r.rouge_l:.2f}",
            f"{r.perplexity:.2f}" if r.perplexity < 999 else "N/A",
            f"{r.tokens_per_sec:.1f}",
            f"{r.score:.1f}",
            "✓" if r.error is None else f"✗ {r.error[:30]}",
        ])

    try:
        import pandas as pd
        df = pd.DataFrame(
            table_data,
            columns=["Модель", "Параметры", "BLEU-1", "BLEU-2", "BLEU-4",
                     "ROUGE-L", "Perplexity", "Tok/sec", "Score (0-100)", "Статус"]
        )
        clearml_logger.report_table(
            title="Benchmark Results",
            series="Comparison Table",
            iteration=1,
            table_plot=df,
        )
    except Exception as e:
        logger.warning(f"Не удалось залогировать таблицу результатов: {e}")

    # 3. Гистограмма Score
    scores = {r.label: r.score for r in results if r.error is None}
    clearml_logger.report_histogram(
        title="Final Score Comparison",
        series="Score",
        values=list(scores.values()),
        xlabels=list(scores.keys()),
        iteration=1,
    )

    # 4. Примеры генерации всех моделей
    if hasattr(clearml_logger, "_examples_log"):
        clearml_logger.report_text(clearml_logger._examples_log, print_console=False)

    logger.info("Результаты залогированы в ClearML")


# ══════════════════════════════════════════════════════════════
# ОСНОВНОЙ БЕНЧМАРК
# ══════════════════════════════════════════════════════════════

def run_model_benchmark(
    model_key: str,
    model_cfg: Dict,
    pairs: List[Dict],
    cfg: Dict,
    clearml_logger,
) -> ModelResult:
    """Запускает бенчмарк для одной модели."""

    result = ModelResult(
        label=model_cfg["label"],
        model_key=model_key,
        params_millions=model_cfg.get("params_millions", 0),
        n_samples=len(pairs),
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Бенчмарк: {model_cfg['label']}")
    logger.info(f"{'='*60}")

    # ── Загрузка модели ──
    try:
        model_type = model_cfg["type"]
        if model_type == "hf":
            adapter = HFModelAdapter(model_cfg["model_name"], is_checkpoint=False)
        elif model_type == "hf_checkpoint":
            adapter = HFModelAdapter(model_cfg["checkpoint"], is_checkpoint=True)
        elif model_type == "ollama":
            adapter = OllamaAdapter(model_cfg["model_name"])
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
    except Exception as e:
        logger.error(f"Ошибка загрузки {model_cfg['label']}: {e}")
        result.error = str(e)[:100]
        return result

    # ── Генерация ответов ──
    hypotheses = []
    references = []
    gen_times = []
    examples_log = []

    logger.info(f"Генерация на {len(pairs)} примерах...")
    for i, pair in enumerate(pairs):
        if i % 50 == 0:
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
            logger.warning(f"Ошибка генерации пример {i}: {e}")
            text = ""
            elapsed = 0.0

        hypotheses.append(text if text else " ")
        references.append(pair["response"])
        gen_times.append(elapsed)

        # Сохраняем первые 5 примеров для логирования
        if len(examples_log) < 5:
            examples_log.append({
                "context": pair["context"][:100],
                "reference": pair["response"],
                "generated": text,
                "time": f"{elapsed:.2f}s",
            })

    # ── Метрики ──
    logger.info("Считаю метрики...")

    bleu = compute_bleu(hypotheses, references)
    result.bleu1 = bleu.get("bleu1", 0.0)
    result.bleu2 = bleu.get("bleu2", 0.0)
    result.bleu4 = bleu.get("bleu4", 0.0)

    rouge = compute_rouge(hypotheses, references)
    result.rouge_l = rouge.get("rouge_l", 0.0)

    # Скорость
    valid_times = [t for t in gen_times if t > 0]
    avg_time = np.mean(valid_times) if valid_times else 0.0
    result.avg_gen_time_sec = round(avg_time, 4)
    # Приблизительно tokens/sec
    avg_response_len = np.mean([len(h.split()) for h in hypotheses]) if hypotheses else 1
    result.tokens_per_sec = round(avg_response_len / max(avg_time, 0.001), 2)

    # Perplexity (только для HF моделей)
    if (cfg.get("compute_perplexity", True)
            and model_type in ("hf", "hf_checkpoint")
            and isinstance(adapter, HFModelAdapter)):
        logger.info("Считаю Perplexity...")
        try:
            from dataset import is_dialogpt
            result.perplexity = compute_hf_perplexity(
                model=adapter.model,
                tokenizer=adapter.tokenizer,
                pairs=pairs[:min(100, len(pairs))],  # лимит для скорости
                device=adapter.device,
                batch_size=cfg["perplexity_batch_size"],
                dialogpt_mode=adapter.dialogpt_mode,
            )
        except Exception as e:
            logger.warning(f"Не удалось посчитать Perplexity: {e}")
            result.perplexity = 999.0
    else:
        result.perplexity = 999.0  # Ollama — нет доступа к logits

    # ── Вывод результатов ──
    logger.info(f"\nРезультаты {model_cfg['label']}:")
    logger.info(f"  BLEU-1:   {result.bleu1:.4f}")
    logger.info(f"  BLEU-2:   {result.bleu2:.4f}")
    logger.info(f"  BLEU-4:   {result.bleu4:.4f}")
    logger.info(f"  ROUGE-L:  {result.rouge_l:.4f}")
    logger.info(f"  PPL:      {result.perplexity:.2f}")
    logger.info(f"  Tok/sec:  {result.tokens_per_sec:.1f}")
    logger.info(f"  Avg time: {result.avg_gen_time_sec:.3f}s")

    # Логируем примеры в ClearML
    if clearml_logger:
        try:
            import pandas as pd
            ex_rows = [[e["context"], e["reference"], e["generated"], e["time"]]
                       for e in examples_log]
            df_ex = pd.DataFrame(ex_rows, columns=["Context", "Reference", "Generated", "Time"])
            clearml_logger.report_table(
                title=f"Examples: {model_cfg['label']}",
                series="generation_samples",
                iteration=1,
                table_plot=df_ex,
            )
        except Exception as e:
            logger.warning(f"Не удалось залогировать примеры генерации: {e}")

    return result


def run_benchmark(cfg: Dict):
    """Основная функция бенчмарка."""

    # ClearML
    clearml_task, clearml_logger = None, None
    if cfg.get("use_clearml", True):
        clearml_task, clearml_logger = init_clearml(cfg)

    # Загрузка данных
    pairs = load_test_pairs(cfg["test_data"], cfg.get("n_samples"))

    # Запускаем бенчмарк для каждой модели
    all_results: List[ModelResult] = []

    for model_key, model_cfg in cfg["models"].items():
        if not model_cfg.get("enabled", True):
            logger.info(f"Пропускаем {model_cfg['label']} (disabled)")
            continue

        result = run_model_benchmark(
            model_key=model_key,
            model_cfg=model_cfg,
            pairs=pairs,
            cfg=cfg,
            clearml_logger=clearml_logger,
        )
        all_results.append(result)

    # Считаем Score для всех моделей
    logger.info("\nСчитаю итоговые Score...")
    for result in all_results:
        if result.error is None:
            result.score = compute_score(result, cfg["score_weights"], all_results)

    # Итоговая таблица в консоль
    logger.info("\n" + "="*80)
    logger.info("ИТОГОВЫЕ РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    logger.info("="*80)
    logger.info(f"{'Модель':<35} {'Params':<8} {'BLEU-4':>7} {'ROUGE-L':>8} {'PPL':>8} {'Tok/s':>8} {'SCORE':>8}")
    logger.info("-"*80)

    sorted_results = sorted(all_results, key=lambda x: x.score, reverse=True)
    for r in sorted_results:
        params_str = f"{r.params_millions}M" if r.params_millions < 1000 else f"{r.params_millions//1000}B"
        ppl_str = f"{r.perplexity:.2f}" if r.perplexity < 999 else "N/A"
        status = "✓" if r.error is None else f"ERR"
        logger.info(
            f"{r.label:<35} {params_str:<8} "
            f"{r.bleu4:>7.2f} {r.rouge_l:>8.2f} {ppl_str:>8} "
            f"{r.tokens_per_sec:>8.1f} {r.score:>8.1f} {status}"
        )

    logger.info("="*80)
    winner = sorted_results[0] if sorted_results else None
    if winner:
        logger.info(f"\nПОБЕДИТЕЛЬ: {winner.label} с Score = {winner.score:.1f}/100")

    # Логируем в ClearML
    log_results_to_clearml(clearml_logger, all_results, cfg)

    # Сохраняем результаты в JSON
    results_path = "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg,
                "results": [asdict(r) for r in all_results],
                "winner": asdict(winner) if winner else None,
            },
            f, indent=2, ensure_ascii=False
        )
    logger.info(f"\nРезультаты сохранены: {results_path}")

    if clearml_task:
        clearml_task.upload_artifact("benchmark_results", results_path)
        clearml_task.close()

    return all_results


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Бенчмарк 4 моделей на Ubuntu Dialogue")
    parser.add_argument("--config", type=str, default="benchmark_config.json")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--no_clearml", action="store_true")
    parser.add_argument("--ollama_model", type=str, default=None,
                        help="Переопределить модель Ollama: llama3.1:8b / mistral:7b / gemma2:9b")
    parser.add_argument("--our_checkpoint", type=str, default=None,
                        help="Путь к нашему checkpoint-best")
    args = parser.parse_args()

    # Загрузка конфига
    cfg = default_benchmark_config()
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f))
        logger.info(f"Конфиг загружен из {args.config}")
    else:
        logger.info("Используем дефолтный конфиг")

    # Переопределения из CLI
    if args.n_samples:
        cfg["n_samples"] = args.n_samples
    if args.no_clearml:
        cfg["use_clearml"] = False
    if args.ollama_model:
        cfg["models"]["ollama_llm"]["model_name"] = args.ollama_model
    if args.our_checkpoint:
        cfg["models"]["gpt2_finetuned"]["checkpoint"] = args.our_checkpoint
        cfg["models"]["our_model"]["checkpoint"] = args.our_checkpoint

    logger.info(f"Конфигурация:\n{json.dumps(cfg, indent=2)}")
    run_benchmark(cfg)