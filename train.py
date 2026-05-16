"""
train.py
Обучение GPT-2 на Ubuntu Dialogue Corpus.
Логирование через ClearML: скаляры, гистограммы, таблицы примеров, артефакты.
"""

import os
import json
import math
import time
import argparse
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast

from dataset import build_tokenizer, get_dataloaders
from model import SupportGPT2

# ──────────────────────────────────────────────
# Стандартный logging (файл + консоль)
# ──────────────────────────────────────────────
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
            if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8"
            else sys.stdout
        ),
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
py_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ClearML — инициализация
# ──────────────────────────────────────────────

def init_clearml(cfg: Dict):
    """
    Инициализирует ClearML Task.
    Возвращает (task, clearml_logger) или (None, None) если ClearML недоступен.
    """
    try:
        from clearml import Task
        task = Task.init(
            project_name=cfg.get("clearml_project", "UbuntuSupportAI"),
            task_name=cfg.get("clearml_task", "GPT2_SupportBot"),
            reuse_last_task_id=False,
        )
        # Передаём весь конфиг как гиперпараметры — видны в UI на вкладке CONFIGURATION
        task.connect(cfg, name="config")
        clearml_logger = task.get_logger()
        py_logger.info(f"ClearML инициализирован. Task ID: {task.id}")
        return task, clearml_logger
    except ImportError:
        py_logger.warning("ClearML не установлен (pip install clearml). Логирование отключено.")
        return None, None
    except Exception as e:
        py_logger.warning(f"ClearML не удалось инициализировать: {e}")
        return None, None


# ──────────────────────────────────────────────
# ClearML — репортеры
# ──────────────────────────────────────────────

def log_scalar(clearml_logger, title: str, series: str, value: float, iteration: int):
    """Логирует одно скалярное значение в ClearML."""
    if clearml_logger is None:
        return
    try:
        clearml_logger.report_scalar(
            title=title, series=series, value=value, iteration=iteration
        )
    except Exception:
        pass


def log_scalars_dict(clearml_logger, metrics: Dict[str, float], title: str, iteration: int):
    """
    Логирует словарь метрик под одним title.
    Например: title="Train", metrics={"loss": 2.3, "perplexity": 10.1}
    Создаёт один граф с двумя series — удобно сравнивать train vs val.
    """
    if clearml_logger is None:
        return
    for series, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(value):
            log_scalar(clearml_logger, title=title, series=series,
                       value=value, iteration=iteration)


def log_text(clearml_logger, message: str):
    """Логирует текст (виден в ClearML Console)."""
    if clearml_logger is None:
        return
    try:
        clearml_logger.report_text(message, print_console=False)
    except Exception:
        pass


def log_histogram(clearml_logger, title: str, series: str, values, iteration: int):
    """Логирует гистограмму значений (распределение весов/градиентов)."""
    if clearml_logger is None:
        return
    try:
        arr = values.detach().cpu().float().numpy() if hasattr(values, "detach") else values
        clearml_logger.report_histogram(
            title=title, series=series,
            values=arr, iteration=iteration,
        )
    except Exception:
        pass


def log_table(clearml_logger, title: str, headers: list, rows: list, iteration: int):
    """Логирует таблицу (примеры генерации: контекст / референс / ответ модели)."""
    if clearml_logger is None:
        return
    try:
        clearml_logger.report_table(
            title=title, series="samples",
            iteration=iteration,
            table_plot={"headers": headers, "data": rows},
        )
    except Exception:
        pass


def upload_artifact(task, name: str, path: str):
    """Загружает файл/директорию как артефакт задачи в ClearML."""
    if task is None:
        return
    try:
        task.upload_artifact(name=name, artifact_object=path)
        py_logger.info(f"Артефакт '{name}' загружен: {path}")
    except Exception as e:
        py_logger.warning(f"Не удалось загрузить артефакт '{name}': {e}")


# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

def default_config() -> Dict:
    return {
        # Данные
        "data_dir": "./data/processed",
        "model_name": "gpt2",
        "max_length": 256,
        "max_train_samples": None,

        # Обучение
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_epochs": 5,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "freeze_layers": 0,

        # Инфраструктура
        "output_dir": "./checkpoints",
        "save_every_n_steps": 500,
        "eval_every_n_steps": 500,
        "patience": 3,
        "fp16": True,
        "num_workers": 2,
        "seed": 42,

        # Resume
        "resume_from": None,   # путь к checkpoint-у, например "./checkpoints/checkpoint-latest"

        # ClearML
        "use_clearml": True,
        "clearml_project": "UbuntuSupportAI",
        "clearml_task": "GPT2_SupportBot",
        "log_grad_histogram_every": 200,
        "log_examples_every": 500,
        "n_log_examples": 3,

        # Лимиты evaluation — ВАЖНО для больших датасетов!
        # None = все примеры (1.8 млн на CPU = ~6 часов)
        "max_val_samples": 2000,
        "max_test_samples": 2000,
    }


# ──────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        py_logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        py_logger.info("Apple MPS")
    else:
        device = torch.device("cpu")
        py_logger.info("CPU (медленно для больших данных)")
    return device


def perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def grad_norm(model: nn.Module) -> float:
    """Суммарная L2-норма градиентов (для мониторинга взрыва/затухания)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return math.sqrt(total)


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: SupportGPT2, loader, device: torch.device, fp16: bool) -> Dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast('cuda', enabled=fp16 and device.type == "cuda"):
            outputs = model(input_ids, attention_mask, labels=labels)

        total_loss += outputs.loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return {"loss": avg_loss, "perplexity": perplexity(avg_loss)}


# ──────────────────────────────────────────────
# Генерация примеров для мониторинга
# ──────────────────────────────────────────────

def generate_examples(model, tokenizer, val_data_path, device, n=3, max_new_tokens=60):
    """Генерирует N примеров ответов из val set для визуального контроля."""
    import json as _json
    from dataset import CONTEXT_TOKEN, RESPONSE_TOKEN

    model.eval()
    examples = []

    with open(val_data_path) as f:
        items = [_json.loads(l) for i, l in enumerate(f) if i < n * 4]

    for item in items[:n]:
        context = item["context"]
        reference = item["response"]

        bos = tokenizer.bos_token or ""
        prompt = f"{bos}{CONTEXT_TOKEN} {context} {RESPONSE_TOKEN}"
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model.generate_response(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        generated = generated.split("\n")[0].strip()

        examples.append({
            "context": context[:120],
            "reference": reference,
            "generated": generated,
        })

    model.train()
    return examples


# ──────────────────────────────────────────────
# Логирование примеров (консоль + ClearML)
# ──────────────────────────────────────────────

def log_generation_examples(
    model, tokenizer, val_data_path, device, clearml_logger, step, n=3
):
    try:
        examples = generate_examples(model, tokenizer, val_data_path, device, n=n)

        # Консоль
        lines = [f"\n{'─'*60}", f"Примеры генерации @ step {step}:"]
        for i, ex in enumerate(examples, 1):
            lines += [
                f"  [{i}] Контекст:  {ex['context'][:100]}",
                f"       Референс: {ex['reference']}",
                f"       Генерация: {ex['generated']}",
                "",
            ]
        lines.append("─" * 60)
        msg = "\n".join(lines)
        py_logger.info(msg)
        log_text(clearml_logger, msg)

        # ClearML: таблица
        log_table(
            clearml_logger,
            title="Generation Examples",
            headers=["context", "reference", "generated"],
            rows=[[ex["context"][:80], ex["reference"], ex["generated"]] for ex in examples],
            iteration=step,
        )
    except Exception as e:
        py_logger.warning(f"Не удалось залогировать примеры: {e}")


# ──────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────

def save_checkpoint(
    model, tokenizer, optimizer, scheduler,
    cfg, step, epoch, best_val_loss, output_dir, tag="latest",
    clearml_task=None, scaler=None,
):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # Сохраняем состояние оптимизатора, scheduler и scaler для resume
    train_state = {
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler":    scaler.state_dict() if scaler else None,
    }
    torch.save(train_state, os.path.join(ckpt_dir, "train_state.pt"))

    meta = {
        "step": step, "epoch": epoch,
        "best_val_loss": best_val_loss, "config": cfg,
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    py_logger.info(f"Checkpoint [{tag}] сохранён: {ckpt_dir}")

    if tag == "best":
        upload_artifact(clearml_task, "best_checkpoint", ckpt_dir)


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, scaler, device):
    """
    Загружает веса + состояние оптимизатора/scheduler/scaler из чекпоинта.
    Возвращает (start_epoch, global_step, best_val_loss).
    """
    meta_path = os.path.join(ckpt_dir, "meta.json")
    state_path = os.path.join(ckpt_dir, "train_state.pt")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json не найден в {ckpt_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    # Загружаем веса модели
    from model import SupportGPT2
    loaded = SupportGPT2.load(ckpt_dir)
    model.model.load_state_dict(loaded.model.state_dict())
    model.to(device)
    py_logger.info(f"Веса модели загружены из {ckpt_dir}")

    # Загружаем состояние оптимизатора и scheduler
    if os.path.exists(state_path):
        train_state = torch.load(state_path, map_location=device, weights_only=True)
        if train_state.get("optimizer") and optimizer:
            optimizer.load_state_dict(train_state["optimizer"])
        if train_state.get("scheduler") and scheduler:
            scheduler.load_state_dict(train_state["scheduler"])
        if train_state.get("scaler") and scaler:
            scaler.load_state_dict(train_state["scaler"])
        py_logger.info("Состояние оптимизатора/scheduler восстановлено")
    else:
        py_logger.warning("train_state.pt не найден — только веса модели загружены")

    start_epoch = meta["epoch"]
    global_step  = meta["step"]
    best_val_loss = meta["best_val_loss"]

    py_logger.info(
        f"Resume: эпоха {start_epoch} | step={global_step} | best_val_loss={best_val_loss:.4f}"
    )
    return start_epoch, global_step, best_val_loss


# ──────────────────────────────────────────────
# Финальная оценка
# ──────────────────────────────────────────────

def run_final_test(
    model, tokenizer, loaders, device, cfg,
    val_data_path, clearml_logger, clearml_task, step
):
    py_logger.info("=" * 60)
    py_logger.info("ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET")
    py_logger.info("=" * 60)

    test_metrics = evaluate(model, loaders["test"], device, cfg["fp16"])

    py_logger.info(
        f"[TEST] loss={test_metrics['loss']:.4f} | "
        f"perplexity={test_metrics['perplexity']:.2f}"
    )

    log_scalars_dict(clearml_logger, {
        "loss":       test_metrics["loss"],
        "perplexity": test_metrics["perplexity"],
    }, title="Test", iteration=step)

    log_text(
        clearml_logger,
        f"Final test | loss={test_metrics['loss']:.4f} | ppl={test_metrics['perplexity']:.2f}",
    )

    if os.path.exists(val_data_path):
        log_generation_examples(
            model, tokenizer, val_data_path, device, clearml_logger, step, n=5
        )

    save_checkpoint(
        model, tokenizer, None, None,
        cfg, step, cfg["num_epochs"], test_metrics["loss"],
        cfg["output_dir"], tag="final",
        clearml_task=clearml_task,
    )
    return test_metrics


# ──────────────────────────────────────────────
# Основной цикл обучения
# ──────────────────────────────────────────────

def train(cfg: Dict):
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ── ClearML ──
    clearml_task, clearml_logger = None, None
    if cfg.get("use_clearml", True):
        clearml_task, clearml_logger = init_clearml(cfg)

    # ── Данные ──
    tokenizer = build_tokenizer(cfg["model_name"])
    loaders = get_dataloaders(
        data_dir=cfg["data_dir"],
        tokenizer=tokenizer,
        batch_size=cfg["batch_size"],
        max_length=cfg["max_length"],
        max_train_samples=cfg["max_train_samples"],
        max_val_samples=cfg.get("max_val_samples", 2000),
        max_test_samples=cfg.get("max_test_samples", 2000),
        num_workers=cfg["num_workers"],
    )
    val_data_path = os.path.join(cfg["data_dir"], "val.jsonl")

    # ── Модель ──
    model = SupportGPT2(
        model_name=cfg["model_name"],
        vocab_size=len(tokenizer),
        freeze_layers=cfg["freeze_layers"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    py_logger.info(f"Параметры: {total_params:,} всего, {trainable_params:,} обучаемых")
    log_scalar(clearml_logger, "Model", "total_parameters", total_params, 0)
    log_scalar(clearml_logger, "Model", "trainable_parameters", trainable_params, 0)

    # ── Оптимизатор ──
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=cfg["learning_rate"])

    # ── Scheduler: linear warmup + cosine decay ──
    # total_steps считается по ВСЕМ эпохам (включая уже пройденные при resume),
    # чтобы LR schedule был правильно масштабирован
    steps_per_epoch = len(loaders["train"]) // cfg["gradient_accumulation_steps"]
    total_steps = steps_per_epoch * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-7),
        ],
        milestones=[warmup_steps],
    )

    scaler = GradScaler('cuda') if cfg["fp16"] and device.type == "cuda" else None

    # ── Состояние ──
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    step_loss_history = []
    start_epoch = 1

    # ── Resume из чекпоинта ──
    resume_from = cfg.get("resume_from")
    if resume_from and os.path.isdir(resume_from):
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1  # продолжаем со СЛЕДУЮЩЕЙ эпохи
        patience_counter = 0  # сбрасываем patience при resume
        py_logger.info(f"Продолжаем обучение с эпохи {start_epoch} | global_step={global_step}")
        py_logger.info(f"LR после restore: {scheduler.get_last_lr()[0]:.2e}")
    elif resume_from:
        py_logger.warning(f"resume_from указан, но папка не найдена: {resume_from}")

    py_logger.info(
        f"Начало обучения | модель={cfg['model_name']} | "
        f"эпох={cfg['num_epochs']} | total_steps={total_steps} | "
        f"warmup={warmup_steps} | device={device} | fp16={cfg['fp16']}"
    )
    log_text(
        clearml_logger,
        f"Training started | model={cfg['model_name']} | steps={total_steps} | device={device}"
    )

    # ════════════════════════════════════════════
    # ЭПОХИ
    # ════════════════════════════════════════════
    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(loaders["train"]):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            with autocast('cuda', enabled=(scaler is not None)):
                outputs = model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss / cfg["gradient_accumulation_steps"]

            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * cfg["gradient_accumulation_steps"]
            epoch_steps += 1

            # ── Optimizer step ──
            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                gn_before_clip = grad_norm(model)

                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                lr = scheduler.get_last_lr()[0]
                step_loss = epoch_loss / epoch_steps
                step_ppl = perplexity(step_loss)

                # ── Логирование каждые 50 шагов ──
                if global_step % 50 == 0:
                    py_logger.info(
                        f"Epoch {epoch}/{cfg['num_epochs']} | "
                        f"Step {global_step}/{total_steps} | "
                        f"loss={step_loss:.4f} | ppl={step_ppl:.2f} | "
                        f"grad_norm={gn_before_clip:.4f} | lr={lr:.2e}"
                    )

                    # ClearML: train скаляры (loss и ppl на одном графике)
                    log_scalars_dict(clearml_logger, {
                        "loss":       step_loss,
                        "perplexity": step_ppl,
                    }, title="Train", iteration=global_step)

                    # ClearML: LR и grad norm — отдельные графики
                    log_scalar(clearml_logger, "Learning Rate", "lr", lr, global_step)
                    log_scalar(clearml_logger,
                               "Gradient Norm", "before_clip", gn_before_clip, global_step)

                    step_loss_history.append({"step": global_step, "loss": step_loss})

                # ── Гистограммы градиентов ──
                grad_hist_every = cfg.get("log_grad_histogram_every", 200)
                if grad_hist_every > 0 and global_step % grad_hist_every == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None and "weight" in name:
                            log_histogram(
                                clearml_logger,
                                title="Weight Gradients",
                                series=name.replace("model.transformer.", "")[:40],
                                values=param.grad,
                                iteration=global_step,
                            )

                # ── Validation ──
                if global_step % cfg["eval_every_n_steps"] == 0:
                    val_metrics = evaluate(model, loaders["val"], device, cfg["fp16"])

                    py_logger.info(
                        f"[VAL] Step {global_step} | "
                        f"loss={val_metrics['loss']:.4f} | "
                        f"ppl={val_metrics['perplexity']:.2f} | "
                        f"best={best_val_loss:.4f} | patience={patience_counter}/{cfg['patience']}"
                    )

                    # ClearML: val скаляры (та же ось loss, что и train → легко сравнить)
                    log_scalars_dict(clearml_logger, {
                        "loss":       val_metrics["loss"],
                        "perplexity": val_metrics["perplexity"],
                    }, title="Validation", iteration=global_step)

                    # Early stopping
                    if val_metrics["loss"] < best_val_loss:
                        delta = best_val_loss - val_metrics["loss"]
                        best_val_loss = val_metrics["loss"]
                        patience_counter = 0
                        py_logger.info(
                            f"  ✓ Val loss улучшился на {delta:.4f} → сохраняем best"
                        )
                        log_scalar(clearml_logger,
                                   "Validation", "best_loss", best_val_loss, global_step)
                        save_checkpoint(
                            model, tokenizer, optimizer, scheduler,
                            cfg, global_step, epoch, best_val_loss,
                            cfg["output_dir"], tag="best",
                            clearml_task=clearml_task, scaler=scaler,
                        )
                    else:
                        patience_counter += 1
                        py_logger.info(
                            f"  ✗ Нет улучшения. Patience: {patience_counter}/{cfg['patience']}"
                        )
                        if patience_counter >= cfg["patience"]:
                            msg = (f"Early stopping @ step {global_step} | "
                                   f"best_val_loss={best_val_loss:.4f}")
                            py_logger.info(msg)
                            log_text(clearml_logger, msg)
                            run_final_test(
                                model, tokenizer, loaders, device, cfg,
                                val_data_path, clearml_logger, clearml_task, global_step
                            )
                            _save_history(train_losses, step_loss_history,
                                          best_val_loss, cfg, clearml_task)
                            if clearml_task:
                                clearml_task.close()
                            return

                    model.train()

                # ── Примеры генерации ──
                ex_every = cfg.get("log_examples_every", 500)
                if (ex_every > 0
                        and global_step % ex_every == 0
                        and os.path.exists(val_data_path)):
                    log_generation_examples(
                        model, tokenizer, val_data_path, device,
                        clearml_logger, global_step,
                        n=cfg.get("n_log_examples", 3),
                    )
                    model.train()

                # ── Периодический checkpoint ──
                if global_step % cfg["save_every_n_steps"] == 0:
                    save_checkpoint(
                        model, tokenizer, optimizer, scheduler,
                        cfg, global_step, epoch, best_val_loss,
                        cfg["output_dir"], tag="latest", scaler=scaler,
                    )

        # ── Конец эпохи ──
        elapsed = time.time() - t0
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        avg_epoch_ppl = perplexity(avg_epoch_loss)

        py_logger.info(
            f"--- Epoch {epoch}/{cfg['num_epochs']} done | "
            f"time={elapsed:.0f}s | "
            f"avg_loss={avg_epoch_loss:.4f} | "
            f"avg_ppl={avg_epoch_ppl:.2f} ---"
        )

        # ClearML: метрики по эпохам (отдельный title, ось X = номер эпохи)
        log_scalars_dict(clearml_logger, {
            "avg_loss":       avg_epoch_loss,
            "avg_perplexity": avg_epoch_ppl,
            "elapsed_sec":    elapsed,
        }, title="Epoch Summary", iteration=epoch)

        log_text(
            clearml_logger,
            f"Epoch {epoch} done | avg_loss={avg_epoch_loss:.4f} | "
            f"avg_ppl={avg_epoch_ppl:.2f} | time={elapsed:.0f}s"
        )

        train_losses.append(avg_epoch_loss)

        # Сохраняем checkpoint после каждой эпохи (для resume)
        save_checkpoint(
            model, tokenizer, optimizer, scheduler,
            cfg, global_step, epoch, best_val_loss,
            cfg["output_dir"], tag=f"epoch-{epoch}", scaler=scaler,
        )
        py_logger.info(f"Checkpoint эпохи {epoch} сохранён → checkpoint-epoch-{epoch}")

    # ── Финальная оценка ──
    run_final_test(
        model, tokenizer, loaders, device, cfg,
        val_data_path, clearml_logger, clearml_task, global_step
    )
    _save_history(train_losses, step_loss_history, best_val_loss, cfg, clearml_task)

    if clearml_task:
        clearml_task.close()

    py_logger.info("Обучение завершено!")


def _save_history(train_losses, step_loss_history, best_val_loss, cfg, clearml_task):
    history = {
        "train_losses_per_epoch": train_losses,
        "step_loss_history": step_loss_history,
        "best_val_loss": best_val_loss,
    }
    path = os.path.join(cfg["output_dir"], "training_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    upload_artifact(clearml_task, "training_history", path)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение GPT-2 на Ubuntu Dialogue Corpus")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None,
                        help="gpt2 | gpt2-medium | gpt2-large | distilgpt2")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Путь к checkpoint для продолжения: ./checkpoints/checkpoint-epoch-1")
    parser.add_argument("--no_clearml", action="store_true",
                        help="Отключить ClearML логирование")
    parser.add_argument("--clearml_project", type=str, default=None)
    parser.add_argument("--clearml_task_name", type=str, default=None)
    args = parser.parse_args()

    cfg = default_config()

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f))

    for key in ["data_dir", "model_name", "batch_size", "num_epochs",
                "learning_rate", "max_train_samples", "output_dir"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.resume_from:
        cfg["resume_from"] = args.resume_from
    if args.clearml_project:
        cfg["clearml_project"] = args.clearml_project
    if args.clearml_task_name:
        cfg["clearml_task"] = args.clearml_task_name
    if args.fp16:
        cfg["fp16"] = True
    if args.no_clearml:
        cfg["use_clearml"] = False

    py_logger.info(f"Конфигурация:\n{json.dumps(cfg, indent=2)}")
    train(cfg)