"""
data_preprocessing.py
Обработка Ubuntu Dialogue Corpus из трёх файлов (dialog1, dialog2, dialog3).
Формат: folder, dialogId, date, from, to, text
"""

import pandas as pd
import re
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Загрузка и парсинг сырых CSV/TSV файлов
# ──────────────────────────────────────────────

COLUMNS = ["folder", "dialogId", "date", "from", "to", "text"]


def load_raw_files(data_dir: str, filenames: List[str] = None) -> pd.DataFrame:
    """Загружает несколько файлов диалогов и объединяет их."""
    if filenames is None:
        filenames = ["dialog1.csv", "dialog2.csv", "dialog3.csv"]

    dfs = []
    for fname in filenames:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"Файл не найден: {path}")
            continue
        try:
            df = pd.read_csv(
                path,
                header=None,
                names=COLUMNS,
                quotechar='"',
                escapechar="\\",
                on_bad_lines="skip",
                dtype=str,
            )
            df["source_file"] = fname
            dfs.append(df)
            logger.info(f"Загружен {fname}: {len(df)} строк")
        except Exception as e:
            logger.error(f"Ошибка при загрузке {fname}: {e}")

    if not dfs:
        raise ValueError("Ни один файл не был загружен. Проверьте пути.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Всего строк после объединения: {len(combined)}")
    return combined


# ──────────────────────────────────────────────
# 2. Очистка текста
# ──────────────────────────────────────────────

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NICK_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,20}[:,]\s*")  # "nick: " в начале
SPECIAL_CHARS = re.compile(r"[^\w\s\.,!\?;:\'\"\-\(\)/\\]")


def clean_text(text: str, remove_urls: bool = True, remove_nicks: bool = True) -> str:
    """Очищает одну реплику."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.strip()

    # Убираем URLs (или заменяем на токен)
    if remove_urls:
        text = URL_PATTERN.sub("[URL]", text)

    # Убираем «nick: » в начале
    if remove_nicks:
        text = NICK_PATTERN.sub("", text)

    # Убираем множественные пробелы/переносы
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────────────────────────────
# Фильтрация технических диалогов
# ──────────────────────────────────────────────

# Технические ключевые слова Ubuntu/Linux
TECH_KEYWORDS = {
    # Команды и пакеты
    "apt", "apt-get", "dpkg", "sudo", "install", "uninstall", "upgrade", "update",
    "terminal", "bash", "shell", "chmod", "chown", "grep", "sudo", "root", "mkdir",
    "cp", "mv", "rm", "ls", "cat", "echo", "export", "source", "wget", "curl",
    # Система
    "ubuntu", "linux", "kernel", "grub", "boot", "partition", "mount", "fstab",
    "systemd", "service", "daemon", "process", "pid", "kill", "cpu", "ram", "disk",
    "driver", "module", "udev", "dbus", "xorg", "display", "resolution",
    # Сеть
    "network", "wifi", "ethernet", "ip", "ssh", "firewall", "iptables", "dns",
    "router", "ping", "hostname", "interface", "ifconfig", "netstat",
    # Файловая система
    "directory", "folder", "file", "path", "home", "usr", "etc", "var", "tmp",
    "permission", "ownership", "symlink", "ext4", "ntfs", "fat32",
    # Ошибки и диагностика
    "error", "warning", "failed", "crash", "segfault", "log", "dmesg", "syslog",
    "debug", "config", "configure", "dependency", "conflict", "broken",
    # Приложения
    "firefox", "python", "java", "gcc", "make", "cmake", "git", "vim", "nano",
    "apache", "nginx", "mysql", "postgresql", "docker", "virtualbox",
}

# Паттерны явного оффтопа — реплики которые точно не техподдержка
OFFTOPIC_PATTERNS = re.compile(
    r"^(lol|haha|hehe|lmao|rofl|wtf|omg|wow|cool|nice|ok|okay|yeah|yep|nope|"
    r"hi|hello|hey|bye|goodbye|thanks|thank you|welcome|np|no problem|sure|"
    r"agreed|disagree|maybe|dunno|idk|afk|brb|gtg)\s*[!?.]*$",
    re.IGNORECASE,
)

# Паттерны вопросов/команд техподдержки в контексте
QUESTION_PATTERN = re.compile(
    r"\b(how|what|why|when|where|which|who|can|could|should|would|is|are|does|do|"
    r"install|fix|solve|configure|setup|enable|disable|run|start|stop|restart|"
    r"update|upgrade|remove|add|create|delete|change|edit|find|check|list|show|"
    r"get|set|use|need|want|help|problem|issue|error|fail)\b",
    re.IGNORECASE,
)


def is_technical_text(text: str) -> bool:
    """Возвращает True если текст содержит технический контент."""
    if not text:
        return False
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))
    # Достаточно одного технического слова
    return bool(words & TECH_KEYWORDS)


def is_offtopic(text: str) -> bool:
    """Возвращает True если реплика явный оффтоп."""
    return bool(OFFTOPIC_PATTERNS.match(text.strip()))


def is_technical_dialog(turns: List[dict], min_tech_ratio: float = 0.3) -> bool:
    """
    Возвращает True если диалог достаточно технический.
    min_tech_ratio — минимальная доля технических реплик в диалоге.
    """
    if not turns:
        return False
    texts = [t["text"] for t in turns if t["text"]]
    if not texts:
        return False
    tech_count = sum(1 for t in texts if is_technical_text(t))
    return (tech_count / len(texts)) >= min_tech_ratio


def is_addressed_response(turn: dict) -> bool:
    """Возвращает True если реплика адресована конкретному пользователю (поле 'to' заполнено)."""
    return bool(turn.get("to", "").strip()) and turn["to"].strip().lower() not in ("", "nan")


# ──────────────────────────────────────────────
# 3. Сборка диалогов
# ──────────────────────────────────────────────

def build_dialogs(df: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    Группирует строки по dialogId и сортирует по времени.
    Возвращает словарь {dialogId: [{"from":..., "to":..., "text":...}, ...]}
    """
    df = df.copy()
    df["text"] = df["text"].fillna("").apply(clean_text)
    df = df[df["text"].str.len() > 2]  # убираем пустые/слишком короткие

    # Парсим дату для сортировки
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    dialogs = defaultdict(list)
    for _, row in df.iterrows():
        dialogs[row["dialogId"]].append({
            "from": str(row["from"]) if pd.notna(row["from"]) else "",
            "to": str(row["to"]) if pd.notna(row["to"]) else "",
            "text": row["text"],
            "date": row["date_parsed"],
        })

    # Сортируем реплики внутри диалога по времени
    for did in dialogs:
        dialogs[did].sort(key=lambda x: x["date"] if pd.notna(x["date"]) else pd.Timestamp.min)

    logger.info(f"Собрано диалогов: {len(dialogs)}")
    return dict(dialogs)


# ──────────────────────────────────────────────
# 4. Формирование пар (контекст → ответ)
# ──────────────────────────────────────────────

def build_context_response_pairs(
    dialogs: Dict[str, List[dict]],
    context_window: int = 3,
    min_response_len: int = 5,
    max_response_len: int = 150,
    technical_filter: bool = True,
    min_tech_ratio: float = 0.3,
    require_addressed: bool = True,
) -> List[Dict]:
    """
    Создаёт пары (context, response) для обучения.

    Параметры фильтрации:
      technical_filter   — оставлять только технические диалоги
      min_tech_ratio     — минимальная доля технических реплик в диалоге (0.0-1.0)
      require_addressed  — брать только адресованные ответы (поле 'to' заполнено)
    """
    pairs = []
    skipped_offtopic = 0
    skipped_not_addressed = 0
    skipped_nontechnical = 0

    for dialog_id, turns in dialogs.items():
        # Фильтр 1: диалог должен быть достаточно техническим
        if technical_filter and not is_technical_dialog(turns, min_tech_ratio):
            skipped_nontechnical += 1
            continue

        texts = [t["text"] for t in turns if t["text"]]
        if len(texts) < 2:
            continue

        for i in range(1, len(turns)):
            turn = turns[i]
            response = turn["text"]
            if not response:
                continue

            # Фильтр 2: только адресованные ответы (to != "")
            if require_addressed and not is_addressed_response(turn):
                skipped_not_addressed += 1
                continue

            # Фильтр 3: ответ не должен быть оффтопом
            if is_offtopic(response):
                skipped_offtopic += 1
                continue

            # Фильтр 4: ответ должен содержать технический контент
            if technical_filter and not is_technical_text(response):
                skipped_offtopic += 1
                continue

            # Фильтр 5: длина ответа
            resp_words = response.split()
            if len(resp_words) < min_response_len or len(resp_words) > max_response_len:
                continue

            # Контекст: последние context_window реплик до ответа
            all_texts = [t["text"] for t in turns[:i] if t["text"]]
            context_turns = all_texts[-context_window:]
            context = " [SEP] ".join(context_turns)

            # Контекст тоже должен быть технически связан
            if technical_filter and not is_technical_text(context):
                continue

            pairs.append({
                "dialog_id": dialog_id,
                "context": context,
                "response": response,
                "context_len": len(context.split()),
                "response_len": len(resp_words),
            })

    logger.info(f"Сформировано пар (контекст, ответ): {len(pairs)}")
    logger.info(f"  Отфильтровано нетехнических диалогов: {skipped_nontechnical}")
    logger.info(f"  Отфильтровано неадресованных реплик: {skipped_not_addressed}")
    logger.info(f"  Отфильтровано оффтопа/нетехн. ответов: {skipped_offtopic}")
    return pairs


# ──────────────────────────────────────────────
# 5. Train / Val / Test сплит
# ──────────────────────────────────────────────

def split_pairs(
    pairs: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Делит пары на train/val/test по уникальным dialog_id (no data leakage)."""
    import random
    random.seed(seed)

    dialog_ids = list({p["dialog_id"] for p in pairs})
    random.shuffle(dialog_ids)

    n = len(dialog_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(dialog_ids[:n_train])
    val_ids = set(dialog_ids[n_train:n_train + n_val])
    test_ids = set(dialog_ids[n_train + n_val:])

    train = [p for p in pairs if p["dialog_id"] in train_ids]
    val = [p for p in pairs if p["dialog_id"] in val_ids]
    test = [p for p in pairs if p["dialog_id"] in test_ids]

    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# ──────────────────────────────────────────────
# 6. Сохранение
# ──────────────────────────────────────────────

def save_splits(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
    output_dir: str,
):
    """Сохраняет сплиты в JSONL формате."""
    os.makedirs(output_dir, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                # Убираем несериализуемые поля
                record = {k: v for k, v in item.items() if k not in ("date",)}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Сохранён {path} ({len(data)} примеров)")


def load_jsonl(path: str) -> List[Dict]:
    """Загружает JSONL файл."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ──────────────────────────────────────────────
# 7. Основной pipeline
# ──────────────────────────────────────────────

def run_preprocessing(
    data_dir: str,
    output_dir: str,
    filenames: List[str] = None,
    context_window: int = 3,
    max_pairs: int = None,
    technical_filter: bool = True,
    min_tech_ratio: float = 0.3,
    require_addressed: bool = True,
):
    """Полный pipeline предобработки."""
    df = load_raw_files(data_dir, filenames)
    dialogs = build_dialogs(df)

    pairs = build_context_response_pairs(
        dialogs,
        context_window=context_window,
        technical_filter=technical_filter,
        min_tech_ratio=min_tech_ratio,
        require_addressed=require_addressed,
    )

    if max_pairs is not None:
        pairs = pairs[:max_pairs]
        logger.info(f"Ограничено до {max_pairs} пар для отладки")

    train, val, test = split_pairs(pairs)
    save_splits(train, val, test, output_dir)

    stats = {
        "total_dialogs": len(dialogs),
        "total_pairs": len(pairs),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "avg_context_len": sum(p["context_len"] for p in pairs) / max(len(pairs), 1),
        "avg_response_len": sum(p["response_len"] for p in pairs) / max(len(pairs), 1),
        "filters": {
            "technical_filter": technical_filter,
            "min_tech_ratio": min_tech_ratio,
            "require_addressed": require_addressed,
        },
    }

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Статистика: {json.dumps(stats, indent=2)}")
    return stats


# ──────────────────────────────────────────────
# CLI запуск
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Предобработка Ubuntu Dialogue Corpus")
    parser.add_argument("--data_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./data/processed_filtered")
    parser.add_argument("--context_window", type=int, default=3)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--files", nargs="+",
                        default=["dialog1.csv", "dialog2.csv", "dialog3.csv"])
    # Фильтры
    parser.add_argument("--no_technical_filter", action="store_true",
                        help="Отключить технический фильтр (по умолчанию включён)")
    parser.add_argument("--min_tech_ratio", type=float, default=0.3,
                        help="Мин. доля техн. реплик в диалоге (default: 0.3)")
    parser.add_argument("--no_require_addressed", action="store_true",
                        help="Не требовать адресованных ответов")
    args = parser.parse_args()

    run_preprocessing(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        filenames=args.files,
        context_window=args.context_window,
        max_pairs=args.max_pairs,
        technical_filter=not args.no_technical_filter,
        min_tech_ratio=args.min_tech_ratio,
        require_addressed=not args.no_require_addressed,
    )
