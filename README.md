**Тема проекта** — генерация реплик для технической поддержки.  

##  Описание задачи

Задача — обучить языковую модель генерировать ответы на вопросы технической поддержки Ubuntu/Linux.  
Входные данные: история диалога (контекст). Выход: следующая реплика техподдержки.

**Датасет:** [Ubuntu Dialogue Corpus](https://arxiv.org/abs/1506.08909) — 26.8M строк реальных IRC-диалогов.  
**Модель:** GPT-2 (OpenAI, 117M параметров), fine-tuning через HuggingFace Transformers.  
**Мониторинг:** ClearML — логирование loss, метрик, примеров генерации.

## Особенности задачи

- входные и выходные последовательности имеют разную длину;
- словарь может быть большим;
- необходимо учитывать контекст сообщения;
- качество ответа сложно оценить автоматически.

---

# Оценка сложности проекта

Проект относится к задачам средней сложности.

---

##  Основные результаты

| Метрика | Значение |
|---|---|
| Test Perplexity | **6.33** |
| BLEU-4 (корпус) | 1.43 |
| ROUGE-L (корпус) | 11.84 |
| ROUGE-L (реальные вопросы) | 12.12 |
| Эпох обучения | 3 |
| Размер train выборки | 500 000 пар |

### Бенчмарк — корпусные данные (200 примеров)

| Модель | BLEU-4 | ROUGE-L | PPL | Score |
|---|---|---|---|---|
| **GPT-2 наш (fine-tuned)** | **1.43** | **11.84** | **5.56** | **97.8** |
| Mistral 7B (Ollama) | 1.18 | 12.65 | N/A | 64.0 |
| GPT-2 Original | 0.17 | 4.41 | 13.66 | 28.5 |

### Бенчмарк — реальные вопросы пользователей (60 вопросов)

| Модель | BLEU-4 | ROUGE-L | PPL | Score |
|---|---|---|---|---|
| Mistral 7B (Ollama) | **1.43** | **20.32** | N/A | 55.7 |
| **GPT-2 наш (fine-tuned)** | 0.39 | 12.12 | **22.98** | **71.7** |
| GPT-2 Original | 0.10 | 2.35 | 84.33 | 18.4 |

---

##  Структура проекта

```
├── data_preprocessing.py   # Загрузка, очистка, фильтрация датасета
├── dataset.py              # PyTorch Dataset + токенизатор
├── model.py                # GPT-2 wrapper (загрузка, генерация, сохранение)
├── train.py                # Цикл обучения (ClearML, checkpoints, early stopping)
├── inference.py            # Движок инференса
├── chat.py                 # Интерактивный CLI чат
├── evaluate.py             # Метрики BLEU, ROUGE, Perplexity
├── benchmark.py            # Бенчмарк на корпусных данных
├── benchmark_real.py       # Честный бенчмарк на реальных вопросах
├── config.json             # Конфигурация обучения
├── benchmark_config.json   # Конфигурация бенчмарка
├── real_user_benchmark.json# 60 реальных вопросов по Ubuntu (15 категорий)
├── requirements.txt        # Зависимости
└── docs/
    └── report.pdf          # Финальный отчёт (ClearML Reports)
```

---

##  Установка

### Требования
- Python 3.10+
- CUDA GPU (рекомендуется, минимум 6 GB VRAM)
- [Ollama](https://ollama.com) — для бенчмарка с Mistral 7B

### 1. Клонировать репозиторий

```bash
git clone https://github.com/gupre/generatereplice.git
cd generatereplice
```

### 2. Создать виртуальное окружение и установить зависимости

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Настроить ClearML (опционально)

```bash
clearml-init
# Ввести credentials с app.clear.ml
```

---

##  Запуск

### Шаг 1 — Предобработка данных

Положи файлы датасета в `./data/raw/` (`dialog1.csv`, `dialog2.csv`, `dialog3.csv`).

```bash
python data_preprocessing.py \
  --data_dir ./data/raw \
  --output_dir ./data/processed_filtered \
  --context_window 3
```

Результат: `data/processed_filtered/train.jsonl`, `val.jsonl`, `test.jsonl`

### Шаг 2 — Обучение

```bash
python train.py --config config.json
```

Для быстрого теста:
```bash
python train.py --config config.json --max_train_samples 5000 --num_epochs 1 --no_clearml
```

Продолжение обучения с чекпоинта:
```bash
python train.py --config config.json --num_epochs 3 --resume_from ./checkpoints/checkpoint-epoch-1
```

### Шаг 3 — Чат с обученной моделью

```bash
python chat.py --checkpoint ./checkpoints/checkpoint-best
```

Команды в чате: `/clear`, `/history`, `/config`, `/temp 0.7`, `/help`, `/quit`

### Шаг 4 — Оценка качества

```bash
python evaluate.py \
  --checkpoint ./checkpoints/checkpoint-best \
  --test_data ./data/processed_filtered/test.jsonl \
  --max_samples 1000
```

### Шаг 5 — Бенчмарк (сравнение моделей)

```bash
# Корпусный бенчмарк
python benchmark.py --config benchmark_config.json

# Честный бенчмарк на реальных вопросах
python benchmark_real.py
```

Для бенчмарка с Mistral 7B нужен запущенный Ollama:
```bash
ollama pull mistral:7b
ollama serve  # в отдельном терминале
```

---

##  Конфигурация обучения

Ключевые параметры в `config.json`:

```json
{
  "model_name": "gpt2",
  "batch_size": 32,
  "num_epochs": 3,
  "learning_rate": 3e-05,
  "max_train_samples": 500000,
  "fp16": true,
  "data_dir": "./data/processed_filtered"
}
```

| Параметр | Значение | Описание |
|---|---|---|
| `model_name` | `gpt2` | Базовая модель (117M параметров) |
| `batch_size` | 32 | Размер батча |
| `gradient_accumulation_steps` | 2 | Эффективный batch = 64 |
| `learning_rate` | 3e-5 | Cosine decay + linear warmup |
| `max_train_samples` | 500 000 | Из 2.9M отфильтрованных пар |
| `fp16` | true | Mixed precision на CUDA |
| `patience` | 3 | Early stopping по val loss |

---

##  Технологии

| Компонент | Технология |
|---|---|
| Модель | GPT-2 (HuggingFace Transformers) |
| Обучение | PyTorch + AdamW + Cosine LR |
| Мониторинг | ClearML |
| Данные | Ubuntu Dialogue Corpus |
| Сравнение | Mistral 7B через Ollama |
| Метрики | sacrebleu, rouge-score |

---

##  Мониторинг в ClearML

Проект: `UbuntuSupportAI`

Задачи:
- `GPT2_filtered` — основное обучение
- `Benchmark_GPT2_vs_Mistral7B` — корпусный бенчмарк
- `Benchmark_RealUsers_GPT2_vs_Mistral` — честный бенчмарк

Логируется: train/val loss, perplexity, learning rate, gradient norm, примеры генерации, сравнительные таблицы.

---

##  Дополнительные материалы

Папка `docs/`:
- `report.pdf` — финальный отчёт проекта (ClearML Reports)

---

##  Ключевые решения и выводы

**Фильтрация данных** — из 18.4M пар оставили 3.5M технических диалогов (убрали 80% шума). Это дало наибольший прирост качества.

**Два бенчмарка** — корпусный тест показал Score 97.8 (наш GPT-2 > Mistral), но он нечестный: модель тестировалась на своих данных. Честный тест на 60 реальных вопросах показал что Mistral 7B лучше по ROUGE-L (20.32 vs 12.12).

**Главный вывод** — специализированная маленькая модель (117M) в 4 раза быстрее и в 60 раз легче Mistral 7B. Для улучшения качества нужен актуальный датасет (Stack Overflow/Ask Ubuntu) вместо IRC-чатов 2004-2012.
