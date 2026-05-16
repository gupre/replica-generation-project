"""
dataset.py
PyTorch Dataset для обучения GPT-2 / DialoGPT на парах (контекст, ответ).

DialoGPT использует нативный формат диалога через EOS-токены между репликами:
  turn1 <|endoftext|> turn2 <|endoftext|> ... response <|endoftext|>
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AutoTokenizer
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Специальные токены для GPT-2 (стандартный режим)
SPECIAL_TOKENS_GPT2 = {
    "bos_token": "<|startoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "<|pad|>",
    "additional_special_tokens": ["<|context|>", "<|response|>", "[SEP]"],
}

# DialoGPT использует только EOS как разделитель — не добавляем лишних токенов
SPECIAL_TOKENS_DIALOGPT = {
    "pad_token": "<|pad|>",
}

CONTEXT_TOKEN = "<|context|>"
RESPONSE_TOKEN = "<|response|>"


def is_dialogpt(model_name: str) -> bool:
    return "dialogpt" in model_name.lower()


# ──────────────────────────────────────────────
# Токенизатор
# ──────────────────────────────────────────────

def build_tokenizer(model_name: str = "gpt2") -> GPT2Tokenizer:
    """Загружает токенизатор и добавляет специальные токены."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if is_dialogpt(model_name):
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DIALOGPT)
    else:
        tokenizer.add_special_tokens(SPECIAL_TOKENS_GPT2)
    logger.info(f"Токенизатор загружен. Vocab size: {len(tokenizer)}")
    return tokenizer


def encode_pair(
    context: str,
    response: str,
    tokenizer: GPT2Tokenizer,
    max_length: int = 256,
    dialogpt_mode: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Кодирует пару (контекст, ответ).

    GPT-2 формат:
      <|startoftext|><|context|> CTX <|response|> RESP <|endoftext|>

    DialoGPT формат (нативный):
      turn1 <|endoftext|> turn2 <|endoftext|> response <|endoftext|>
      Loss считается только на токенах response.
    """
    eos = tokenizer.eos_token  # <|endoftext|>

    if dialogpt_mode:
        # Разбиваем контекст обратно на отдельные реплики
        turns = context.split(" [SEP] ")
        # Собираем в формат DialoGPT
        dialog_parts = [t.strip() + eos for t in turns if t.strip()]
        context_str = "".join(dialog_parts)
        response_str = response.strip() + eos
        full_text = context_str + response_str

        encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()

        # Маскируем контекст — учимся только на response
        context_enc = tokenizer(
            context_str,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        context_len = context_enc["input_ids"].shape[1]
        labels[:context_len] = -100

    else:
        # Стандартный GPT-2 формат
        bos = tokenizer.bos_token or ""
        full_text = f"{bos}{CONTEXT_TOKEN} {context} {RESPONSE_TOKEN} {response}{eos}"
        encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        response_token_id = tokenizer.convert_tokens_to_ids(RESPONSE_TOKEN)
        labels = input_ids.clone()
        response_start = (input_ids == response_token_id).nonzero(as_tuple=True)[0]
        if len(response_start) > 0:
            labels[: response_start[0] + 1] = -100

    # PAD → -100
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class UbuntuDialogDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: GPT2Tokenizer,
        max_length: int = 256,
        max_samples: Optional[int] = None,
        dialogpt_mode: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dialogpt_mode = dialogpt_mode
        self.data = self._load(jsonl_path, max_samples)
        logger.info(f"Загружено {len(self.data)} примеров из {jsonl_path}")

    def _load(self, path: str, max_samples: Optional[int]) -> List[Dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return encode_pair(
            context=item["context"],
            response=item["response"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            dialogpt_mode=self.dialogpt_mode,
        )


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    tokenizer: GPT2Tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = 2000,
    max_test_samples: Optional[int] = 2000,
    num_workers: int = 0,
    dialogpt_mode: bool = False,
) -> Dict[str, DataLoader]:

    limits = {"train": max_train_samples, "val": max_val_samples, "test": max_test_samples}
    loaders = {}

    for split in ["train", "val", "test"]:
        path = f"{data_dir}/{split}.jsonl"
        ds = UbuntuDialogDataset(
            path, tokenizer,
            max_length=max_length,
            max_samples=limits[split],
            dialogpt_mode=dialogpt_mode,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info(f"{split} loader: {len(ds)} samples, {len(loaders[split])} batches")

    return loaders

