"""
inference.py
Движок инференса: загружает обученную модель и генерирует ответы на вопросы техподдержки.
"""

import torch
import logging
from typing import List, Optional, Dict
from transformers import GPT2Tokenizer, AutoTokenizer

from model import SupportGPT2
from dataset import CONTEXT_TOKEN, RESPONSE_TOKEN, SPECIAL_TOKENS

logger = logging.getLogger(__name__)


class SupportBot:
    """
    Генератор реплик техподдержки.
    Принимает историю диалога, возвращает ответ.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.3,
        context_window: int = 3,
    ):
        # Определяем устройство
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info(f"Inference device: {self.device}")

        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        # Загрузка модели
        self.model = SupportGPT2.load(checkpoint_dir, vocab_size=len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Модель загружена из {checkpoint_dir}")

        # Параметры генерации
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.context_window = context_window

        # Специальные токены IDs
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def _build_prompt(self, history: List[str]) -> str:
        """Собирает промпт из истории диалога."""
        # Берём последние context_window реплик
        context_turns = history[-self.context_window:]
        context = " [SEP] ".join(context_turns)
        bos = self.tokenizer.bos_token or ""
        prompt = f"{bos}{CONTEXT_TOKEN} {context} {RESPONSE_TOKEN}"
        return prompt

    def generate(
        self,
        history: List[str],
        num_candidates: int = 1,
    ) -> List[str]:
        """
        Генерирует ответ(ы) на основе истории диалога.

        Args:
            history: список реплик (последняя = последний вопрос пользователя)
            num_candidates: сколько вариантов ответа вернуть

        Returns:
            Список строк-ответов
        """
        prompt = self._build_prompt(history)

        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=200,  # оставляем место для ответа
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Репликация для num_candidates
        if num_candidates > 1:
            input_ids = input_ids.repeat(num_candidates, 1)
            attention_mask = attention_mask.repeat(num_candidates, 1)

        new_tokens = self.model.generate_response(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            num_return_sequences=1,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        responses = []
        for i in range(new_tokens.shape[0]):
            text = self.tokenizer.decode(new_tokens[i], skip_special_tokens=True).strip()
            # Убираем хвост после первого переноса строки / повторяющийся контекст
            text = text.split("\n")[0].strip()
            if text:
                responses.append(text)

        return responses if responses else ["I'm not sure how to help with that."]

    def single_turn(self, user_message: str) -> str:
        """Быстрый однооборотный ответ (без истории)."""
        return self.generate([user_message], num_candidates=1)[0]

    def update_config(self, **kwargs):
        """Обновляет параметры генерации на лету."""
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
                logger.debug(f"Updated {key}={val}")
