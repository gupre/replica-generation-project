"""
model.py
Модель генерации реплик на основе GPT-2 (fine-tuning).
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, AutoModelForCausalLM, GPT2Config
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Основная модель: GPT-2 для диалогов
# ──────────────────────────────────────────────

class SupportGPT2(nn.Module):
    """
    Обёртка над GPT2LMHeadModel для fine-tuning на диалогах техподдержки.
    Поддерживает:
      - полный fine-tuning
      - заморозку нижних слоёв (parameter-efficient)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        vocab_size: Optional[int] = None,
        freeze_layers: int = 0,
    ):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Расширяем embedding если добавили спецтокены
        if vocab_size and vocab_size != self.model.config.vocab_size:
            self.model.resize_token_embeddings(vocab_size)
            logger.info(f"Embeddings resized to {vocab_size}")

        # Заморозка нижних N трансформерных блоков
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Параметры: {total_params:,} всего, {trainable_params:,} обучаемых"
        )

    def _freeze_layers(self, n: int):
        """Замораживает embedding + первые n трансформерных блоков."""
        # Embeddings
        for param in self.model.transformer.wte.parameters():
            param.requires_grad = False
        for param in self.model.transformer.wpe.parameters():
            param.requires_grad = False

        # Нижние блоки
        for i in range(min(n, len(self.model.transformer.h))):
            for param in self.model.transformer.h[i].parameters():
                param.requires_grad = False

        logger.info(f"Заморожено {n} нижних блоков + embeddings")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def generate_response(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.3,
        num_return_sequences: int = 1,
        pad_token_id: int = 50256,
        eos_token_id: int = 50256,
    ) -> torch.Tensor:
        """
        Генерация ответа через nucleus sampling (top-p).
        Возвращает тензор с новыми токенами (без prompt).
        """
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        # Возвращаем только новые токены
        new_tokens = output[:, input_ids.shape[1]:]
        return new_tokens

    def save(self, path: str):
        self.model.save_pretrained(path)
        logger.info(f"Модель сохранена: {path}")

    @classmethod
    def load(cls, path: str, vocab_size: Optional[int] = None) -> "SupportGPT2":
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.model = GPT2LMHeadModel.from_pretrained(path)
        if vocab_size and vocab_size != instance.model.config.vocab_size:
            instance.model.resize_token_embeddings(vocab_size)
        return instance
