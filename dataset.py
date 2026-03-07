import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import Counter

# MAX_VOCAB = 15000

class DialogueDataset(Dataset):
    def __init__(self, csv_file, max_len=30, vocab=None):
        self.data = pd.read_csv(csv_file)

        # 🔹 Убираем NaN
        self.data = self.data.dropna(subset=["context", "response"])

        # 🔹 Приводим к строке (защита)
        self.data["context"] = self.data["context"].astype(str)
        self.data["response"] = self.data["response"].astype(str)

        self.max_len = max_len

        self.special_tokens = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }

        if vocab is None:
            self.vocab = self.build_vocab(self.data)
        else:
            self.vocab = vocab

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # -------------------------
    # Построение словаря
    # -------------------------
    def build_vocab(self, data, max_vocab=15000):
        counter = Counter()

        for _, row in data.iterrows():
            counter.update(row["context"].split())
            counter.update(row["response"].split())

        # сортируем по частоте
        most_common = counter.most_common(max_vocab)

        vocab = dict(self.special_tokens)

        for word, _ in most_common:
            if word not in vocab:
                vocab[word] = len(vocab)

        print(f"[INFO] Размер словаря (ограниченный): {len(vocab)}")
        return vocab

    # -------------------------
    # Текст → индексы
    # -------------------------
    def text_to_indices(self, text, add_sos_eos=False):
        tokens = text.split()
        indices = []

        if add_sos_eos:
            indices.append(self.vocab["<SOS>"])

        for token in tokens:
            indices.append(self.vocab.get(token, self.vocab["<UNK>"]))

        if add_sos_eos:
            indices.append(self.vocab["<EOS>"])

        # padding
        if len(indices) < self.max_len:
            indices += [self.vocab["<PAD>"]] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return torch.tensor(indices)

    # -------------------------
    # Длина датасета
    # -------------------------
    def __len__(self):
        return len(self.data)

    # -------------------------
    # Получение элемента
    # -------------------------
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        context_tensor = self.text_to_indices(row["context"])
        response_tensor = self.text_to_indices(row["response"], add_sos_eos=True)

        return context_tensor, response_tensor