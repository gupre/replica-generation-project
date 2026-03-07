import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

from dataset import DialogueDataset
from model import Encoder, Decoder, Seq2Seq

# ==========================
# ЛОГ ФУНКЦИЯ
# ==========================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Используем устройство: {device}")

# ==========================
# Гиперпараметры
# ==========================    
BATCH_SIZE = 32
EMBED_SIZE = 128
HIDDEN_SIZE = 128
EPOCHS = 15
LR = 0.0007

log(f"BATCH_SIZE={BATCH_SIZE}, EMBED={EMBED_SIZE}, HIDDEN={HIDDEN_SIZE}, LR={LR}")

# ==========================
# Данные
# ==========================
log("Загрузка датасета...")
train_dataset = DialogueDataset("data/train.csv", max_len=30)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=2,
)

vocab_size = len(train_dataset.vocab)

log(f"Размер датасета: {len(train_dataset)}")
log(f"Размер словаря: {vocab_size}")

# ==========================
# Модель
# ==========================
encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder, device).to(device)

total_params = sum(p.numel() for p in model.parameters())
log(f"Параметров в модели: {total_params:,}")

# ==========================
# Оптимизация
# ==========================
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ==========================
# Тренировка
# ==========================
log("Старт обучения...")

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    total_loss = 0

    for batch_idx, (context, response) in enumerate(train_loader):
        context = context.to(device)
        response = response.to(device)

        optimizer.zero_grad()

        output = model(context, response)
        output_dim = output.shape[-1]

        output = output[:, 1:].reshape(-1, output_dim)
        response = response[:, 1:].reshape(-1)

        loss = criterion(output, response)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            log(f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    epoch_time = time.time() - start_time

    log(f"Эпоха {epoch+1} завершена | "
        f"Средний Loss: {avg_loss:.4f} | "
        f"Время: {epoch_time:.2f} сек")

    if torch.cuda.is_available():
        log(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

log("Обучение завершено ✅")

torch.save({
    "model_state": model.state_dict(),
    "vocab": train_dataset.vocab
}, "chat_model.pt")

log("Модель сохранена в chat_model.pt")