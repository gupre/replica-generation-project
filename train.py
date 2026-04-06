import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os

from utils import *
from model import *

# =========================
# CONFIG
# =========================
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "checkpoint.pth"
BEST_MODEL_PATH = "best_model.pth"

USE_CLEARML = True

# =========================
# CLEARML
# =========================
if USE_CLEARML:
    from clearml import Task

    task = Task.init(
        project_name="DualEncoderChatbot",
        task_name="training",
    )
    logger = task.get_logger()

print("🚀 Device:", DEVICE)

# =========================
# DATA
# =========================
print("\n📂 Loading data...")
train_df = create_dataframe("training.csv")
val_df = create_dataframe("validation.csv")

# 🔥 ускорение (можешь убрать позже)
train_df = train_df.sample(frac=0.5)

print("Train size:", train_df.shape)
print("Val size:", val_df.shape)

# =========================
# VOCAB
# =========================
vocab = create_vocab(train_df, max_vocab=30000)
word_to_id = create_word_to_id(vocab)

# =========================
# EMBEDDINGS
# =========================
id_to_vec, emb_dim = create_id_to_vec(word_to_id, "glove.6B.100d.txt")

# =========================
# DATASET
# =========================
train_dataset = DialogueDataset(train_df, word_to_id)
val_dataset = DialogueDataset(val_df, word_to_id)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print("📦 Train batches:", len(train_loader))
print("📦 Val batches:", len(val_loader))

# =========================
# MODEL
# =========================
encoder = Encoder(emb_dim, 64, len(vocab), id_to_vec)
model = DualEncoder(encoder).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

# =========================
# LOAD CHECKPOINT
# =========================
start_epoch = 0
best_val_loss = float("inf")

if os.path.exists(CHECKPOINT_PATH):
    print("\n🔁 Loading checkpoint...")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"✅ Resuming from epoch {start_epoch}")

# =========================
# VALIDATION
# =========================
def evaluate():
    model.eval()
    total_loss = 0
    total = 0

    with torch.no_grad():
        for c, r, y in val_loader:
            c, r, y = c.to(DEVICE), r.to(DEVICE), y.to(DEVICE)

            score = model(c, r)
            y = y.squeeze(1)

            loss = loss_fn(score, y)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

    return total_loss / total

# =========================
# TRAIN
# =========================
def train():
    global best_val_loss

    print("\n🔥 Training started")

    for epoch in range(start_epoch, EPOCHS):
        model.train()

        epoch_start = time.time()

        total_loss = 0
        correct = 0
        total = 0

        total_batches = len(train_loader)

        print(f"\n🚀 Epoch {epoch} started | total batches: {total_batches}")

        for i, (c, r, y) in enumerate(train_loader):

            c, r, y = c.to(DEVICE), r.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            score = model(c, r)
            y = y.squeeze(1)

            loss = loss_fn(score, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)

            pred = (torch.sigmoid(score) >= 0.5).float()
            batch_acc = (pred == y).float().mean().item()

            correct += (pred == y).sum().item()
            total += y.size(0)

            # =========================
            # LOG BATCH
            # =========================
            if i % 50 == 0:
                percent = (i / total_batches) * 100

                print(
                    f"[Epoch {epoch}] "
                    f"{i}/{total_batches} ({percent:.2f}%) | "
                    f"Loss: {loss.item():.4f} | Acc: {batch_acc:.4f}"
                )

                if USE_CLEARML:
                    global_step = epoch * total_batches + i

                    logger.report_scalar("batch_loss", "train", loss.item(), global_step)
                    logger.report_scalar("batch_acc", "train", batch_acc, global_step)

        # =========================
        # EPOCH METRICS
        # =========================
        epoch_loss = total_loss / total
        epoch_acc = correct / total

        val_loss = evaluate()

        print(f"\n⏱ Epoch time: {time.time()-epoch_start:.2f}s")

        print(f"\n📊 Epoch {epoch}")
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # =========================
        # CLEARML EPOCH LOGS
        # =========================
        if USE_CLEARML:
            logger.report_scalar("Loss", "train", epoch_loss, epoch)
            logger.report_scalar("Loss", "val", val_loss, epoch)
            logger.report_scalar("Accuracy", "train", epoch_acc, epoch)

        # =========================
        # SAVE BEST MODEL
        # =========================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("🏆 Best model saved")

        # =========================
        # SAVE CHECKPOINT
        # =========================
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, CHECKPOINT_PATH)

        print("💾 Checkpoint saved")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()