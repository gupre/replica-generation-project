import pandas as pd
import glob
import re
import csv
import random

MAX_LEN = 20
TRAIN_RATIO = 0.9
CHUNK_SIZE = 10000
MAX_PAIRS = 50000

def clean_text(text):
    if not isinstance(text, str):
        return None
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================
# Сбор пар
# ==========================

files = glob.glob("data/*.csv")
random.shuffle(files)

print(f"[INFO] Найдено {len(files)} файлов")

pairs = []

for file in files:
    print(f"[INFO] Обработка файла {file}")

    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):
        chunk = chunk.sort_values(by=["folder", "dialogueID", "date"])

        for dialogue_id, group in chunk.groupby("dialogueID"):
            group = group.reset_index(drop=True)

            for i in range(1, len(group)):
                context = clean_text(group.loc[i-1, "text"])
                response = clean_text(group.loc[i, "text"])

                if not context or not response:
                    continue

                if len(context.split()) > MAX_LEN:
                    continue
                if len(response.split()) > MAX_LEN:
                    continue

                pairs.append((context, response))

                if len(pairs) >= MAX_PAIRS:
                    break

            if len(pairs) >= MAX_PAIRS:
                break

        if len(pairs) >= MAX_PAIRS:
            break

    if len(pairs) >= MAX_PAIRS:
        break


print(f"[INFO] Всего собрано пар: {len(pairs)}")

# ==========================
# Перемешивание и split
# ==========================

random.shuffle(pairs)

split_index = int(len(pairs) * TRAIN_RATIO)
train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

print(f"[INFO] Train: {len(train_pairs)}")
print(f"[INFO] Val: {len(val_pairs)}")


# ==========================
# Сохранение
# ==========================

with open("data/train.csv", "w", newline="", encoding="utf-8") as train_file:
    writer = csv.writer(train_file)
    writer.writerow(["context", "response"])
    writer.writerows(train_pairs)

with open("data/val.csv", "w", newline="", encoding="utf-8") as val_file:
    writer = csv.writer(val_file)
    writer.writerow(["context", "response"])
    writer.writerows(val_pairs)

print("[INFO] Готово ✅")