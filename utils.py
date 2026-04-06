import pandas as pd
import numpy as np
import torch

# =========================
# DATAFRAME
# =========================
def create_dataframe(csvfile):
    df = pd.read_csv(csvfile)
    df = df.dropna()
    return df

# =========================
# VOCAB (ограниченный)
# =========================
def create_vocab(dataframe, max_vocab=30000):
    print("🧠 Building vocab...")

    word_freq = {}

    for i, (_, row) in enumerate(dataframe.iterrows()):
        if i % 10000 == 0:
            print(f"   processed {i}")

        words = str(row["Context"]).split() + str(row["Utterance"]).split()

        for w in words:
            w = w.lower()
            word_freq[w] = word_freq.get(w, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    vocab = ["<UNK>"] + [w for w, _ in sorted_words[:max_vocab]]

    print("✅ vocab size:", len(vocab))
    return vocab

def create_word_to_id(vocab):
    return {w: i for i, w in enumerate(vocab)}

# =========================
# EMBEDDINGS
# =========================
def create_id_to_vec(word_to_id, glovefile):
    id_to_vec = {}
    vector_dim = None

    print("🔤 Loading GloVe...")

    with open(glovefile, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"   read {i} lines")

            parts = line.split()
            word = parts[0]
            vec = np.array(parts[1:], dtype='float32')

            if vector_dim is None:
                vector_dim = len(vec)

            if word in word_to_id:
                id_to_vec[word_to_id[word]] = torch.FloatTensor(vec)

    for word, idx in word_to_id.items():
        if idx not in id_to_vec:
            id_to_vec[idx] = torch.randn(vector_dim) * 0.01

    print("✅ embeddings ready")
    return id_to_vec, vector_dim

# =========================
# DATASET
# =========================
class DialogueDataset(torch.utils.data.Dataset):
    def __init__(self, df, word_to_id):
        self.df = df
        self.word_to_id = word_to_id

    def __len__(self):
        return len(self.df)

    def to_ids(self, text):
        # 🔥 ограничение длины
        return [self.word_to_id.get(w, 0) for w in str(text).split()[:40]]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        c = self.to_ids(row["Context"])
        r = self.to_ids(row["Utterance"])
        y = float(row["Label"])

        return c, r, y

# =========================
# COLLATE
# =========================
def collate_fn(batch):
    contexts, responses, labels = zip(*batch)

    max_c = max(len(x) for x in contexts)
    max_r = max(len(x) for x in responses)

    c_tensor = torch.zeros(len(batch), max_c).long()
    r_tensor = torch.zeros(len(batch), max_r).long()

    for i in range(len(batch)):
        c_tensor[i, :len(contexts[i])] = torch.LongTensor(contexts[i])
        r_tensor[i, :len(responses[i])] = torch.LongTensor(responses[i])

    y = torch.FloatTensor(labels).unsqueeze(1)

    return c_tensor, r_tensor, y