import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, id_to_vec):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)

        weights = torch.zeros(vocab_size, input_size)
        for i, vec in id_to_vec.items():
            weights[i] = vec
        self.embedding.weight = nn.Parameter(weights)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        return self.dropout(h[-1])

class DualEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        h = encoder.lstm.hidden_size
        self.M = nn.Parameter(torch.randn(h, h))

    def forward(self, context, response):
        c = self.encoder(context)
        r = self.encoder(response)
        score = torch.sum(torch.matmul(c, self.M) * r, dim=1)  # батч-матрица
        return score