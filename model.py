import torch
import torch.nn as nn

# ==========================
# Encoder
# ==========================
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


# ==========================
# Decoder
# ==========================
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell


# ==========================
# Seq2Seq
# ==========================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        input_token = target[:, 0]  # <SOS>

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output

            top1 = output.argmax(1)

            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            input_token = target[:, t] if use_teacher else top1

        return outputs