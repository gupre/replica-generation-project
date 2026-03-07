import torch
from model import Encoder, Decoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Загрузка модели
# ==========================
checkpoint = torch.load("chat_model.pt", map_location=device)
vocab = checkpoint["vocab"]

inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

EMBED_SIZE = 128
HIDDEN_SIZE = 128

encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder, device).to(device)

model.load_state_dict(checkpoint["model_state"])
model.eval()

# ==========================
# Текст → индексы
# ==========================
def text_to_tensor(text, max_len=30):
    tokens = text.split()
    indices = []

    for token in tokens:
        indices.append(vocab.get(token, vocab["<UNK>"]))

    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return torch.tensor(indices).unsqueeze(0).to(device)


# ==========================
# Генерация ответа
# ==========================
def generate_reply(text, max_len=30):
    with torch.no_grad():
        source = text_to_tensor(text)

        hidden, cell = model.encoder(source)

        input_token = torch.tensor([vocab["<SOS>"]]).to(device)

        result_tokens = []

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell)

            temperature = 0.6
            top_k = 20

            probs = torch.softmax(output / temperature, dim=1)
            top_probs, top_indices = torch.topk(probs, top_k)

            top_probs = top_probs.squeeze(0)
            top_indices = top_indices.squeeze(0)

            top_probs = top_probs / top_probs.sum()

            token_id = top_indices[torch.multinomial(top_probs, 1)].item()

            if token_id == vocab["<EOS>"]:
                break

            result_tokens.append(inv_vocab.get(token_id, "<UNK>"))

            input_token = torch.tensor([token_id]).to(device)

        return " ".join(result_tokens)


# ==========================
# Чат
# ==========================
print("Модель загружена. Введите сообщение (exit для выхода).")

while True:
    user_input = input("Вы: ")

    if user_input.lower() == "exit":
        break

    reply = generate_reply(user_input)
    print("Бот:", reply)