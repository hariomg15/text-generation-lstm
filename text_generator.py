import torch
import torch.nn as nn
import torch.optim as optim
import requests
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 1. DOWNLOAD DATASET

print("Downloading dataset...")
url = "https://www.gutenberg.org/files/11/11-0.txt"
text = requests.get(url).text


# 2. CLEAN + PREPROCESS TEXT

print("Preprocessing text...")

text = text.lower()
text = re.sub(r'[^a-zA-Z0-9\s.,;?!]', ' ', text)
text = re.sub(r'\s+', ' ', text)

chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

vocab_size = len(chars)
print("Vocab size:", vocab_size)

# Sequence length
seq_length = 100

# Encode full text
encoded_text = np.array([char_to_idx[c] for c in text])


# 3. CREATE DATASET CLASS

class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        X = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(X), torch.tensor(y)


dataset = TextDataset(encoded_text, seq_length)
loader = DataLoader(dataset, batch_size=64, shuffle=True)



# 4. LSTM MODEL

class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, prev_state=None):
        x = self.embed(x)
        if prev_state is None:
            output, state = self.lstm(x)
        else:
            output, state = self.lstm(x, prev_state)
        logits = self.fc(output)
        return logits, state


model = LSTMGenerator(vocab_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)



# 5. TRAINING LOOP

print("Training model... (6–8 minutes approx)")
epochs = 8  

for epoch in range(epochs):
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits.transpose(1, 2), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss:.4f}")



# 6. TEXT GENERATION FUNCTION

def generate_text(model, start_text, length=300):
    model.eval()
    chars_input = torch.tensor([char_to_idx[c] for c in start_text]).unsqueeze(0).to(device)

    generated = start_text
    state = None

    for _ in range(length):
        logits, state = model(chars_input, state)
        prob = torch.softmax(logits[:, -1, :], dim=-1).detach().cpu().numpy().ravel()
        next_idx = np.random.choice(len(prob), p=prob)
        next_char = idx_to_char[next_idx]
        generated += next_char
        chars_input = torch.tensor([[next_idx]]).to(device)

    return generated



# 7. GENERATE SAMPLE TEXT

print("\nGenerating sample text...\n")
sample = generate_text(model, "alice was", 400)
print(sample)

# Save output
with open("generated_output.txt", "w") as f:
    f.write(sample)

print("\nGenerated text saved as generated_output.txt")
