import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# Read the file
with open('data.txt', 'r') as file:
    corpus = file.read().splitlines()

# Tokenize and build vocabulary
tokens = [word for sentence in corpus for word in sentence.split()]
vocab = Counter(tokens)
vocab_size = len(vocab)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Prepare sequences for training
sequences = [[word_to_idx[word] for word in sentence.split()] for sentence in corpus]
sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
padded_sequences = pad_sequence(sequences, batch_first=True)

# RNN Model Definition
class ContextAwareRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(ContextAwareRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, hidden=None):
        embeds = self.embedding(inputs)
        output, hidden = self.rnn(embeds, hidden)
        output = self.fc(output)
        log_probs = self.log_softmax(output)
        return log_probs, hidden

# Hyperparameters
embed_size = 64
hidden_size = 128
epochs = 10

# Model, Loss, Optimizer
model = ContextAwareRNN(vocab_size, embed_size, hidden_size)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# ... (same as previous code until the training loop)

# Training loop with more epochs
epochs = 30  # Increase the number of epochs for more training passes
for epoch in range(epochs):
    for sequence in padded_sequences:
        inputs = sequence[:-1].unsqueeze(0)
        targets = sequence[1:].unsqueeze(0)
        model.zero_grad()
        log_probs, _ = model(inputs)
        loss = loss_function(log_probs.squeeze(0), targets.squeeze(0))
        loss.backward()
        optimizer.step()

# Prediction Function to Generate Longer Texts
def generate_text(context, length=10):
    with torch.no_grad():
        generated_text = []
        for _ in range(length):
            context_tokens = [word_to_idx[word] for word in context.split()]
            context_tensor = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0)
            log_probs, _ = model(context_tensor)
            prediction = torch.argmax(log_probs[0, -1]).item()
            predicted_word = idx_to_word[prediction]
            generated_text.append(predicted_word)
            context += ' ' + predicted_word
        return ' '.join(generated_text)

# Infinite Loop for User Input
while True:
    user_input = input("Please enter some words from the corpus (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print(f"Generated text: {generate_text(user_input)}")
