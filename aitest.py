  import torch
import torch.nn as nn
import torch.optim as optim

# Data
corpus = ["I like cats", "I like dogs", "Dogs and cats are great"]
tokens = list(set(word for sentence in corpus for word in sentence.lower().split()))
word_to_idx = {word: idx for idx, word in enumerate(tokens)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
tokenized_corpus = [[word_to_idx[word.lower()] for word in sentence.split()] for sentence in corpus]

# Model definition
class TinyLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=10):
        super(TinyLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim * 2, vocab_size)
        self.activation_function = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear(embeds)
        log_probs = self.activation_function(out)
        return log_probs

# Training data
train_data = [(sentence[i:i+2], sentence[i+2]) for sentence in tokenized_corpus for i in range(len(sentence) - 2)]

# Model training
model = TinyLM(len(tokens))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    total_loss = 0
    for context, target in train_data:
        context_tensor = torch.tensor(context, dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_tensor)
        loss = loss_function(log_probs, torch.tensor([target], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# Prediction function
def predict_next_word(context):
    with torch.no_grad():
        context_words = [word.lower() for word in context.split() if word.lower() in word_to_idx]
        if len(context_words) < 2:
            return "Not enough known words in the input. Please enter two words from the corpus."
        context_tensor = torch.tensor([word_to_idx[word] for word in context_words[:2]], dtype=torch.long)
        log_probs = model(context_tensor)
        prediction = torch.argmax(log_probs).item()
        return idx_to_word[prediction]

# Accepting user input
user_input = input('Please enter two words: ')
print(f"Predicted next word: {predict_next_word(user_input)}")

