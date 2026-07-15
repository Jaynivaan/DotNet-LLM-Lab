"""
gs


Lesson: 07
==========

Topic: Self Attention
Goal: Understand how every token decides which other tokens are important before making a prediction.

A transformer converts tokenIds into embeddings.  Each embedding compares itself with every other embedding using Query, key and Value vectors. 
The resulting attention weights determine how much information each token receives before passing through the rest of the network.
"""

import torch

torch.manual_seed(7)

tokens = ["The", "cat", "sat"]

embeddings = torch.tensor(
    [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],        
    ]
)

query = embeddings[2]
keys = embeddings
values = embeddings

scores = torch.matmul(keys, query)

attention_weights = torch.softmax(scores, dim=0)

context_vector = torch.matmul(attention_weights, values)

print("=" * 69)
print("Tokens")
print("=" * 69)
print (tokens)

print("\nEmbeddings")
print("=" * 69)
print(embeddings)

print("\nQueryToken")
print("=" * 69)
print(tokens[2])

print("\nAttention Scores")
print("=" * 69)

for token, score in zip(tokens, scores):
    print(f"{token:10} -> {score.item():.4f}")

print("\nAttention Weights")
print("=" * 69)

for token, weight in zip(tokens, attention_weights):
    print(f"{token:10} -> {weight.item():.4f}")

print("\nContext Vector")
print("=" * 69)
print(context_vector)

