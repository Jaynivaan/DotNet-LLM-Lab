"""
gs

Lesson:11
=========

Topic: Attention Scores
Goal: understand how every word decides which other words deserve attention.
==============================
"""

import torch

torch.manual_seed(7)

words = [
    "the",
    "witness",
    "is",
    "pure",
    "awareness"   
]

embedding_size = 4

embeddings = torch.rand((len(words), embedding_size))

query_weights = torch.rand(embedding_size,embedding_size)
key_weights = torch.rand(embedding_size, embedding_size)

queries = embeddings @ query_weights
keys = embeddings @ key_weights

attention_scores = queries @ keys.T

print("=" * 69)
print("words")
print("=" * 69)

for i, word in enumerate(words):
    print(f"{i} -> {word}")

print("\nQuery vectors")
print("=" * 69)
print(queries)

print("\nKey Vectors")
print("=" * 69)
print(keys)

print("\nAttention Scores")
print("=" * 69)
print(attention_scores)

print("\nAttention Score shape")
print("=" * 69) 
print(attention_scores.shape)

print("*" * 69)
print("-" * 69)
print("*" * 69)
